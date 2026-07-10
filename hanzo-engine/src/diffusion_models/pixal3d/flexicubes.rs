//! FlexiCubes dual-marching-cubes surface extraction (nvdiffrec FlexiCubes, inference path).
//!
//! Turns the mesh decoder's per-cube 101-channel features into a textured triangle mesh. The SDF grid
//! is 1 (outside) everywhere except at active-voxel corners, so a cube can only cross the surface if it
//! touches an sdf<0 vertex: we enumerate exactly those cubes instead of the dense res^3 grid. Cubes are
//! processed in the reference's (z,y,x) C-order so the 4-cube quad winding matches. Dual vertices are
//! beta-weighted means of alpha-weighted edge zero-crossings; colours interpolate the same way.

use std::collections::HashMap;

use hanzo_3d::{Mesh, Vec3};
use hanzo_ml::{Result, Tensor};

use super::flexicubes_tables::{CHECK, DMC, NUM_VD};

const CUBE_CORNERS: [[i32; 3]; 8] = [
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
    [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
];
// 12 edges, each a pair of corner indices (fixed orientation so shared edges match across cubes).
const CUBE_EDGES: [usize; 24] = [
    0, 1, 1, 5, 4, 5, 0, 4, 2, 3, 3, 7, 6, 7, 2, 6, 2, 0, 3, 1, 7, 5, 6, 4,
];
const WEIGHT_SCALE: f32 = 0.99;

/// A decoded mesh with per-vertex RGB colour (glTF COLOR_0).
pub struct TexturedMesh {
    pub mesh: Mesh,
    pub colors: Vec<[f32; 3]>,
}

#[derive(Clone)]
struct Vert {
    sdf: f32,
    x: [f32; 3],     // deformed position in [-0.5, 0.5]
    color: [f32; 6], // post-sigmoid
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn vid(c: [i32; 3], resv: i32) -> u64 {
    ((c[0] as i64 * resv as i64 + c[1] as i64) * resv as i64 + c[2] as i64) as u64
}

/// alpha-weighted zero crossing along an edge: interp of x0,x1 by weights w=(s0*a0, s1*a1).
/// ue = (x0*w1 - x1*w0)/(w1 - w0) per TRELLIS `_linear_interp`.
fn interp3(x0: [f32; 3], x1: [f32; 3], w0: f32, w1: f32) -> [f32; 3] {
    let d = w1 - w0;
    std::array::from_fn(|k| (x0[k] * w1 - x1[k] * w0) / d)
}

fn interp6(c0: &[f32; 6], c1: &[f32; 6], w0: f32, w1: f32) -> [f32; 6] {
    let d = w1 - w0;
    std::array::from_fn(|k| (c0[k] * w1 - c1[k] * w0) / d)
}

/// Extract the FlexiCubes surface from the decoder output. `coords`/`feats` are the active voxels at
/// resolution `res` (= decoder resolution * 4) with `feats` shaped [Nv, 101].
pub fn extract(coords: &[[i32; 3]], feats: &Tensor, res: i32) -> Result<TexturedMesh> {
    let nv = coords.len();
    let f = feats.to_vec2::<f32>()?; // [Nv, 101]
    let resv = res + 1;
    let sdf_bias = -1.0 / res as f32;
    let inv = (1.0 - 1e-8) / (2.0 * res as f32);

    // sparse_cube2verts: accumulate per-vertex sdf/deform/color means over the voxel corners.
    struct Acc {
        sdf: f32,
        deform: [f32; 3],
        color: [f32; 6],
        n: f32,
    }
    let mut vacc: HashMap<[i32; 3], Acc> = HashMap::new();
    for (v, c) in coords.iter().enumerate() {
        let row = &f[v];
        for j in 0..8 {
            let vc = [c[0] + CUBE_CORNERS[j][0], c[1] + CUBE_CORNERS[j][1], c[2] + CUBE_CORNERS[j][2]];
            let a = vacc.entry(vc).or_insert(Acc { sdf: 0.0, deform: [0.0; 3], color: [0.0; 6], n: 0.0 });
            a.sdf += row[j] + sdf_bias;
            for k in 0..3 {
                a.deform[k] += row[8 + j * 3 + k];
            }
            for k in 0..6 {
                a.color[k] += row[53 + j * 6 + k];
            }
            a.n += 1.0;
        }
    }
    let verts: HashMap<[i32; 3], Vert> = vacc
        .into_iter()
        .map(|(vc, a)| {
            let sdf = a.sdf / a.n;
            let x = std::array::from_fn(|k| {
                let d = a.deform[k] / a.n;
                vc[k] as f32 / res as f32 - 0.5 + inv * d.tanh()
            });
            let color = std::array::from_fn(|k| sigmoid(a.color[k] / a.n));
            (vc, Vert { sdf, x, color })
        })
        .collect();

    // per-voxel weights (beta[12], alpha[8], gamma) for cubes at active positions.
    let mut wmap: HashMap<[i32; 3], [f32; 21]> = HashMap::with_capacity(nv);
    for (v, c) in coords.iter().enumerate() {
        wmap.insert(*c, std::array::from_fn(|k| f[v][32 + k]));
    }

    let sdf_at = |c: [i32; 3]| verts.get(&c).map_or(1.0, |v| v.sdf);
    let in_range = |p: [i32; 3]| (0..res).contains(&p[0]) && (0..res).contains(&p[1]) && (0..res).contains(&p[2]);

    // candidate surf cubes: those touching an sdf<0 vertex.
    let mut cand: Vec<[i32; 3]> = Vec::new();
    let mut seen: HashMap<[i32; 3], ()> = HashMap::new();
    for (vc, v) in verts.iter() {
        if v.sdf >= 0.0 {
            continue;
        }
        for cc in &CUBE_CORNERS {
            let p = [vc[0] - cc[0], vc[1] - cc[1], vc[2] - cc[2]];
            if in_range(p) && seen.insert(p, ()).is_none() {
                cand.push(p);
            }
        }
    }

    // keep genuine surf cubes (mixed occupancy), sorted (z,y,x) = reference reg_c order.
    struct Cube {
        p: [i32; 3],
        occ: [bool; 8],
        case: usize,
        beta: [f32; 12],
        alpha: [f32; 8],
        gamma: f32,
    }
    let norm_beta = |r: f32| r.tanh() * WEIGHT_SCALE + 1.0;
    let norm_gamma = |r: f32| sigmoid(r) * WEIGHT_SCALE + (1.0 - WEIGHT_SCALE) / 2.0;
    let mut cubes: Vec<Cube> = Vec::new();
    for p in cand {
        let mut occ = [false; 8];
        let mut s = 0;
        for j in 0..8 {
            let cc = [p[0] + CUBE_CORNERS[j][0], p[1] + CUBE_CORNERS[j][1], p[2] + CUBE_CORNERS[j][2]];
            occ[j] = sdf_at(cc) < 0.0;
            s += occ[j] as usize;
        }
        if s == 0 || s == 8 {
            continue;
        }
        let case = (0..8).filter(|&j| occ[j]).map(|j| 1usize << j).sum();
        let w = wmap.get(&p);
        let beta = std::array::from_fn(|k| norm_beta(w.map_or(0.0, |w| w[k])));
        let alpha = std::array::from_fn(|k| norm_beta(w.map_or(0.0, |w| w[12 + k])));
        let gamma = norm_gamma(w.map_or(0.0, |w| w[20]));
        cubes.push(Cube { p, occ, case, beta, alpha, gamma });
    }
    cubes.sort_by(|a, b| a.p.cmp(&b.p));
    if cubes.is_empty() {
        return Ok(TexturedMesh { mesh: Mesh::default(), colors: Vec::new() });
    }

    // DMC ambiguity resolution: invert a case when it shares an ambiguous face with an ambiguous neighbour.
    let amb: HashMap<[i32; 3], [i64; 5]> = cubes
        .iter()
        .filter(|c| CHECK[c.case][0] == 1)
        .map(|c| (c.p, CHECK[c.case]))
        .collect();
    for c in cubes.iter_mut() {
        let pc = CHECK[c.case];
        if pc[0] != 1 {
            continue;
        }
        let adj = [c.p[0] + pc[1] as i32, c.p[1] + pc[2] as i32, c.p[2] + pc[3] as i32];
        if in_range(adj) && amb.get(&adj).map_or(false, |a| a[0] == 1) {
            c.case = pc[4] as usize;
        }
    }

    // dual vertices + per-cube-edge -> vd index map.
    let mut vd_pos: Vec<[f32; 3]> = Vec::new();
    let mut vd_col: Vec<[f32; 6]> = Vec::new();
    let mut vd_gamma: Vec<f32> = Vec::new();
    let mut vd_of_cube_edge: Vec<[i32; 12]> = vec![[-1; 12]; cubes.len()];

    let edge_vids = |p: [i32; 3]| -> [(u64, [i32; 3]); 8] {
        std::array::from_fn(|j| {
            let cc = [p[0] + CUBE_CORNERS[j][0], p[1] + CUBE_CORNERS[j][1], p[2] + CUBE_CORNERS[j][2]];
            (vid(cc, resv), cc)
        })
    };

    for (ci, c) in cubes.iter().enumerate() {
        let corner = edge_vids(c.p);
        for vdj in 0..NUM_VD[c.case] as usize {
            let group = &DMC[c.case][vdj];
            let (mut acc, mut acc_c, mut bsum) = ([0.0f32; 3], [0.0f32; 6], 0.0f32);
            let vd_index = vd_pos.len() as i32;
            for &e in group.iter() {
                if e < 0 {
                    continue;
                }
                let e = e as usize;
                let (ia, ib) = (CUBE_EDGES[2 * e], CUBE_EDGES[2 * e + 1]);
                let (ca, cb) = (corner[ia].1, corner[ib].1);
                let (va, vb) = (verts.get(&ca), verts.get(&cb));
                let (s0, s1) = (va.map_or(1.0, |v| v.sdf), vb.map_or(1.0, |v| v.sdf));
                let x0 = va.map_or_else(|| default_x(ca, res), |v| v.x);
                let x1 = vb.map_or_else(|| default_x(cb, res), |v| v.x);
                let col0 = va.map_or([0.5; 6], |v| v.color);
                let col1 = vb.map_or([0.5; 6], |v| v.color);
                let (a0, a1) = (c.alpha[ia], c.alpha[ib]);
                let beta = c.beta[e];
                let ue = interp3(x0, x1, s0 * a0, s1 * a1);
                let uc = interp6(&col0, &col1, s0 * a0, s1 * a1);
                for k in 0..3 {
                    acc[k] += ue[k] * beta;
                }
                for k in 0..6 {
                    acc_c[k] += uc[k] * beta;
                }
                bsum += beta;
                vd_of_cube_edge[ci][e] = vd_index;
            }
            vd_pos.push(std::array::from_fn(|k| acc[k] / bsum));
            vd_col.push(std::array::from_fn(|k| acc_c[k] / bsum));
            vd_gamma.push(c.gamma);
        }
    }

    // triangulate: group surf edges shared by exactly 4 cubes into quads, split by gamma.
    struct EdgeHit {
        vd: i32,
        s0: f32, // sdf of the edge's first endpoint (for winding)
    }
    let mut edge_groups: HashMap<(u64, u64), Vec<EdgeHit>> = HashMap::new();
    for (ci, c) in cubes.iter().enumerate() {
        let corner = edge_vids(c.p);
        for e in 0..12 {
            let vd = vd_of_cube_edge[ci][e];
            if vd < 0 {
                continue;
            }
            let (ia, ib) = (CUBE_EDGES[2 * e], CUBE_EDGES[2 * e + 1]);
            let (key_a, key_b) = (corner[ia].0, corner[ib].0);
            let s0 = verts.get(&corner[ia].1).map_or(1.0, |v| v.sdf);
            edge_groups.entry((key_a, key_b)).or_default().push(EdgeHit { vd, s0 });
        }
    }

    let mut faces: Vec<[u32; 3]> = Vec::new();
    for hits in edge_groups.values() {
        if hits.len() != 4 {
            continue;
        }
        // hits are in cube-sorted order (cubes iterated sorted). quad = the 4 dual verts.
        let q: [i32; 4] = std::array::from_fn(|k| hits[k].vd);
        let quad = if hits[0].s0 > 0.0 {
            [q[0], q[1], q[3], q[2]]
        } else {
            [q[2], q[3], q[1], q[0]]
        };
        let g: [f32; 4] = std::array::from_fn(|k| vd_gamma[quad[k] as usize]);
        let (g02, g13) = (g[0] * g[2], g[1] * g[3]);
        let tri = if g02 > g13 {
            [[quad[0], quad[1], quad[2]], [quad[0], quad[2], quad[3]]]
        } else {
            [[quad[0], quad[1], quad[3]], [quad[3], quad[1], quad[2]]]
        };
        for t in tri {
            faces.push([t[0] as u32, t[1] as u32, t[2] as u32]);
        }
    }

    let vertices: Vec<Vec3> = vd_pos.iter().map(|p| Vec3::new(p[0], p[1], p[2])).collect();
    let colors: Vec<[f32; 3]> = vd_col.iter().map(|c| [c[0], c[1], c[2]]).collect();
    Ok(TexturedMesh {
        mesh: Mesh::new(vertices, faces),
        colors,
    })
}

fn default_x(c: [i32; 3], res: i32) -> [f32; 3] {
    std::array::from_fn(|k| c[k] as f32 / res as f32 - 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hanzo_ml::Device;

    // TRELLIS_FIX=/oracle/fixtures cargo test -p hanzo-engine pixal3d::flexicubes -- --ignored --nocapture
    #[test]
    #[ignore = "needs TRELLIS_FIX (mesh_dec_io + mesh_out)"]
    fn flexicubes_vs_golden() {
        let dir = std::env::var("TRELLIS_FIX").expect("TRELLIS_FIX");
        let dev = Device::Cpu;
        let io = hanzo_ml::safetensors::load(format!("{dir}/mesh_dec_io.safetensors"), &dev).unwrap();
        let coords: Vec<[i32; 3]> = io["out_coords"]
            .to_vec2::<f32>()
            .unwrap()
            .iter()
            .map(|r| [r[1] as i32, r[2] as i32, r[3] as i32])
            .collect();
        let tm = extract(&coords, &io["out_feats"], 256).unwrap();

        let gold = hanzo_ml::safetensors::load(format!("{dir}/mesh_out.safetensors"), &dev).unwrap();
        let gv = gold["vertices"].dim(0).unwrap();
        let gf = gold["faces"].dim(0).unwrap();
        println!(
            "flexicubes: verts {} (golden {gv})  faces {} (golden {gf})",
            tm.mesh.vertices.len(),
            tm.mesh.faces.len()
        );
        // Chamfer: mean nearest-neighbour distance golden -> ours (geometry match, order-independent).
        let ours: Vec<[f32; 3]> = tm.mesh.vertices.iter().map(|v| [v.x, v.y, v.z]).collect();
        let gverts = gold["vertices"].to_vec2::<f32>().unwrap();
        let mut chamfer = 0.0f64;
        let step = (gverts.len() / 2000).max(1); // subsample for speed
        let mut cnt = 0;
        for (i, g) in gverts.iter().enumerate() {
            if i % step != 0 {
                continue;
            }
            let mut best = f32::INFINITY;
            for o in &ours {
                let d = (g[0] - o[0]).powi(2) + (g[1] - o[1]).powi(2) + (g[2] - o[2]).powi(2);
                best = best.min(d);
            }
            chamfer += best.sqrt() as f64;
            cnt += 1;
        }
        chamfer /= cnt as f64;
        println!("chamfer(golden->ours) mean = {chamfer:.6} (grid cell = {:.6})", 1.0 / 256.0);
        assert!(!tm.mesh.faces.is_empty());
        assert!(chamfer < 1.0 / 256.0, "chamfer {chamfer} exceeds one grid cell");
    }

    // Writes coarse (occupancy) + fine (FlexiCubes textured) GLBs for before/after comparison.
    // PIXAL3D_OUT=/dir TRELLIS_FIX=/fix cargo test -p hanzo-engine pixal3d::flexicubes::write_ -- --ignored --nocapture
    #[test]
    #[ignore = "needs TRELLIS_FIX; writes GLB artifacts to PIXAL3D_OUT"]
    fn write_textured_glb_artifact() {
        use super::super::{glb, mesh};
        let dir = std::env::var("TRELLIS_FIX").expect("TRELLIS_FIX");
        let out = std::env::var("PIXAL3D_OUT").unwrap_or_else(|_| "/tmp".into());
        let dev = Device::Cpu;
        let io = hanzo_ml::safetensors::load(format!("{dir}/mesh_dec_io.safetensors"), &dev).unwrap();

        // fine textured mesh from the exact decoder output.
        let coords: Vec<[i32; 3]> = io["out_coords"]
            .to_vec2::<f32>()
            .unwrap()
            .iter()
            .map(|r| [r[1] as i32, r[2] as i32, r[3] as i32])
            .collect();
        let tm = extract(&coords, &io["out_feats"], 256).unwrap();
        let fine = glb::mesh_to_glb_colored(&tm.mesh, Some(&tm.colors));
        std::fs::write(format!("{out}/pixal3d_fine_textured.glb"), &fine).unwrap();

        // coarse occupancy mesh from the same active voxels (res 64) for comparison.
        let vin: Vec<[i32; 3]> = io["coords"]
            .to_vec2::<f32>()
            .unwrap()
            .iter()
            .map(|r| [r[1] as i32, r[2] as i32, r[3] as i32])
            .collect();
        let r = 64usize;
        let mut occ = vec![false; r * r * r];
        for c in &vin {
            occ[(c[0] as usize * r + c[1] as usize) * r + c[2] as usize] = true;
        }
        let coarse_mesh = mesh::occupancy_to_mesh(&occ, r);
        let coarse = glb::mesh_to_glb(&coarse_mesh);
        std::fs::write(format!("{out}/pixal3d_coarse.glb"), &coarse).unwrap();

        println!(
            "coarse: {} verts / {} faces ({} bytes)  ->  fine: {} verts / {} faces / {} colors ({} bytes)",
            coarse_mesh.vertices.len(),
            coarse_mesh.faces.len(),
            coarse.len(),
            tm.mesh.vertices.len(),
            tm.mesh.faces.len(),
            tm.colors.len(),
            fine.len(),
        );
        assert_eq!(&fine[0..4], b"glTF");
    }
}
