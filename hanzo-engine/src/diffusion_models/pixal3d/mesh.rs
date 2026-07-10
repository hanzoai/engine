//! Surface-mesh extraction from an occupancy grid.
//!
//! The sparse-structure decoder yields a res^3 occupancy field; thresholding it gives the coarse
//! shape. This emits the boundary surface (a quad for every occupied->empty voxel face) as a
//! watertight [`Mesh`] in the [-0.5, 0.5]^3 cube TRELLIS uses. It is the coarse-geometry output; the
//! SLAT stage refines it to a detailed FlexiCubes mesh.

use std::collections::HashMap;

use hanzo_3d::{Mesh, Vec3};

/// (neighbor offset, 4 CCW corner offsets giving an outward-facing quad).
const FACES: [([i32; 3], [[i32; 3]; 4]); 6] = [
    ([1, 0, 0], [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]]),
    ([-1, 0, 0], [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]]),
    ([0, 1, 0], [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]]),
    ([0, -1, 0], [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]]),
    ([0, 0, 1], [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]),
    ([0, 0, -1], [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]]),
];

struct Dedup {
    verts: Vec<Vec3>,
    map: HashMap<[i32; 3], u32>,
    res: f32,
}

impl Dedup {
    fn get(&mut self, corner: [i32; 3]) -> u32 {
        if let Some(&i) = self.map.get(&corner) {
            return i;
        }
        let i = self.verts.len() as u32;
        // lattice point in 0..=res -> world in [-0.5, 0.5]
        let w = |c: i32| c as f32 / self.res - 0.5;
        self.verts.push(Vec3::new(w(corner[0]), w(corner[1]), w(corner[2])));
        self.map.insert(corner, i);
        i
    }
}

/// Build the boundary surface mesh from a row-major `res^3` occupancy grid
/// (index `(x*res + y)*res + z`).
pub fn occupancy_to_mesh(occ: &[bool], res: usize) -> Mesh {
    let at = |x: i32, y: i32, z: i32| -> bool {
        x >= 0
            && y >= 0
            && z >= 0
            && (x as usize) < res
            && (y as usize) < res
            && (z as usize) < res
            && occ[(x as usize * res + y as usize) * res + z as usize]
    };
    let mut d = Dedup {
        verts: Vec::new(),
        map: HashMap::new(),
        res: res as f32,
    };
    let mut faces: Vec<[u32; 3]> = Vec::new();
    for x in 0..res as i32 {
        for y in 0..res as i32 {
            for z in 0..res as i32 {
                if !at(x, y, z) {
                    continue;
                }
                for (n, corners) in &FACES {
                    if at(x + n[0], y + n[1], z + n[2]) {
                        continue; // interior face
                    }
                    let v: [u32; 4] = std::array::from_fn(|k| {
                        d.get([x + corners[k][0], y + corners[k][1], z + corners[k][2]])
                    });
                    faces.push([v[0], v[1], v[2]]);
                    faces.push([v[0], v[2], v[3]]);
                }
            }
        }
    }
    Mesh::new(d.verts, faces)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_voxel_is_a_cube() {
        // A 2^3 grid with one occupied cell -> a closed cube: 6 faces * 2 tris, 8 unique verts.
        let mut occ = vec![false; 8];
        occ[(1 * 2 + 1) * 2 + 1] = true; // voxel (1,1,1)
        let mesh = occupancy_to_mesh(&occ, 2);
        assert_eq!(mesh.faces.len(), 12);
        assert_eq!(mesh.vertices.len(), 8);
    }

    #[test]
    fn shared_face_is_culled() {
        // Two adjacent voxels: the shared face is dropped -> 10 exposed faces * 2 tris.
        let mut occ = vec![false; 27];
        let idx = |x: usize, y: usize, z: usize| (x * 3 + y) * 3 + z;
        occ[idx(1, 1, 1)] = true;
        occ[idx(2, 1, 1)] = true;
        let mesh = occupancy_to_mesh(&occ, 3);
        assert_eq!(mesh.faces.len(), 20);
    }
}
