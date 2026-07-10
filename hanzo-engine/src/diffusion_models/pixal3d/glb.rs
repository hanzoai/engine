//! Binary glTF (`.glb`) writer for a decoded [`hanzo_3d::Mesh`].
//!
//! GLB is the product export the `/v1/3d` endpoint serves. hanzo-3d ships OBJ/PLY (generic mesh
//! interchange); GLB carries the primitive + PBR material a viewer renders directly, so it lives with
//! the pixal3d product path. Zero deps beyond serde_json: GLB is a 12-byte header + a JSON chunk + a
//! BIN chunk, both padded to 4 bytes.

use hanzo_3d::Mesh;
use serde_json::{json, Value};

const MAGIC: u32 = 0x4654_6C67; // "glTF"
const VERSION: u32 = 2;
const CHUNK_JSON: u32 = 0x4E4F_534A; // "JSON"
const CHUNK_BIN: u32 = 0x004E_4942; // "BIN\0"

const CT_FLOAT: u32 = 5126;
const CT_UINT: u32 = 5125;
const TARGET_ARRAY: u32 = 34962; // ARRAY_BUFFER
const TARGET_ELEMENT: u32 = 34963; // ELEMENT_ARRAY_BUFFER
const MODE_TRIANGLES: u32 = 4;

const JSON_PAD: u8 = 0x20; // space
const BIN_PAD: u8 = 0x00;

fn pad4(buf: &mut Vec<u8>, fill: u8) {
    while buf.len() % 4 != 0 {
        buf.push(fill);
    }
}

/// A staged section of the BIN buffer plus the bufferView it produced.
struct Bin {
    data: Vec<u8>,
    views: Vec<Value>,
}

impl Bin {
    fn new() -> Self {
        Self {
            data: Vec::new(),
            views: Vec::new(),
        }
    }

    /// Append a 4-byte-aligned section, register its bufferView, return the view index.
    fn push(&mut self, bytes: &[u8], target: u32) -> usize {
        pad4(&mut self.data, BIN_PAD);
        let offset = self.data.len();
        self.data.extend_from_slice(bytes);
        let idx = self.views.len();
        self.views.push(json!({
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": bytes.len(),
            "target": target,
        }));
        idx
    }
}

fn floats_le(vals: impl Iterator<Item = f32>) -> Vec<u8> {
    let mut b = Vec::new();
    for v in vals {
        b.extend_from_slice(&v.to_le_bytes());
    }
    b
}

/// Serialize `mesh` to a binary glTF 2.0 container.
pub fn mesh_to_glb(mesh: &Mesh) -> Vec<u8> {
    let mut bin = Bin::new();
    let mut accessors: Vec<Value> = Vec::new();
    let mut attributes = serde_json::Map::new();

    let n_idx = mesh.faces.len() * 3;
    let idx_bytes = {
        let mut b = Vec::with_capacity(n_idx * 4);
        for f in &mesh.faces {
            for &i in f {
                b.extend_from_slice(&i.to_le_bytes());
            }
        }
        b
    };
    let idx_view = bin.push(&idx_bytes, TARGET_ELEMENT);
    let idx_accessor = accessors.len();
    accessors.push(json!({
        "bufferView": idx_view,
        "componentType": CT_UINT,
        "count": n_idx,
        "type": "SCALAR",
    }));

    let n_vert = mesh.vertices.len();
    let pos_bytes = floats_le(
        mesh.vertices
            .iter()
            .flat_map(|v| [v.x, v.y, v.z].into_iter()),
    );
    let (min, max) = bounds(mesh);
    let pos_view = bin.push(&pos_bytes, TARGET_ARRAY);
    attributes.insert("POSITION".into(), json!(accessors.len()));
    accessors.push(json!({
        "bufferView": pos_view,
        "componentType": CT_FLOAT,
        "count": n_vert,
        "type": "VEC3",
        "min": min,
        "max": max,
    }));

    if !mesh.normals.is_empty() {
        let nrm_bytes = floats_le(
            mesh.normals
                .iter()
                .flat_map(|v| [v.x, v.y, v.z].into_iter()),
        );
        let view = bin.push(&nrm_bytes, TARGET_ARRAY);
        attributes.insert("NORMAL".into(), json!(accessors.len()));
        accessors.push(json!({
            "bufferView": view,
            "componentType": CT_FLOAT,
            "count": mesh.normals.len(),
            "type": "VEC3",
        }));
    }

    if !mesh.uvs.is_empty() {
        let uv_bytes = floats_le(mesh.uvs.iter().flat_map(|uv| uv.iter().copied()));
        let view = bin.push(&uv_bytes, TARGET_ARRAY);
        attributes.insert("TEXCOORD_0".into(), json!(accessors.len()));
        accessors.push(json!({
            "bufferView": view,
            "componentType": CT_FLOAT,
            "count": mesh.uvs.len(),
            "type": "VEC2",
        }));
    }

    let m = &mesh.material;
    let doc = json!({
        "asset": {"version": "2.0", "generator": "hanzo-engine pixal3d"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{
            "attributes": attributes,
            "indices": idx_accessor,
            "material": 0,
            "mode": MODE_TRIANGLES,
        }]}],
        "materials": [{"pbrMetallicRoughness": {
            "baseColorFactor": m.base_color,
            "metallicFactor": m.metallic,
            "roughnessFactor": m.roughness,
        }}],
        "accessors": accessors,
        "bufferViews": bin.views,
        "buffers": [{"byteLength": bin.data.len()}],
    });

    assemble(&serde_json::to_vec(&doc).expect("glb json"), bin.data)
}

fn bounds(mesh: &Mesh) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for v in &mesh.vertices {
        for (k, c) in [v.x, v.y, v.z].into_iter().enumerate() {
            min[k] = min[k].min(c);
            max[k] = max[k].max(c);
        }
    }
    if mesh.vertices.is_empty() {
        (([0.0; 3]), ([0.0; 3]))
    } else {
        (min, max)
    }
}

fn assemble(json_bytes: &[u8], mut bin: Vec<u8>) -> Vec<u8> {
    let mut json = json_bytes.to_vec();
    pad4(&mut json, JSON_PAD);
    pad4(&mut bin, BIN_PAD);
    let total = 12 + 8 + json.len() + 8 + bin.len();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&MAGIC.to_le_bytes());
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(json.len() as u32).to_le_bytes());
    out.extend_from_slice(&CHUNK_JSON.to_le_bytes());
    out.extend_from_slice(&json);
    out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    out.extend_from_slice(&CHUNK_BIN.to_le_bytes());
    out.extend_from_slice(&bin);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u32_le(b: &[u8], off: usize) -> u32 {
        u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
    }

    #[test]
    fn unit_cube_is_valid_glb() {
        let glb = mesh_to_glb(&hanzo_3d::unit_cube());

        assert_eq!(&glb[0..4], b"glTF");
        assert_eq!(u32_le(&glb, 4), 2);
        assert_eq!(u32_le(&glb, 8) as usize, glb.len());
        assert_eq!(glb.len() % 4, 0);

        let json_len = u32_le(&glb, 12) as usize;
        assert_eq!(u32_le(&glb, 16), CHUNK_JSON);
        let json_start = 20;
        let json: Value =
            serde_json::from_slice(&glb[json_start..json_start + json_len]).expect("json chunk");

        let bin_off = json_start + json_len; // BIN chunk header: [len][type][data]
        let bin_chunk_len = u32_le(&glb, bin_off) as usize;
        assert_eq!(u32_le(&glb, bin_off + 4), CHUNK_BIN);
        assert_eq!(bin_off + 8 + bin_chunk_len, glb.len());
        // buffers[0].byteLength is the unpadded payload; the chunk may carry <4 bytes of trailing pad.
        let declared = json["buffers"][0]["byteLength"].as_u64().unwrap() as usize;
        assert!(bin_chunk_len >= declared && bin_chunk_len - declared < 4);

        // Cube: 8 verts, 12 tris -> 36 indices; POSITION + indices accessors, no normals/uvs.
        assert_eq!(json["accessors"][0]["count"].as_u64().unwrap(), 36);
        assert_eq!(json["accessors"][0]["componentType"].as_u64().unwrap(), CT_UINT as u64);
        let pos = &json["accessors"][1];
        assert_eq!(pos["count"].as_u64().unwrap(), 8);
        assert_eq!(pos["type"].as_str().unwrap(), "VEC3");
        assert_eq!(pos["min"], json!([0.0, 0.0, 0.0]));
        assert_eq!(pos["max"], json!([1.0, 1.0, 1.0]));
        assert_eq!(json["meshes"][0]["primitives"][0]["mode"].as_u64().unwrap(), 4);
    }

    #[test]
    fn normals_and_uvs_add_accessors() {
        use hanzo_3d::Vec3;
        let mut mesh = hanzo_3d::unit_cube();
        mesh.normals = mesh
            .vertices
            .iter()
            .map(|_| Vec3::new(0.0, 0.0, 1.0))
            .collect();
        mesh.uvs = mesh.vertices.iter().map(|_| [0.5, 0.5]).collect();
        let glb = mesh_to_glb(&mesh);
        let json_len = u32_le(&glb, 12) as usize;
        let json: Value = serde_json::from_slice(&glb[20..20 + json_len]).unwrap();
        let attrs = &json["meshes"][0]["primitives"][0]["attributes"];
        assert!(attrs["POSITION"].is_number());
        assert!(attrs["NORMAL"].is_number());
        assert!(attrs["TEXCOORD_0"].is_number());
        assert_eq!(json["accessors"].as_array().unwrap().len(), 4);
    }
}
