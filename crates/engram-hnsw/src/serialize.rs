use std::io::{self, Read, Write};

use crate::error::HnswError;
use crate::graph::{HnswGraph, HnswParams};
use crate::node::Node;

const MAGIC: u32 = 0x484E_5357; // "HNSW" in ASCII hex
const VERSION: u32 = 1;
const MAX_NODES: usize = 10_000_000;
const MAX_LAYERS: usize = 64;
const MAX_NEIGHBORS_PER_LAYER: usize = 10_000;

impl HnswGraph {
    /// Serialize graph to a custom binary format.
    pub fn serialize<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        write_u32(writer, MAGIC)?;
        write_u32(writer, VERSION)?;
        self.write_params(writer)?;
        self.write_nodes(writer)?;
        self.write_footer(writer)
    }

    fn write_params<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        let params = self.params();
        write_u32(writer, params.dimension as u32)?;
        write_u32(writer, params.max_connections as u32)?;
        write_u32(writer, params.max_connections_layer0 as u32)?;
        write_u32(writer, params.ef_construction as u32)?;
        write_u32(writer, params.ef_search as u32)
    }

    fn write_nodes<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        let nodes = self.nodes();
        if nodes.len() > u32::MAX as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("node count {} exceeds u32::MAX", nodes.len()),
            ));
        }
        write_u32(writer, nodes.len() as u32)?;
        for node in nodes.values() {
            write_node(writer, node)?;
        }
        Ok(())
    }

    fn write_footer<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        let entry_point = self.entry_point().unwrap_or(u64::MAX);
        write_u64(writer, entry_point)?;
        write_u32(writer, self.max_level() as u32)
    }

    /// Deserialize graph from custom binary format.
    pub fn deserialize<R: Read>(reader: &mut R) -> Result<Self, HnswError> {
        validate_header(reader)?;
        let params = read_params(reader)?;
        let node_count = read_io_u32(reader).map_err(corrupted)? as usize;
        if node_count > MAX_NODES {
            return Err(HnswError::IndexCorrupted(format!(
                "node count {node_count} exceeds maximum {MAX_NODES}"
            )));
        }
        let mut graph = HnswGraph::new(params);

        for _ in 0..node_count {
            let node = read_node(reader, graph.dimension())?;
            graph.nodes_mut().insert(node.id, node);
        }

        let (entry_point, max_level) = read_footer(reader)?;
        graph.set_entry_point(entry_point);
        graph.set_max_level(max_level);
        Ok(graph)
    }
}

fn validate_header<R: Read>(reader: &mut R) -> Result<(), HnswError> {
    let magic = read_io_u32(reader).map_err(|e| HnswError::IndexCorrupted(e.to_string()))?;
    if magic != MAGIC {
        return Err(HnswError::IndexCorrupted(format!(
            "invalid magic: expected {MAGIC:#X}, got {magic:#X}"
        )));
    }
    let version = read_io_u32(reader).map_err(|e| HnswError::IndexCorrupted(e.to_string()))?;
    if version != VERSION {
        return Err(HnswError::RebuildRequired);
    }
    Ok(())
}

fn read_params<R: Read>(reader: &mut R) -> Result<HnswParams, HnswError> {
    let dimension = read_io_u32(reader).map_err(corrupted)? as usize;
    let max_connections = read_io_u32(reader).map_err(corrupted)? as usize;
    let max_connections_layer0 = read_io_u32(reader).map_err(corrupted)? as usize;
    let ef_construction = read_io_u32(reader).map_err(corrupted)? as usize;
    let ef_search = read_io_u32(reader).map_err(corrupted)? as usize;

    let mut params = HnswParams::new(dimension)?;
    params = params.with_max_connections(max_connections)?;
    params.max_connections_layer0 = max_connections_layer0;
    params = params.with_ef_construction(ef_construction)?;
    params = params.with_ef_search(ef_search)?;
    Ok(params)
}

fn read_footer<R: Read>(reader: &mut R) -> Result<(Option<u64>, usize), HnswError> {
    let entry_raw = read_io_u64(reader).map_err(corrupted)?;
    let entry_point = if entry_raw == u64::MAX {
        None
    } else {
        Some(entry_raw)
    };
    let max_level = read_io_u32(reader).map_err(corrupted)? as usize;
    Ok((entry_point, max_level))
}

fn write_node<W: Write>(writer: &mut W, node: &Node) -> io::Result<()> {
    write_u64(writer, node.id)?;
    write_u32(writer, node.level as u32)?;
    write_u32(writer, node.vector.len() as u32)?;
    for &value in &node.vector {
        write_f32(writer, value)?;
    }
    write_u32(writer, node.neighbors.len() as u32)?;
    for layer_neighbors in &node.neighbors {
        write_u32(writer, layer_neighbors.len() as u32)?;
        for &neighbor_id in layer_neighbors {
            write_u64(writer, neighbor_id)?;
        }
    }
    Ok(())
}

fn read_node<R: Read>(reader: &mut R, expected_dim: usize) -> Result<Node, HnswError> {
    let id = read_io_u64(reader).map_err(corrupted)?;
    let level = read_io_u32(reader).map_err(corrupted)? as usize;
    let vector = read_vector(reader, expected_dim)?;
    let neighbors = read_neighbor_lists(reader)?;
    Ok(Node {
        id,
        vector,
        level,
        neighbors,
    })
}

fn read_vector<R: Read>(reader: &mut R, expected_dim: usize) -> Result<Vec<f32>, HnswError> {
    let dim = read_io_u32(reader).map_err(corrupted)? as usize;
    if dim != expected_dim {
        return Err(HnswError::IndexCorrupted(format!(
            "vector dim mismatch: expected {expected_dim}, got {dim}"
        )));
    }
    let mut vector = Vec::with_capacity(dim);
    for _ in 0..dim {
        vector.push(read_io_f32(reader).map_err(corrupted)?);
    }
    Ok(vector)
}

fn read_neighbor_lists<R: Read>(reader: &mut R) -> Result<Vec<Vec<u64>>, HnswError> {
    let layer_count = read_io_u32(reader).map_err(corrupted)? as usize;
    if layer_count > MAX_LAYERS {
        return Err(HnswError::IndexCorrupted(format!(
            "layer count {layer_count} exceeds maximum {MAX_LAYERS}"
        )));
    }
    let mut neighbors = Vec::with_capacity(layer_count);
    for _ in 0..layer_count {
        let neighbor_count = read_io_u32(reader).map_err(corrupted)? as usize;
        if neighbor_count > MAX_NEIGHBORS_PER_LAYER {
            return Err(HnswError::IndexCorrupted(format!(
                "neighbor count {neighbor_count} exceeds maximum {MAX_NEIGHBORS_PER_LAYER}"
            )));
        }
        let mut layer_neighbors = Vec::with_capacity(neighbor_count);
        for _ in 0..neighbor_count {
            layer_neighbors.push(read_io_u64(reader).map_err(corrupted)?);
        }
        neighbors.push(layer_neighbors);
    }
    Ok(neighbors)
}

fn corrupted(error: io::Error) -> HnswError {
    HnswError::IndexCorrupted(error.to_string())
}

fn write_u32<W: Write>(writer: &mut W, value: u32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn write_u64<W: Write>(writer: &mut W, value: u64) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn write_f32<W: Write>(writer: &mut W, value: f32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn read_io_u32<R: Read>(reader: &mut R) -> io::Result<u32> {
    let mut buffer = [0u8; 4];
    reader.read_exact(&mut buffer)?;
    Ok(u32::from_le_bytes(buffer))
}

fn read_io_u64<R: Read>(reader: &mut R) -> io::Result<u64> {
    let mut buffer = [0u8; 8];
    reader.read_exact(&mut buffer)?;
    Ok(u64::from_le_bytes(buffer))
}

fn read_io_f32<R: Read>(reader: &mut R) -> io::Result<f32> {
    let mut buffer = [0u8; 4];
    reader.read_exact(&mut buffer)?;
    Ok(f32::from_le_bytes(buffer))
}
