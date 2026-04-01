use std::fs;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use engram_storage::Database;

use crate::error::CoreError;
use crate::indexes::IndexSet;

const INDEX_FILENAME: &str = "indexes.hnsw";

const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0100_0000_01b3;
const DETERMINISTIC_RNG_MULTIPLIER: u64 = 0x9e37_79b9_7f4a_7c15;

pub fn load_or_rebuild(
    index_directory: &str,
    database: &Database,
    build_params: impl Fn() -> Result<engram_hnsw::HnswParams, CoreError>,
) -> Result<IndexSet, CoreError> {
    let index_path = Path::new(index_directory).join(INDEX_FILENAME);
    if index_path.exists()
        && let Ok(indexes) = load_from_disk(&index_path) {
            return Ok(indexes);
        }
    rebuild_from_database(database, build_params)
}

fn load_from_disk(path: &Path) -> Result<IndexSet, CoreError> {
    let file = fs::File::open(path).map_err(|error| CoreError::IndexCorrupted(error.to_string()))?;
    let mut reader = BufReader::new(file);
    IndexSet::deserialize(&mut reader).map_err(CoreError::Hnsw)
}

fn rebuild_from_database(
    database: &Database,
    build_params: impl Fn() -> Result<engram_hnsw::HnswParams, CoreError>,
) -> Result<IndexSet, CoreError> {
    let mut indexes = IndexSet::new(build_params)?;
    let memories = database.get_unindexed_memories(usize::MAX)?;
    for memory in &memories {
        let id = parse_memory_id(&memory.id)?;
        let embedding = match extract_embedding(memory) {
            Some(emb) => emb,
            None => continue,
        };
        indexes.insert(id, &embedding, deterministic_rng(id))?;
    }
    Ok(indexes)
}

pub fn save_to_disk(index_directory: &str, indexes: &IndexSet) -> Result<(), CoreError> {
    let index_path = Path::new(index_directory).join(INDEX_FILENAME);
    fs::create_dir_all(index_directory)
        .map_err(|error| CoreError::RebuildFailed(error.to_string()))?;
    let file = fs::File::create(&index_path)
        .map_err(|error| CoreError::RebuildFailed(error.to_string()))?;
    let mut writer = BufWriter::new(file);
    indexes
        .serialize(&mut writer)
        .map_err(|error| CoreError::RebuildFailed(error.to_string()))?;
    Ok(())
}

fn parse_memory_id(id: &str) -> Result<u64, CoreError> {
    let hash = hash_string_to_u64(id);
    Ok(hash)
}

pub fn hash_string_to_u64(value: &str) -> u64 {
    let mut hash: u64 = FNV_OFFSET_BASIS;
    for byte in value.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

pub fn extract_embeddings_from_memory(memory: &engram_storage::Memory) -> Result<engram_embeddings::ThreeFieldEmbedding, CoreError> {
    extract_embedding(memory).ok_or_else(|| CoreError::IndexCorrupted("missing embeddings".into()))
}

fn extract_embedding(memory: &engram_storage::Memory) -> Option<engram_embeddings::ThreeFieldEmbedding> {
    let context = bytes_to_f32_vec(memory.embedding_context.as_deref()?)?;
    let action = bytes_to_f32_vec(memory.embedding_action.as_deref()?)?;
    let result = bytes_to_f32_vec(memory.embedding_result.as_deref()?)?;
    Some(engram_embeddings::ThreeFieldEmbedding {
        context,
        action,
        result,
    })
}

fn bytes_to_f32_vec(bytes: &[u8]) -> Option<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) {
        return None;
    }
    let floats = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    Some(floats)
}

pub fn deterministic_rng(id: u64) -> f64 {
    let mixed = id.wrapping_mul(DETERMINISTIC_RNG_MULTIPLIER);
    (mixed >> 11) as f64 / (1u64 << 53) as f64
}
