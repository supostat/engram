use crate::error::HnswError;

/// Compute cosine similarity between two vectors.
/// Returns value in [-1.0, 1.0] where 1.0 = identical direction.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32, HnswError> {
    if a.len() != b.len() {
        return Err(HnswError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    if a.is_empty() {
        return Err(HnswError::EmptyVector);
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let magnitude_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let magnitude_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if magnitude_a == 0.0 || magnitude_b == 0.0 {
        return Ok(0.0);
    }

    Ok(dot / (magnitude_a * magnitude_b))
}
