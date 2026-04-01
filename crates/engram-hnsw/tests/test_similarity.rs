use engram_hnsw::{HnswError, cosine_similarity};

#[test]
fn test_identical_vectors() {
    let vector = vec![1.0, 2.0, 3.0];
    let result = cosine_similarity(&vector, &vector).unwrap();
    assert!((result - 1.0).abs() < 1e-6, "expected 1.0, got {result}");
}

#[test]
fn test_orthogonal_vectors() {
    let a = vec![1.0, 0.0];
    let b = vec![0.0, 1.0];
    let result = cosine_similarity(&a, &b).unwrap();
    assert!(result.abs() < 1e-6, "expected 0.0, got {result}");
}

#[test]
fn test_opposite_vectors() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![-1.0, -2.0, -3.0];
    let result = cosine_similarity(&a, &b).unwrap();
    assert!((result + 1.0).abs() < 1e-6, "expected -1.0, got {result}");
}

#[test]
fn test_dimension_mismatch() {
    let a = vec![1.0, 2.0];
    let b = vec![1.0, 2.0, 3.0];
    let result = cosine_similarity(&a, &b);
    assert_eq!(
        result,
        Err(HnswError::DimensionMismatch {
            expected: 2,
            got: 3
        })
    );
}

#[test]
fn test_empty_vectors() {
    let a: Vec<f32> = vec![];
    let b: Vec<f32> = vec![];
    let result = cosine_similarity(&a, &b);
    assert_eq!(result, Err(HnswError::EmptyVector));
}

#[test]
fn test_zero_vector() {
    let zero = vec![0.0, 0.0, 0.0];
    let other = vec![1.0, 2.0, 3.0];
    let result = cosine_similarity(&zero, &other).unwrap();
    assert!(result.abs() < 1e-6, "expected 0.0, got {result}");
}

#[test]
fn test_known_values() {
    // cos([1,2,3], [4,5,6]) = (4+10+18) / (sqrt(14) * sqrt(77))
    // = 32 / sqrt(1078) ≈ 0.974631846
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let result = cosine_similarity(&a, &b).unwrap();
    let expected = 32.0 / (14.0_f32.sqrt() * 77.0_f32.sqrt());
    assert!(
        (result - expected).abs() < 1e-6,
        "expected {expected}, got {result}"
    );
}

#[test]
fn test_nan_input() {
    let a = vec![f32::NAN, 1.0];
    let b = vec![1.0, 1.0];
    let result = cosine_similarity(&a, &b).unwrap();
    assert!(result.is_nan(), "NaN input should produce NaN result");
}

#[test]
fn test_infinity_input() {
    let a = vec![f32::INFINITY, 1.0];
    let b = vec![1.0, 1.0];
    let result = cosine_similarity(&a, &b).unwrap();
    assert!(
        result.is_nan() || result.is_finite(),
        "INFINITY input should not panic, got {result}"
    );
}

#[test]
fn test_single_dimension_vectors() {
    let a = vec![3.0];
    let b = vec![5.0];
    let result = cosine_similarity(&a, &b).unwrap();
    assert!(
        (result - 1.0).abs() < 1e-6,
        "same-sign 1-dim vectors should have similarity 1.0, got {result}"
    );

    let c = vec![-2.0];
    let result_opposite = cosine_similarity(&a, &c).unwrap();
    assert!(
        (result_opposite + 1.0).abs() < 1e-6,
        "opposite-sign 1-dim vectors should have similarity -1.0, got {result_opposite}"
    );
}
