use engram_hnsw::{HnswError, HnswGraph, HnswParams};

#[test]
fn test_graph_new() {
    let params = HnswParams::new(128).unwrap();
    let graph = HnswGraph::new(params);

    assert_eq!(graph.dimension(), 128);
    assert!(graph.is_empty());
}

#[test]
fn test_graph_is_empty() {
    let graph = HnswGraph::new(HnswParams::new(64).unwrap());

    assert!(graph.is_empty());
    assert_eq!(graph.len(), 0);
    assert!(!graph.contains(0));
}

#[test]
fn test_random_level_distribution() {
    let graph = HnswGraph::new(HnswParams::new(3).unwrap());

    let sample_count = 10_000;
    let mut level_counts = [0usize; 10];

    for i in 0..sample_count {
        let uniform_value = ((i as f64 + 1.0) / (sample_count as f64 + 1.0))
            .clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
        let level = graph.random_level(uniform_value);
        if level < level_counts.len() {
            level_counts[level] += 1;
        }
    }

    assert!(
        level_counts[0] > sample_count / 2,
        "level 0 should have majority of nodes, got {}",
        level_counts[0]
    );

    for window in level_counts.windows(2) {
        if window[0] == 0 {
            break;
        }
        assert!(
            window[0] >= window[1],
            "level distribution should decay: {} < {}",
            window[0],
            window[1]
        );
    }
}

#[test]
fn test_params_defaults() {
    let params = HnswParams::new(256).unwrap();

    assert_eq!(params.max_connections, 16);
    assert_eq!(params.max_connections_layer0, 32);
    assert_eq!(params.ef_construction, 200);
    assert_eq!(params.ef_search, 50);
    assert_eq!(params.dimension, 256);
}

// --- Parameter validation tests ---

#[test]
fn test_params_dimension_zero_rejected() {
    let result = HnswParams::new(0);
    assert!(matches!(result, Err(HnswError::InvalidParameter(_))));
}

#[test]
fn test_params_dimension_too_large_rejected() {
    let result = HnswParams::new(65537);
    assert!(matches!(result, Err(HnswError::InvalidParameter(_))));
}

#[test]
fn test_params_dimension_boundary_accepted() {
    assert!(HnswParams::new(1).is_ok());
    assert!(HnswParams::new(65536).is_ok());
}

#[test]
fn test_params_max_connections_one_rejected() {
    let result = HnswParams::new(128).unwrap().with_max_connections(1);
    assert!(matches!(result, Err(HnswError::InvalidParameter(_))));
}

#[test]
fn test_params_max_connections_too_large_rejected() {
    let result = HnswParams::new(128).unwrap().with_max_connections(257);
    assert!(matches!(result, Err(HnswError::InvalidParameter(_))));
}

#[test]
fn test_params_max_connections_boundary_accepted() {
    assert!(
        HnswParams::new(128)
            .unwrap()
            .with_max_connections(2)
            .is_ok()
    );
    assert!(
        HnswParams::new(128)
            .unwrap()
            .with_max_connections(256)
            .is_ok()
    );
}

#[test]
fn test_params_ef_construction_zero_rejected() {
    let result = HnswParams::new(128).unwrap().with_ef_construction(0);
    assert!(matches!(result, Err(HnswError::InvalidParameter(_))));
}

#[test]
fn test_params_ef_search_zero_rejected() {
    let result = HnswParams::new(128).unwrap().with_ef_search(0);
    assert!(matches!(result, Err(HnswError::InvalidParameter(_))));
}

// --- random_level boundary tests ---

#[test]
fn test_random_level_near_zero_capped_at_32() {
    let graph = HnswGraph::new(HnswParams::new(3).unwrap());
    let level = graph.random_level(0.0);
    assert!(level <= 32, "level should be capped at 32, got {level}");
}

#[test]
fn test_random_level_negative_clamped() {
    let graph = HnswGraph::new(HnswParams::new(3).unwrap());
    let level = graph.random_level(-1.0);
    assert!(
        level <= 32,
        "negative input should be clamped, got level {level}"
    );
}

#[test]
fn test_random_level_near_one_gives_zero() {
    let graph = HnswGraph::new(HnswParams::new(3).unwrap());
    let level = graph.random_level(1.0);
    assert_eq!(level, 0, "uniform_value near 1.0 should give level 0");
}

#[test]
fn test_random_level_exactly_one_gives_zero() {
    let graph = HnswGraph::new(HnswParams::new(3).unwrap());
    let level = graph.random_level(1.0);
    assert_eq!(level, 0);
}
