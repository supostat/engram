use engram_hnsw::Node;

#[test]
fn test_node_creation() {
    let vector = vec![1.0, 2.0, 3.0];
    let node = Node::new(42, vector.clone(), 3);

    assert_eq!(node.id, 42);
    assert_eq!(node.vector, vector);
    assert_eq!(node.level, 3);
}

#[test]
fn test_node_neighbors_initialized() {
    let node = Node::new(1, vec![0.5, 0.5], 4);

    assert_eq!(node.neighbors.len(), 5); // level + 1
    for neighbor_list in &node.neighbors {
        assert!(neighbor_list.is_empty());
    }
}

#[test]
fn test_node_level_zero() {
    let node = Node::new(0, vec![1.0], 0);

    assert_eq!(node.level, 0);
    assert_eq!(node.neighbors.len(), 1);
    assert!(node.neighbors[0].is_empty());
}
