use std::sync::Arc;
use std::thread;

use engram_embeddings::{Embedder, EmbeddingCache};

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn cache_is_send_sync() {
    assert_send_sync::<EmbeddingCache>();
}

#[test]
fn embedder_is_send_sync() {
    assert_send_sync::<Embedder>();
}

#[test]
fn concurrent_cache_reads() {
    let cache = Arc::new(EmbeddingCache::new());
    cache.insert("k1", Some("document"), vec![1.0, 2.0]);
    cache.insert("k2", Some("document"), vec![3.0, 4.0]);
    let handles: Vec<_> = (0..10)
        .map(|_| {
            let cache = Arc::clone(&cache);
            thread::spawn(move || {
                assert_eq!(cache.get("k1", Some("document")), Some(vec![1.0, 2.0]));
                assert_eq!(cache.get("k2", Some("document")), Some(vec![3.0, 4.0]));
            })
        })
        .collect();
    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn interior_mutation_via_shared_ref() {
    let cache = Arc::new(EmbeddingCache::new());
    let handles: Vec<_> = (0..10)
        .map(|i| {
            let cache = Arc::clone(&cache);
            thread::spawn(move || {
                cache.insert(&format!("k{i}"), Some("document"), vec![i as f32]);
            })
        })
        .collect();
    for handle in handles {
        handle.join().unwrap();
    }
    for i in 0..10 {
        assert_eq!(
            cache.get(&format!("k{i}"), Some("document")),
            Some(vec![i as f32])
        );
    }
}
