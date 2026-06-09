//! Weighted Reciprocal Rank Fusion for hybrid search.
//!
//! Fuses dense-vector and full-text hits via weighted Reciprocal Rank Fusion.
//! Both input slices arrive sorted best-first (HNSW by descending similarity,
//! FTS5 by ascending rank), so only each hit's rank position contributes —
//! the raw similarity/rank scores are intentionally ignored. A memory present
//! in both lists accumulates both weighted reciprocal-rank terms.

use std::collections::HashMap;

use crate::config::SearchConfig;

pub(crate) fn merge_results(
    vector_results: &[(String, f32)],
    sparse_results: &[(String, f64)],
    search: &SearchConfig,
) -> Vec<(String, f64)> {
    let k = search.rrf_k as f64;
    let mut combined: HashMap<String, f64> = HashMap::new();
    for (rank, (memory_id, _)) in vector_results.iter().enumerate() {
        *combined.entry(memory_id.clone()).or_insert(0.0) +=
            search.vector_weight / (k + (rank + 1) as f64);
    }
    for (rank, (memory_id, _)) in sparse_results.iter().enumerate() {
        *combined.entry(memory_id.clone()).or_insert(0.0) +=
            search.sparse_weight / (k + (rank + 1) as f64);
    }
    let mut results: Vec<(String, f64)> = combined.into_iter().collect();
    results.sort_by(|a, b| b.1.total_cmp(&a.1));
    results
}

pub(crate) fn limit_results(results: Vec<(String, f64)>, top_k: usize) -> Vec<(String, f64)> {
    results.into_iter().take(top_k).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vector_hit(id: &str) -> (String, f32) {
        (id.to_string(), 0.0)
    }

    fn sparse_hit(id: &str) -> (String, f64) {
        (id.to_string(), 0.0)
    }

    #[test]
    fn merge_ranks_by_position_not_score() {
        let vector = vec![vector_hit("a"), vector_hit("b")];
        let merged = merge_results(&vector, &[], &SearchConfig::default());
        assert_eq!(merged[0].0, "a");
        assert_eq!(merged[1].0, "b");
        assert!(merged[0].1 > merged[1].1);
    }

    #[test]
    fn merge_sums_contributions_for_shared_id() {
        let search = SearchConfig::default();
        let vector = vec![vector_hit("a"), vector_hit("b")];
        let sparse = vec![sparse_hit("a"), sparse_hit("c")];
        let merged = merge_results(&vector, &sparse, &search);

        let k = search.rrf_k as f64;
        let score_a = merged.iter().find(|(id, _)| id == "a").unwrap().1;
        let expected_a = search.vector_weight / (k + 1.0) + search.sparse_weight / (k + 1.0);
        assert!((score_a - expected_a).abs() < 1e-12);

        let score_b = merged.iter().find(|(id, _)| id == "b").unwrap().1;
        assert!(score_a > score_b);
    }

    // Asymmetric weights must drive the final ranking, not just rank position.
    // Construction: X is rank-1 in the dense-vector list and absent from sparse;
    // Y is rank-1 in sparse and far down the vector list (rank-9). With heavy
    // vector weighting (0.9 / 0.1) the dense source dominates and X leads; the
    // symmetric default (0.7 / 0.3) lets Y's strong sparse hit win instead, so
    // the two configs produce opposite orderings.
    //
    // Exact RRF (k = 60), heavy-vector 0.9 / 0.1:
    //   X = 0.9 / (60 + 1)              = 0.014754098360655738
    //   Y = 0.9 / (60 + 9) + 0.1/(60+1) = 0.014682822523164649  → X > Y
    // Symmetric default 0.7 / 0.3:
    //   X = 0.7 / (60 + 1)              = 0.011475409836065573
    //   Y = 0.7 / (60 + 9) + 0.3/(60+1) = 0.015062960323117129  → Y > X
    //
    // Y sits at vector rank-9 because, sharing both lists, it accrues the heavy
    // vector term too; only past rank-8 does X's single rank-1 vector hit
    // overtake Y's rank-9 vector hit plus its rank-1 sparse hit at 0.9 / 0.1.
    #[test]
    fn asymmetric_weights_drive_final_ranking() {
        let vector = vec![
            vector_hit("x"),
            vector_hit("v2"),
            vector_hit("v3"),
            vector_hit("v4"),
            vector_hit("v5"),
            vector_hit("v6"),
            vector_hit("v7"),
            vector_hit("v8"),
            vector_hit("y"),
        ];
        let sparse = vec![sparse_hit("y")];

        let heavy_vector = SearchConfig {
            rrf_k: 60,
            vector_weight: 0.9,
            sparse_weight: 0.1,
        };
        let merged = merge_results(&vector, &sparse, &heavy_vector);

        let score_x = merged.iter().find(|(id, _)| id == "x").unwrap().1;
        let score_y = merged.iter().find(|(id, _)| id == "y").unwrap().1;
        let expected_x = 0.9 / 61.0;
        let expected_y = 0.9 / 69.0 + 0.1 / 61.0;
        assert!((score_x - expected_x).abs() < 1e-12);
        assert!((score_y - expected_y).abs() < 1e-12);
        assert!(
            score_x > score_y,
            "heavy vector weighting must let the vector rank-1 id win: x={score_x}, y={score_y}"
        );
        let x_position = merged.iter().position(|(id, _)| id == "x").unwrap();
        let y_position = merged.iter().position(|(id, _)| id == "y").unwrap();
        assert!(x_position < y_position, "x must outrank y under 0.9 / 0.1");

        let symmetric = SearchConfig::default();
        let merged_symmetric = merge_results(&vector, &sparse, &symmetric);
        let symmetric_x = merged_symmetric.iter().find(|(id, _)| id == "x").unwrap().1;
        let symmetric_y = merged_symmetric.iter().find(|(id, _)| id == "y").unwrap().1;
        assert!(
            symmetric_y > symmetric_x,
            "the symmetric default must produce the opposite order: x={symmetric_x}, y={symmetric_y}"
        );
    }

    #[test]
    fn merge_empty_inputs_yield_empty() {
        let merged = merge_results(&[], &[], &SearchConfig::default());
        assert!(merged.is_empty());
    }

    #[test]
    fn merge_scores_sparse_only_doc_absent_from_vector() {
        let search = SearchConfig::default();
        let sparse = vec![sparse_hit("only_sparse")];
        let merged = merge_results(&[], &sparse, &search);

        let k = search.rrf_k as f64;
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].0, "only_sparse");
        let expected = search.sparse_weight / (k + 1.0);
        assert!((merged[0].1 - expected).abs() < 1e-12);
    }
}
