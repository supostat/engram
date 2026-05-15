//! Config file loading and layered merge.
//!
//! `Config::load()` resolves three layers — global `~/.engram/engram.toml`,
//! project-local `<project>/.engram/engram.toml`, and `ENGRAM_*` env
//! overrides — with priority `env > project-local > global`. This module
//! owns the file-tree side: it reads each TOML file into a generic
//! `toml::Value` tree, deep-merges the project-local tree over the global
//! one, and enforces the secret invariant that `api_key` values always come
//! from the global layer. `config.rs` keeps the `Config` shape and the
//! `Config::load()` orchestration that drives these functions.

use std::path::Path;

use toml::Value;

use crate::config::{ENGRAM_CONFIG_SUBPATH, home_directory, resolve_project_dir};
use crate::error::CoreError;

/// Reads the global `~/.engram/engram.toml` into a generic TOML tree.
///
/// Returns `Ok(None)` when `HOME` is unset or the file is absent — both mean
/// "no global layer", not an error. A present-but-malformed file is a
/// `ConfigParseError`.
pub(crate) fn load_global_config_tree() -> Result<Option<Value>, CoreError> {
    let Some(home) = home_directory() else {
        return Ok(None);
    };
    let global_config_path = Path::new(&home).join(ENGRAM_CONFIG_SUBPATH);
    parse_config_tree_if_present(&global_config_path)
}

/// Reads the project-local `<project>/.engram/engram.toml` into a generic
/// TOML tree, discovering the project root by walking up from the current
/// directory like `.git`.
///
/// Returns `Ok(None)` when the current directory is unavailable, no project
/// `.engram/` marker exists, or the project has no `engram.toml`. A
/// present-but-malformed project-local file is a `ConfigParseError`.
pub(crate) fn load_project_config_tree() -> Result<Option<Value>, CoreError> {
    let Ok(current_directory) = std::env::current_dir() else {
        return Ok(None);
    };
    let Ok(project_directory) = resolve_project_dir(&current_directory, None) else {
        return Ok(None);
    };
    let project_config_path = project_directory.join(ENGRAM_CONFIG_SUBPATH);
    parse_config_tree_if_present(&project_config_path)
}

/// Parses the TOML file at `path` into a generic tree.
///
/// A missing file yields `Ok(None)`. Any other read failure, or a parse
/// failure, yields `ConfigParseError` so the caller never silently proceeds
/// on a config it could not read.
pub(crate) fn parse_config_tree_if_present(path: &Path) -> Result<Option<Value>, CoreError> {
    let contents = match std::fs::read_to_string(path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(CoreError::ConfigParseError(format!(
                "failed to read {}: {error}",
                path.display()
            )));
        }
    };
    let tree = contents
        .parse::<Value>()
        .map_err(|error| CoreError::ConfigParseError(error.to_string()))?;
    Ok(Some(tree))
}

/// Deep-merges `overlay` into `base`, with `overlay` winning field-by-field.
///
/// Two tables merge recursively per key; any other pairing (scalar, array, or
/// mismatched types) replaces the `base` slot wholesale.
pub(crate) fn deep_merge(base: &mut Value, overlay: Value) {
    match (base, overlay) {
        (Value::Table(base_table), Value::Table(overlay_table)) => {
            for (key, overlay_value) in overlay_table {
                match base_table.get_mut(&key) {
                    Some(base_value) => deep_merge(base_value, overlay_value),
                    None => {
                        base_table.insert(key, overlay_value);
                    }
                }
            }
        }
        (base_slot, overlay_value) => *base_slot = overlay_value,
    }
}

/// Clones `<section>.api_key` out of `tree`, if present.
pub(crate) fn secret_at(tree: &Value, section: &str) -> Option<Value> {
    tree.get(section)
        .and_then(|section_value| section_value.get("api_key"))
        .cloned()
}

/// Restores the `api_key` of `<section>` in `tree` to `global_secret`.
///
/// When `global_secret` is `Some`, the key is inserted (overwriting any
/// project-local value). When `None`, any project-local key is removed, so a
/// project-local config can never introduce a secret the global layer lacks.
pub(crate) fn restore_secret(tree: &mut Value, section: &str, global_secret: Option<Value>) {
    let Some(Value::Table(section_table)) = tree.get_mut(section) else {
        return;
    };
    match global_secret {
        Some(secret) => {
            section_table.insert("api_key".to_string(), secret);
        }
        None => {
            section_table.remove("api_key");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deep_merge_overlay_wins_on_type_mismatch() {
        let mut base = toml::toml! {
            [embedding]
            provider = "voyage"
            model = "voyage-4"
            dimension = 1024
            tags = ["a", "b"]
        }
        .into();
        let overlay: Value = toml::toml! {
            // Table-typed slot replaced by a scalar.
            embedding = 7
        }
        .into();

        deep_merge(&mut base, overlay);

        // The whole `[embedding]` table is replaced wholesale by the scalar.
        assert_eq!(base.get("embedding"), Some(&Value::Integer(7)));
    }

    #[test]
    fn deep_merge_replaces_scalar_with_table() {
        let mut base = toml::toml! {
            [embedding]
            provider = "voyage"
        }
        .into();
        let overlay: Value = toml::toml! {
            [embedding.provider]
            nested = true
        }
        .into();

        deep_merge(&mut base, overlay);

        // The scalar `provider = "voyage"` is replaced wholesale by a table.
        let provider = base
            .get("embedding")
            .and_then(|section| section.get("provider"))
            .expect("provider slot present");
        assert_eq!(provider.get("nested"), Some(&Value::Boolean(true)));
    }

    #[test]
    fn deep_merge_replaces_array_wholesale() {
        let mut base = toml::toml! {
            [embedding]
            tags = ["global-a", "global-b", "global-c"]
        }
        .into();
        let overlay: Value = toml::toml! {
            [embedding]
            tags = ["project-only"]
        }
        .into();

        deep_merge(&mut base, overlay);

        let tags = base
            .get("embedding")
            .and_then(|section| section.get("tags"))
            .and_then(Value::as_array)
            .expect("tags array present");
        // Arrays are not concatenated — the overlay array replaces the base array.
        assert_eq!(tags.len(), 1);
        assert_eq!(tags[0].as_str(), Some("project-only"));
    }

    #[test]
    fn deep_merge_inserts_key_present_only_in_overlay() {
        let mut base = toml::toml! {
            [embedding]
            provider = "voyage"
        }
        .into();
        let overlay: Value = toml::toml! {
            [embedding]
            hyde_threshold = 9

            [hnsw]
            ef_search = 80
        }
        .into();

        deep_merge(&mut base, overlay);

        // Key absent in base, present in overlay — inserted into the existing table.
        assert_eq!(
            base.get("embedding")
                .and_then(|section| section.get("hyde_threshold")),
            Some(&Value::Integer(9))
        );
        // Section absent in base entirely — inserted at the top level.
        assert_eq!(
            base.get("hnsw")
                .and_then(|section| section.get("ef_search")),
            Some(&Value::Integer(80))
        );
    }

    #[test]
    fn restore_secret_ignores_absent_section() {
        let mut tree: Value = toml::toml! {
            [embedding]
            provider = "voyage"
        }
        .into();
        let before = tree.clone();

        // `llm` section is absent — the early return must leave the tree untouched.
        restore_secret(&mut tree, "llm", Some(Value::String("global-key".into())));

        assert_eq!(tree, before);
    }

    #[test]
    fn restore_secret_inserts_global_key() {
        let mut tree: Value = toml::toml! {
            [embedding]
            provider = "voyage"
            api_key = "project-local-key"
        }
        .into();

        restore_secret(
            &mut tree,
            "embedding",
            Some(Value::String("global-key".into())),
        );

        assert_eq!(
            tree.get("embedding")
                .and_then(|section| section.get("api_key")),
            Some(&Value::String("global-key".into()))
        );
    }

    #[test]
    fn restore_secret_removes_key_when_global_absent() {
        let mut tree: Value = toml::toml! {
            [embedding]
            provider = "voyage"
            api_key = "project-local-key"
        }
        .into();

        restore_secret(&mut tree, "embedding", None);

        assert_eq!(
            tree.get("embedding")
                .and_then(|section| section.get("api_key")),
            None
        );
    }
}
