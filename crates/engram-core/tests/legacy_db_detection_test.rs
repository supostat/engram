use tempfile::tempdir;

use engram_core::error::CoreError;
use engram_core::server;

#[test]
fn check_legacy_database_detects_legacy() {
    let project = tempdir().expect("project");
    let home = tempdir().expect("home");
    let home_engram = home.path().join(".engram");
    std::fs::create_dir_all(&home_engram).expect("create home .engram");
    std::fs::write(home_engram.join("engram.db"), b"stub").expect("write legacy db");

    let result = server::check_legacy_database(project.path(), home.path());
    let error = result.expect_err("legacy db should trigger error");
    match &error {
        CoreError::LegacyDatabaseDetected {
            legacy_path,
            project_path,
        } => {
            assert!(legacy_path.contains(".engram/engram.db"));
            assert_eq!(project_path, &project.path().to_string_lossy().to_string());
        }
        other => panic!("expected LegacyDatabaseDetected, got {other:?}"),
    }
    assert!(error.to_string().contains("[6017]"));
}

#[test]
fn check_legacy_database_ok_when_project_has_db() {
    let project = tempdir().expect("project");
    let home = tempdir().expect("home");
    let project_engram = project.path().join(".engram");
    std::fs::create_dir_all(&project_engram).expect("create project .engram");
    std::fs::write(project_engram.join("engram.db"), b"stub").expect("write project db");

    let home_engram = home.path().join(".engram");
    std::fs::create_dir_all(&home_engram).expect("create home .engram");
    std::fs::write(home_engram.join("engram.db"), b"stub").expect("write home db");

    let result = server::check_legacy_database(project.path(), home.path());
    assert!(result.is_ok(), "project db present should pass");
}

#[test]
fn check_legacy_database_ok_when_no_legacy() {
    let project = tempdir().expect("project");
    let home = tempdir().expect("home");
    let result = server::check_legacy_database(project.path(), home.path());
    assert!(result.is_ok(), "no db anywhere should pass");
}
