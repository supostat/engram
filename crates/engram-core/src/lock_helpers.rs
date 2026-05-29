//! Poison-recovering accessors for the locks held inside [`ServerState`].
//!
//! Engram runs as a single-user, per-project daemon: one server process owns
//! one database, one index set, and one router. When a thread panics while
//! holding one of these locks, the standard-library lock becomes *poisoned*
//! and every subsequent `.lock()/.read()/.write()` returns `Err`. A raw
//! `.unwrap()` on that result turns a single recoverable hiccup into a
//! cascading crash that takes down the whole daemon and every in-flight
//! client request.
//!
//! For this deployment model recovery is strictly preferable to a cascading
//! panic: the protected state is plain in-memory data, and a poisoned guard
//! still references fully-formed (if possibly partially-updated) state that
//! later operations can read and overwrite. These helpers therefore recover
//! unconditionally via [`PoisonError::into_inner`], returning the *bare*
//! guard. Recovery is total — there is no error path a caller could meaningfully
//! handle — so the helpers deliberately do not wrap the guard in `Result`.

use std::sync::{MutexGuard, RwLockReadGuard, RwLockWriteGuard};

use engram_router::Router;
use engram_storage::Database;

use crate::indexes::IndexSet;
use crate::server::ServerState;

/// Locks the database mutex, recovering the guard if the lock was poisoned.
pub fn lock_db(state: &ServerState) -> MutexGuard<'_, Database> {
    state
        .database
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Locks the router mutex, recovering the guard if the lock was poisoned.
pub fn lock_router(state: &ServerState) -> MutexGuard<'_, Router> {
    state
        .router
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Acquires a shared read guard on the index set, recovering if poisoned.
pub fn read_indexes(state: &ServerState) -> RwLockReadGuard<'_, IndexSet> {
    state
        .indexes
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Acquires an exclusive write guard on the index set, recovering if poisoned.
pub fn write_indexes(state: &ServerState) -> RwLockWriteGuard<'_, IndexSet> {
    state
        .indexes
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}
