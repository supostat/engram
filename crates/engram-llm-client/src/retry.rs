use crate::error::ApiError;

pub struct RetryConfig {
    pub max_retries: u32,
    pub initial_backoff_ms: u64,
    pub max_backoff_ms: u64,
    pub backoff_multiplier: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff_ms: 100,
            max_backoff_ms: 10_000,
            backoff_multiplier: 2.0,
        }
    }
}

pub fn compute_backoff(config: &RetryConfig, attempt: u32) -> u64 {
    let backoff = config.initial_backoff_ms as f64
        * config.backoff_multiplier.powi(attempt as i32);
    if !backoff.is_finite() {
        return config.max_backoff_ms;
    }
    (backoff as u64).min(config.max_backoff_ms)
}

pub fn execute_with_retry<F, T>(config: &RetryConfig, operation: F) -> Result<T, ApiError>
where
    F: Fn() -> Result<T, ApiError>,
{
    let mut last_error: Option<ApiError> = None;

    for attempt in 0..=config.max_retries {
        match operation() {
            Ok(value) => return Ok(value),
            Err(error) => {
                if !error.is_retryable() {
                    return Err(error);
                }
                last_error = Some(error);
                if attempt < config.max_retries {
                    let backoff_ms = compute_backoff(config, attempt);
                    std::thread::sleep(std::time::Duration::from_millis(backoff_ms));
                }
            }
        }
    }

    Err(last_error.expect("at least one attempt must have been made"))
}
