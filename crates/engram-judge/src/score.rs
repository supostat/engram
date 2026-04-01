pub struct JudgeInput {
    pub context: String,
    pub action: String,
    pub result: String,
    pub days_since_update: f64,
    pub used_count: u64,
}

pub struct JudgeScore {
    pub score: f32,
    pub reason: String,
    pub degraded: bool,
}
