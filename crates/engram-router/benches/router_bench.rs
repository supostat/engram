use criterion::{Criterion, criterion_group, criterion_main};
use engram_router::{Mode, Router};

fn bench_choose_action(criterion: &mut Criterion) {
    let router = Router::new(0.1, 0.1);
    criterion.bench_function("choose_action", |bencher| {
        bencher.iter(|| router.decide(Mode::Coding, 0.5));
    });
    // Target: < 1us
}

fn bench_mode_detection_short(criterion: &mut Criterion) {
    criterion.bench_function("mode_detect_short", |bencher| {
        bencher.iter(|| Mode::detect("fix the bug in the trace"));
    });
}

fn bench_mode_detection_long(criterion: &mut Criterion) {
    let long_text = "implement the feature module with proper error handling and testing, \
        then review the code quality and refactor if needed, \
        estimate the timeline and plan the deployment schedule"
        .to_string();
    criterion.bench_function("mode_detect_long", |bencher| {
        bencher.iter(|| Mode::detect(&long_text));
    });
}

fn bench_update_and_choose_1000x(criterion: &mut Criterion) {
    criterion.bench_function("update_and_choose_1000x", |bencher| {
        bencher.iter(|| {
            let mut router = Router::new(0.1, 0.1);
            for i in 0..1000u32 {
                let rng = (i as f32) / 1000.0;
                let decision = router.decide(Mode::Debug, rng);
                router.update(Mode::Debug, &decision, 0.8);
            }
        });
    });
    // Target: < 1ms for 1000 iterations
}

fn bench_convergence(criterion: &mut Criterion) {
    criterion.bench_function("convergence_100_sequences", |bencher| {
        bencher.iter(|| {
            let mut router = Router::new(0.1, 0.0);
            for i in 0..100u32 {
                let decision = router.decide(Mode::Coding, 0.5);
                let reward = if i % 2 == 0 { 0.9 } else { 0.1 };
                router.update(Mode::Coding, &decision, reward);
            }
        });
    });
}

criterion_group!(
    benches,
    bench_choose_action,
    bench_mode_detection_short,
    bench_mode_detection_long,
    bench_update_and_choose_1000x,
    bench_convergence
);
criterion_main!(benches);
