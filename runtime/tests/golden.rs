use std::path::PathBuf;

use stems_runtime::audio::read_stereo;
use stems_runtime::{STEM_NAMES, Separator};

fn signal_to_error_db(estimate: &[f32], expected: &[f32]) -> f64 {
    let signal: f64 = expected.iter().map(|value| (*value as f64).powi(2)).sum();
    let noise: f64 = estimate.iter().zip(expected).map(|(a, b)| ((*a - *b) as f64).powi(2)).sum();
    10.0 * ((signal + 1e-12) / (noise + 1e-12)).log10()
}

#[test]
fn matches_the_python_pipeline() {
    let (Some(model), Some(golden)) = (std::env::var_os("STEMS_MODEL"), std::env::var_os("STEMS_GOLDEN")) else {
        eprintln!("set STEMS_MODEL and STEMS_GOLDEN to run this test");
        return;
    };
    let golden = PathBuf::from(golden);
    let [left, right] = read_stereo(&golden.join("excerpt.wav")).unwrap();
    let mut separator = Separator::load(&PathBuf::from(model), 8).unwrap();
    let stems = separator.separate(&left, &right, 2).unwrap();
    for (name, channels) in STEM_NAMES.iter().zip(stems.iter()) {
        let [expected_left, expected_right] = read_stereo(&golden.join(format!("expected_{name}.wav"))).unwrap();
        let left_db = signal_to_error_db(&channels[0], &expected_left);
        let right_db = signal_to_error_db(&channels[1], &expected_right);
        eprintln!("{name}: left {left_db:.1} dB, right {right_db:.1} dB");
        assert!(left_db > 40.0 && right_db > 40.0, "{name} differs from the Python pipeline");
    }
}
