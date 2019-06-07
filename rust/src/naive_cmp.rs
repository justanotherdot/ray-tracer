pub fn naive_approx_equal_float(x: &f64, y: &f64) -> bool {
    const F64_EPSILON: f64 = 0.00001;
    // TODO Needs checks for NaN and ±∞ etc.
    if *x == std::f64::NAN && *y == std::f64::NAN {
        return false;
    }

    (x - y).abs() < F64_EPSILON
}
