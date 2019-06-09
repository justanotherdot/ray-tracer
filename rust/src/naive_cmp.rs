use std::cmp::Ordering;

pub fn naive_approx_equal_float(x: &f64, y: &f64) -> bool {
    const F64_EPSILON: f64 = 0.00001;
    // TODO Needs checks for NaN and ±∞ etc.
    if *x == std::f64::NAN && *y == std::f64::NAN {
        return false;
    }

    (x - y).abs() < F64_EPSILON
}

pub fn naive_approx_float_cmp(x: &f64, y: &f64) -> Ordering {
    const F64_EPSILON: f64 = 0.00001;
    // TODO Needs checks for NaN and ±∞ etc.
    if *x == std::f64::NAN && *y == std::f64::NAN {
        return Ordering::Greater;
        //panic!("NaN or inf discovered while comparing f64");
    }

    let delta = (x - y).abs();
    if delta < F64_EPSILON {
        Ordering::Equal
    } else {
        if x > y {
            Ordering::Greater
        } else {
            Ordering::Less
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn naive_approx_equal_float_works() {
        assert!(naive_approx_equal_float(&(0.15 + 0.15), &(0.1 + 0.2)));
    }

    #[test]
    fn naive_approx_float_cmp_works() {
        assert_eq!(
            naive_approx_float_cmp(&(0.15 + 0.15), &(0.1 + 0.2)),
            Ordering::Equal
        );
    }
}
