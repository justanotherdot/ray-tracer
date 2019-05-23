use smallvec::*;
use std::ops::Index;

struct Matrix {
    #[allow(dead_code)]
    dims: (usize, usize),
    #[allow(dead_code)]
    data: SmallVec<[f64; 0]>,
}

fn naive_approx_equal_float(x: &f64, y: &f64) -> bool {
    const F64_EPSILON: f64 = 0.00001;
    // TODO Needs checks for NaN and ±∞ etc.
    if *x == std::f64::NAN && *y == std::f64::NAN {
        return false;
    }

    (x - y).abs() < F64_EPSILON
}

impl std::cmp::PartialEq for Matrix {
    fn eq(&self, other: &Self) -> bool {
        if other.dims.0 != self.dims.0 || other.dims.1 != self.dims.1 {
            return false;
        }
        self.data
            .iter()
            .zip(&other.data)
            .all(|(x, y)| naive_approx_equal_float(x, y))
    }
}

trait SquareMatrix {
    fn from_vec(vec: Vec<f64>) -> Self;
}

impl SquareMatrix for Matrix {
    // A dependantly typed language could encode this in the type system.
    // Rust is not one of those languages, so instead we'll have do the checks at runtime here.
    fn from_vec(vec: Vec<f64>) -> Self {
        let dim = (vec.len() as f64).log2() as usize;
        assert!(vec.len() == 4 || vec.len() == 9 || vec.len() == 16);
        Matrix {
            dims: (dim, dim),
            data: SmallVec::from_vec(vec),
        }
    }
}

// Not sure if this is weird.
impl Index<(usize, usize)> for Matrix {
    type Output = f64;

    fn index(&self, ixs: (usize, usize)) -> &f64 {
        // TODO Convert `assert`s to `Result`s.
        assert!(ixs.0 < self.dims.0);
        assert!(ixs.1 < self.dims.1);
        &self.data[(ixs.0 * self.dims.0) + ixs.1]
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn constructing_and_inspecting_a_4x4_matrix() {
        #[rustfmt::skip]
        let m: Matrix = SquareMatrix::from_vec(vec![
           1.,   2.,   3.,  4.,
           5.5,  6.5,  7.5, 8.5,
           9.,   10.,  11., 12.,
           13.5, 14.5, 15.5, 16.5,
        ]);
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(0, 3)], 4.0);
        assert_eq!(m[(1, 0)], 5.5);
        assert_eq!(m[(1, 2)], 7.5);
        assert_eq!(m[(2, 2)], 11.0);
        assert_eq!(m[(3, 0)], 13.5);
        assert_eq!(m[(3, 2)], 15.5);
    }

    #[test]
    #[should_panic]
    fn constructing_invalid_matrices_01() {
        #[rustfmt::skip]
        let _: Matrix = SquareMatrix::from_vec(vec![
           1.,   2.,   3.,  4.,
           5.5,  6.5,  7.5, 8.5,
           9.,   10.,  11., 12.,
           13.5, 14.5, 15.5
        ]);
    }

    #[test]
    #[should_panic]
    fn constructing_invalid_matrices_02() {
        #[rustfmt::skip]
        let _: Matrix = SquareMatrix::from_vec(vec![1., 2.]);
    }

    #[test]
    #[should_panic]
    fn constructing_invalid_matrices_03() {
        #[rustfmt::skip]
        let _: Matrix = SquareMatrix::from_vec(vec![1.]);
    }

    #[test]
    fn a_2x2_matrix_ought_to_be_representable() {
        #[rustfmt::skip]
        let m: Matrix = SquareMatrix::from_vec(vec![
            -3., 5.,
            1., -2.,
        ]);
        assert_eq!(m[(0, 0)], -3.);
        assert_eq!(m[(0, 1)], 5.0);
        assert_eq!(m[(1, 0)], 1.0);
        assert_eq!(m[(1, 1)], -2.0);
    }

    #[test]
    fn a_3x3_matrix_ought_to_be_representable() {
        #[rustfmt::skip]
        let m: Matrix = SquareMatrix::from_vec(vec![
            -3., 5., 0.,
            1., -2., -7.,
            0., 1., 1.,
        ]);
        assert_eq!(m[(0, 0)], -3.0);
        assert_eq!(m[(1, 1)], -2.0);
        assert_eq!(m[(2, 2)], 1.0);
    }

    #[test]
    fn matrix_equality_with_identical_matrices() {
        #[rustfmt::skip]
        let m: Matrix = SquareMatrix::from_vec(vec![
            1., 2., 3., 4.,
            5., 6., 7., 8.,
            9., 8., 7., 6.,
            5., 4., 3., 2.,
        ]);

        #[rustfmt::skip]
        let n: Matrix = SquareMatrix::from_vec(vec![
            1., 2., 3., 4.,
            5., 6., 7., 8.,
            9., 8., 7., 6.,
            5., 4., 3., 2.,
        ]);

        assert!(m == n);
    }

    #[test]
    fn matrix_inequality_with_non_identical_matrices() {
        #[rustfmt::skip]
        let m: Matrix = SquareMatrix::from_vec(vec![
            1., 2., 3., 4.,
            5., 6., 7., 8.,
            9., 8., 7., 6.,
            5., 4., 3., 2.,
        ]);

        #[rustfmt::skip]
        let n: Matrix = SquareMatrix::from_vec(vec![
            -1., -2., -3., -4.,
            -5., -6., -7., -8.,
            -9., -8., -7., -6.,
            -5., -4., -3., -2.,
        ]);

        assert!(m != n);
    }

    #[test]
    fn matrix_inequality_with_matrices_of_differing_dims() {
        #[rustfmt::skip]
        let m: Matrix = SquareMatrix::from_vec(vec![
            1., 2., 3., 4.,
            5., 6., 7., 8.,
            9., 8., 7., 6.,
            5., 4., 3., 2.,
        ]);

        #[rustfmt::skip]
        let n: Matrix = SquareMatrix::from_vec(vec![
            -1., -2., -3.,
            -5., -6., -7.,
            -9., -8., -7.,
        ]);

        assert!(m != n);
    }
}
