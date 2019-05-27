use smallvec::*;
use std::cmp::PartialEq;
use std::ops::{Index, IndexMut, Mul};

#[allow(dead_code)]
fn identity_matrix_from_dims(num_rows: usize, num_cols: usize) -> Matrix {
    assert!(num_rows == num_cols);
    let dim = num_rows;
    (0..dim).fold(Matrix::new(dim, dim), |mut m, i| {
        m[(i, i)] = 1.;
        m
    })
}

/// Generate an identity matrix with the same dimensions as this Matrix.
fn identity_matrix_from_square_matrix<M>(m: &M) -> Matrix
where
    M: SquareMatrix,
{
    let dim = m.dim();
    (0..dim).fold(Matrix::new(dim, dim), |mut m, i| {
        m[(i, i)] = 1.;
        m
    })
}

#[derive(Debug, Clone)]
pub struct Matrix {
    #[allow(dead_code)]
    dims: (usize, usize),
    #[allow(dead_code)]
    data: SmallVec<[f64; 0]>,
}

impl Matrix {
    pub fn new(num_rows: usize, num_cols: usize) -> Self {
        Matrix {
            dims: (num_rows, num_cols),
            data: smallvec![0.0; num_rows*num_cols],
        }
    }
}

fn naive_approx_equal_float(x: &f64, y: &f64) -> bool {
    const F64_EPSILON: f64 = 0.00001;
    // TODO Needs checks for NaN and ±∞ etc.
    if *x == std::f64::NAN && *y == std::f64::NAN {
        return false;
    }

    (x - y).abs() < F64_EPSILON
}

impl PartialEq for Matrix {
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

pub trait IdentityMatrix
where
    Self: SquareMatrix,
{
    fn identity(self: &Self) -> Matrix;
}

impl IdentityMatrix for Matrix {
    fn identity(&self) -> Matrix {
        identity_matrix_from_square_matrix(self)
    }
}

pub trait SquareMatrix {
    fn dim(&self) -> usize;
    fn from_vec(vec: Vec<f64>) -> Self;
    fn from_nested_vec(vec: Vec<Vec<f64>>) -> Self;
}

impl SquareMatrix for Matrix {
    fn dim(&self) -> usize {
        self.dims.0
    }

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

    fn from_nested_vec(vec: Vec<Vec<f64>>) -> Self {
        let vec: Vec<f64> = vec.into_iter().flatten().collect();
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

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, ixs: (usize, usize)) -> &mut f64 {
        // TODO Convert `assert`s to `Result`s.
        assert!(ixs.0 < self.dims.0);
        assert!(ixs.1 < self.dims.1);
        &mut self.data[(ixs.0 * self.dims.0) + ixs.1]
    }
}

fn matrix_mul(a: Matrix, b: Matrix) -> Matrix {
    let (num_rows, num_cols) = a.dims;

    // TODO Convert `assert`s to `Result`s.
    assert!(num_rows == num_cols);
    assert!(a.dims == b.dims);

    let mut m = Matrix::new(num_rows, num_cols);
    for row in 0..num_rows {
        for col in 0..num_cols {
            m[(row, col)] = (0..num_cols).fold(0.0, |acc, i| acc + a[(row, i)] * b[(i, col)]);
        }
    }
    m
}

impl Mul for Matrix {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        matrix_mul(self, rhs)
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

    #[test]
    fn multiplying_two_matrices() {
        #[rustfmt::skip]
        let m: Matrix = SquareMatrix::from_vec(vec![
            1., 2., 3., 4.,
            5., 6., 7., 8.,
            9., 8., 7., 6.,
            5., 4., 3., 2.,
        ]);

        #[rustfmt::skip]
        let n: Matrix = SquareMatrix::from_vec(vec![
            -2., 1., 2., 3.,
            3., 2., 1., -1.,
            4., 3., 6., 5.,
            1., 2., 7., 8.,
        ]);

        let lhs_mn = m * n;
        #[rustfmt::skip]
        let rhs_mn = SquareMatrix::from_vec(vec![
            20., 22., 50., 48.,
            44., 54., 114., 108.,
            40., 58., 110., 102.,
            16., 26., 46., 42.,
        ]);

        assert_eq!(lhs_mn, rhs_mn);
    }

    #[test]
    fn multiplying_two_matrices_2x2() {
        #[rustfmt::skip]
        let m: Matrix = SquareMatrix::from_vec(vec![
            1., 2.,
            5., 6.,
        ]);

        #[rustfmt::skip]
        let n: Matrix = SquareMatrix::from_vec(vec![
            -2., 1.,
            3.,  2.,
        ]);

        let lhs_mn = m * n;
        #[rustfmt::skip]
        let rhs_mn = SquareMatrix::from_vec(vec![
            4., 5.,
            8., 17.,
        ]);

        assert_eq!(lhs_mn, rhs_mn);
    }

    #[test]
    fn multiplying_two_matrices_3x3() {
        #[rustfmt::skip]
        let m: Matrix = SquareMatrix::from_vec(vec![
            1., 2., 3.,
            5., 6., 7.,
            9., 8., 7.,
        ]);

        #[rustfmt::skip]
        let n: Matrix = SquareMatrix::from_vec(vec![
            -2., 1., 2.,
            3., 2., 1.,
            4., 3., 6.,
        ]);

        let lhs_mn = m * n;
        #[rustfmt::skip]
        let rhs_mn = SquareMatrix::from_vec(vec![
            16., 14., 22.,
            36., 38., 58.,
            34., 46., 68.,
        ]);

        assert_eq!(lhs_mn, rhs_mn);
    }

    #[test]
    #[should_panic]
    fn invalid_matrix_multiplication_01() {
        #[rustfmt::skip]
        let m: Matrix = SquareMatrix::from_vec(vec![
            1., 2., 3.,
            5., 6., 7.,
            9., 8., 7.,
        ]);

        #[rustfmt::skip]
        let n: Matrix = SquareMatrix::from_vec(vec![
            -2., 1., 2., 2.,
            3., 2., 1., 2.,
            4., 3., 6., 2.,
            2., 2., 2., 2.,
        ]);

        let _ = m * n;
    }

    #[test]
    fn matrix_multiplication_is_not_associative() {
        #[rustfmt::skip]
        let m: Matrix = SquareMatrix::from_vec(vec![
            1., 2., 3., 4.,
            5., 6., 7., 8.,
            9., 8., 7., 6.,
            5., 4., 3., 2.,
        ]);

        #[rustfmt::skip]
        let n: Matrix = SquareMatrix::from_vec(vec![
            -2., 1., 2., 3.,
            3., 2., 1., -1.,
            4., 3., 6., 5.,
            1., 2., 7., 8.,
        ]);

        let mn = m.clone() * n.clone();
        let nm = n * m;

        assert!(mn != nm);
    }

    #[test]
    fn identity_matrices_have_the_right_shape() {
        #[rustfmt::skip]
        let m: Matrix = SquareMatrix::from_vec(vec![
            1., 2., 3., 4.,
            5., 6., 7., 8.,
            9., 8., 7., 6.,
            5., 4., 3., 2.,
        ]);

        let actual_ident = m.identity();

        #[rustfmt::skip]
        let expected_ident = SquareMatrix::from_vec(vec![
            1., 0., 0., 0.,
            0., 1., 0., 0.,
            0., 0., 1., 0.,
            0., 0., 0., 1.,
        ]);

        assert_eq!(actual_ident, expected_ident);
    }

    #[test]
    fn multiplying_a_matrix_by_the_identity_matrix() {
        let m: Matrix = SquareMatrix::from_nested_vec(vec![
            vec![1., 2., 3., 4.],
            vec![5., 6., 7., 8.],
            vec![9., 8., 7., 6.],
            vec![5., 4., 3., 2.],
        ]);

        assert_eq!(m.clone() * m.identity(), m);
    }
}
