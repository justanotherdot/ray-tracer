use crate::coordinate::{Point, Vector};
use smallvec::*;
use std::cmp::PartialEq;
use std::ops::{Index, IndexMut, Mul};

#[allow(dead_code)]
fn identity_matrix_from_dims(num_rows: usize, num_cols: usize) -> Matrix {
    assert!(num_rows == num_cols);
    let dim = num_rows;
    (0..dim).fold(Matrix::empty(dim, dim), |mut m, i| {
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
    (0..dim).fold(Matrix::empty(dim, dim), |mut m, i| {
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
    pub fn empty(num_rows: usize, num_cols: usize) -> Self {
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
    fn identity(&self) -> Matrix;
}

impl IdentityMatrix for Matrix {
    fn identity(&self) -> Matrix {
        identity_matrix_from_square_matrix(self)
    }
}

pub trait SquareMatrix {
    fn from_vec(vec: Vec<f64>) -> Self;
    fn from_nested_vec(vec: Vec<Vec<f64>>) -> Self;
    fn dim(&self) -> usize;
    fn transpose(&self) -> Self;
    fn determinant(&self) -> i64;
    fn submatrix(&self, exc_row: usize, exc_col: usize) -> Self;
    fn minor(&self, exc_row: usize, exc_col: usize) -> i64;
    fn cofactor(&self, exc_row: usize, exc_col: usize) -> i64;
    fn is_invertible(&self) -> bool;
    fn inverse(&self) -> Self;
}

impl SquareMatrix for Matrix {
    fn dim(&self) -> usize {
        self.dims.0
    }

    // TODO One day having instances of From for SquareMatrix
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

    // TODO One day having instances of From for SquareMatrix
    fn from_nested_vec(vec: Vec<Vec<f64>>) -> Self {
        let vec: Vec<f64> = vec.into_iter().flatten().collect();
        let dim = (vec.len() as f64).log2() as usize;
        assert!(vec.len() == 4 || vec.len() == 9 || vec.len() == 16);
        Matrix {
            dims: (dim, dim),
            data: SmallVec::from_vec(vec),
        }
    }

    fn transpose(&self) -> Self {
        let dim = self.dim();
        let mut m = self.clone();
        for row in 0..dim {
            for col in 0..dim {
                m[(col, row)] = self[(row, col)];
            }
        }
        m
    }

    fn determinant(&self) -> i64 {
        let m = self;
        if m.dim() == 2 {
            let a = m[(0, 0)];
            let b = m[(0, 1)];
            let c = m[(1, 0)];
            let d = m[(1, 1)];

            (a * d - b * c).round() as i64
        } else {
            let mut det = 0;
            for col in 0..m.dim() {
                det += (m[(0, col)] as i64) * m.cofactor(0, col);
            }
            det
        }
    }

    /// submatrix deletes exactly one row and one column,
    /// effectively reducing the dimension by one.
    fn submatrix(&self, exc_row: usize, exc_col: usize) -> Self {
        let dim = self.dim() - 1;
        let mut m: Matrix = Matrix::empty(dim, dim);
        // TODO Yuck.
        let mut target_row = 0;
        let mut target_col = 0;
        for row in 0..self.dim() {
            for col in 0..self.dim() {
                if row == exc_row || col == exc_col {
                    continue;
                }

                m[(target_row, target_col)] = self[(row, col)];
                target_col = (target_col + 1) % dim;
            }
            if row != exc_row {
                target_row += 1;
            }
        }
        m
    }

    fn minor(&self, exc_row: usize, exc_col: usize) -> i64 {
        let submatrix = self.submatrix(exc_row, exc_col);
        submatrix.determinant()
    }

    fn cofactor(&self, exc_row: usize, exc_col: usize) -> i64 {
        let factor = if (exc_row + exc_col) % 2 == 1 { -1 } else { 1 };
        factor * self.minor(exc_row, exc_col)
    }

    fn is_invertible(&self) -> bool {
        self.determinant() != 0
    }

    fn inverse(&self) -> Self {
        assert!(self.is_invertible());

        let mut copy = Matrix::empty(self.dim(), self.dim());

        let det = self.determinant() as f64;
        for row in 0..copy.dim() {
            for col in 0..copy.dim() {
                copy[(col, row)] = self.cofactor(row, col) as f64 / det;
            }
        }

        copy
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

pub fn matrix_mul(a: &Matrix, b: &Matrix) -> Matrix {
    let (num_rows, num_cols) = a.dims;

    // TODO Convert `assert`s to `Result`s.
    assert!(num_rows == num_cols);
    assert!(a.dims == b.dims);

    let mut m = Matrix::empty(num_rows, num_cols);
    for row in 0..num_rows {
        for col in 0..num_cols {
            m[(row, col)] = (0..num_cols).fold(0.0, |acc, i| acc + a[(row, i)] * b[(i, col)]);
        }
    }
    m
}

// n.b. This is just a hack to potentially avoid a lot of costly allocations.
pub fn matrix_mul_mut(a: &Matrix, b: &Matrix, m: &mut Matrix) {
    let (num_rows, num_cols) = a.dims;

    // TODO Convert `assert`s to `Result`s.
    assert!(num_rows == num_cols);
    assert!(a.dims == b.dims);

    for row in 0..num_rows {
        for col in 0..num_cols {
            m[(row, col)] = (0..num_cols).fold(0.0, |acc, i| acc + a[(row, i)] * b[(i, col)]);
        }
    }
}

impl Mul for Matrix {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        matrix_mul(&self, &rhs)
    }
}

impl Mul<Vector> for Matrix {
    type Output = Vector;

    fn mul(self, rhs: Self::Output) -> Self::Output {
        assert!(self.dims.1 == rhs.len());
        let mut v = Vector::new(0., 0., 0.);
        for row in 0..self.dims.0 {
            for col in 0..self.dims.1 {
                v[row] += rhs[col] * self[(row, col)];
            }
        }
        v
    }
}

impl Mul<Point> for Matrix {
    type Output = Point;

    fn mul(self, rhs: Self::Output) -> Self::Output {
        assert!(self.dims.1 == rhs.len());
        let mut p = Point::new(0., 0., 0.);
        for row in 0..self.dims.0 {
            for col in 0..self.dims.1 {
                p[row] += rhs[col] * self[(row, col)];
            }
        }
        p.w = 1.0;
        p
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::coordinate::*;

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

        assert_eq!(&matrix_mul(&m, &m.identity()), &m);
    }

    #[test]
    fn multiplying_matrices_mutably() {
        let m: Matrix = SquareMatrix::from_nested_vec(vec![
            vec![1., 2., 3., 4.],
            vec![5., 6., 7., 8.],
            vec![9., 8., 7., 6.],
            vec![5., 4., 3., 2.],
        ]);

        let mut n = Matrix::empty(m.dim(), m.dim());

        matrix_mul_mut(&m, &m.identity(), &mut n);
        assert_eq!(&m, &n);
    }

    #[test]
    fn transposing_a_matrix() {
        let m: Matrix = SquareMatrix::from_nested_vec(vec![
            vec![0., 9., 3., 0.],
            vec![9., 8., 0., 8.],
            vec![1., 8., 5., 3.],
            vec![0., 0., 5., 8.],
        ]);

        let n: Matrix = SquareMatrix::from_nested_vec(vec![
            vec![0., 9., 1., 0.],
            vec![9., 8., 8., 0.],
            vec![3., 0., 5., 5.],
            vec![0., 8., 3., 8.],
        ]);

        assert_eq!(m.transpose(), n);
    }

    #[test]
    fn transposing_the_identity_matrix() {
        let identity = identity_matrix_from_dims(3, 3);

        assert_eq!(identity.transpose(), identity);
    }

    #[test]
    fn calculating_the_determinant_of_a_2x2_matrix() {
        #[rustfmt::skip]
        let m: Matrix = SquareMatrix::from_nested_vec(vec![
            vec![1., 5.],
            vec![-3., 2.],
        ]);

        assert_eq!(m.determinant(), 17);
    }

    #[test]
    fn a_submatrix_of_a_3x3_matrix_is_a_2x2_matrix() {
        #[rustfmt::skip]
        let m: Matrix = SquareMatrix::from_nested_vec(vec![
            vec![1., 5., 0.],
            vec![-3., 2., 7.],
            vec![0., 6., -3.],
        ]);

        #[rustfmt::skip]
        let n: Matrix = SquareMatrix::from_nested_vec(vec![
            vec![-3., 2.],
            vec![0., 6.],
        ]);

        assert_eq!(m.submatrix(0, 2), n);
    }

    #[test]
    fn a_submatrix_of_a_4x4_matrix_is_a_3x3_matrix() {
        #[rustfmt::skip]
        let m: Matrix = SquareMatrix::from_nested_vec(vec![
            vec![-6., 1., 1., 6.],
            vec![-8., 5., 8., 6.],
            vec![-1., 0., 8., 2.],
            vec![-7., 1., -1., 1.],
        ]);

        #[rustfmt::skip]
        let n: Matrix = SquareMatrix::from_nested_vec(vec![
            vec![-6., 1., 6.],
            vec![-8., 8., 6.],
            vec![-7., -1., 1.],
        ]);

        assert_eq!(m.submatrix(2, 1), n);
    }

    #[test]
    fn calculating_a_minor_of_a_3x3_matrix() {
        #[rustfmt::skip]
        let m: Matrix = SquareMatrix::from_nested_vec(vec![
            vec![3., 5., 0.],
            vec![2., -1., -7.],
            vec![6., -1., 5.],
        ]);

        assert_eq!(m.submatrix(1, 0).determinant(), 25);
        assert_eq!(m.minor(1, 0), 25);
        assert_eq!(m.minor(1, 0), m.submatrix(1, 0).determinant());
    }

    #[test]
    fn calculating_a_cofactor_of_a_3x3_matrix() {
        #[rustfmt::skip]
        let m: Matrix = SquareMatrix::from_nested_vec(vec![
            vec![3., 5., 0.],
            vec![2., -1., -7.],
            vec![6., -1., 5.],
        ]);

        assert_eq!(m.minor(0, 0), -12);
        assert_eq!(m.cofactor(0, 0), -12);
        assert_eq!(m.minor(1, 0), 25);
        assert_eq!(m.cofactor(1, 0), -25);
    }

    #[test]
    fn calculating_the_determinant_of_a_3x3_matrix() {
        #[rustfmt::skip]
        let m: Matrix = SquareMatrix::from_nested_vec(vec![
            vec![1., 2., 6.],
            vec![-5., 8., -4.],
            vec![2., 6., 4.],
        ]);

        assert_eq!(m.cofactor(0, 0), 56);
        assert_eq!(m.cofactor(0, 1), 12);
        assert_eq!(m.cofactor(0, 2), -46);
        assert_eq!(m.determinant(), -196);
    }

    #[test]
    fn calculating_the_determinant_of_a_4x4_matrix() {
        #[rustfmt::skip]
        let m: Matrix = SquareMatrix::from_nested_vec(vec![
            vec![-2., -8., 3., 5.],
            vec![-3., 1., 7., 3.],
            vec![1., 2., -9., 6.],
            vec![-6., 7., 7., -9.],
        ]);

        assert_eq!(m.cofactor(0, 0), 690);
        assert_eq!(m.cofactor(0, 1), 447);
        assert_eq!(m.cofactor(0, 2), 210);
        assert_eq!(m.cofactor(0, 3), 51);
        assert_eq!(m.determinant(), -4071);
    }

    #[test]
    fn testing_an_invertible_matrix_for_invertibility() {
        #[rustfmt::skip]
        let m: Matrix = SquareMatrix::from_nested_vec(vec![
            vec![6., 4., 4., 4.],
            vec![5., 5., 7., 6.],
            vec![4., -9., 3., -7.],
            vec![9., 1., 7., -6.],
        ]);

        assert_eq!(m.determinant(), -2120);
        assert_eq!(m.is_invertible(), true);
    }

    #[test]
    fn testing_a_noninvertible_matrix_for_invertibility() {
        #[rustfmt::skip]
        let m: Matrix = SquareMatrix::from_nested_vec(vec![
            vec![-4., 2., -2., -3.],
            vec![9., 6., 2., 6.],
            vec![0., -5., 1., -5.],
            vec![0., 0., 0., 0.],
        ]);

        assert_eq!(m.determinant(), 0);
        assert_eq!(m.is_invertible(), false);
    }

    #[test]
    fn calculating_the_inverse_of_a_matrix() {
        #[rustfmt::skip]
        let m: Matrix = SquareMatrix::from_nested_vec(vec![
            vec![-5., 2., 6., -8.],
            vec![1., -5., 1., 8.],
            vec![7., 7., -6., -7.],
            vec![1., -3., 7., 4.],
        ]);

        let n = m.inverse();

        assert_eq!(m.determinant(), 532);
        assert_eq!(m.cofactor(2, 3), -160);
        assert_eq!(n[(3, 2)], -160. / 532.);
        assert_eq!(m.cofactor(3, 2), 105);
        assert_eq!(n[(2, 3)], 105. / 532.);

        let expected_n: Matrix = SquareMatrix::from_nested_vec(vec![
            vec![0.21805, 0.45113, 0.24060, -0.04511],
            vec![-0.80827, -1.45677, -0.44361, 0.52068],
            vec![-0.07895, -0.22368, -0.05263, 0.19737],
            vec![-0.52256, -0.81391, -0.30075, 0.30639],
        ]);

        assert_eq!(n, expected_n);
    }

    #[test]
    fn calculating_the_inverse_of_another_matrix() {
        #[rustfmt::skip]
        let m: Matrix = SquareMatrix::from_nested_vec(vec![
            vec![8., -5., 9., 2.],
            vec![7., 5., 6., 1.],
            vec![-6., 0., 9., 6.],
            vec![-3., 0., -9., -4.],
        ]);

        #[rustfmt::skip]
        let m_inverse: Matrix = SquareMatrix::from_nested_vec(vec![
            vec![-0.15385, -0.15385, -0.28205, -0.53846],
            vec![-0.07692,  0.12308,  0.02564,  0.03077],
            vec![ 0.35897,  0.35897,  0.43590,  0.92308],
            vec![-0.69231, -0.69231, -0.76923, -1.92308],
        ]);

        assert_eq!(m.inverse(), m_inverse);
    }

    #[test]
    fn calculating_the_inverse_of_a_third_matrix() {
        #[rustfmt::skip]
        let m: Matrix = SquareMatrix::from_nested_vec(vec![
            vec![9., 3., 0., 9.],
            vec![-5., -2., -6., -3.],
            vec![-4., 9., 6., 4.],
            vec![-7., 6., 6., 2.],
        ]);

        #[rustfmt::skip]
        let m_inverse: Matrix = SquareMatrix::from_nested_vec(vec![
            vec![-0.04074, -0.07778,  0.14444, -0.22222],
            vec![-0.07778,  0.03333,  0.36667, -0.33333],
            vec![-0.02901, -0.14630, -0.10926,  0.12963],
            vec![ 0.17778,  0.06667, -0.26667,  0.33333],
        ]);

        assert_eq!(m.inverse(), m_inverse);
    }

    #[test]
    fn multiplying_a_product_by_its_inverse() {
        #[rustfmt::skip]
        let a: Matrix = SquareMatrix::from_nested_vec(vec![
            vec![ 3., -9.,  7.,  3.],
            vec![ 3., -8.,  2., -9.],
            vec![-4.,  4.,  4.,  1.],
            vec![-6.,  5., -1.,  1.],
        ]);

        #[rustfmt::skip]
        let b: Matrix = SquareMatrix::from_nested_vec(vec![
            vec![ 8.,  2.,  2.,  2.],
            vec![ 3., -1.,  7.,  0.],
            vec![ 7.,  0.,  5.,  4.],
            vec![ 6., -2.,  0.,  5.],
        ]);

        let c = a.clone() * b.clone();
        assert_eq!(c * b.inverse(), a);
    }

    #[test]
    fn a_matrix_multiplied_by_a_point() {
        #[rustfmt::skip]
        let m: Matrix = SquareMatrix::from_nested_vec(vec![
            vec![1., 2., 3., 4.],
            vec![2., 4., 4., 2.],
            vec![8., 6., 4., 1.],
            vec![0., 0., 0., 1.],
        ]);

        #[rustfmt::skip]
        let b = Point::new(1., 2., 3.);

        assert_eq!(m * b, Point::new(18., 24., 33.));
    }

    #[test]
    fn a_matrix_multiplied_by_a_vector() {
        #[rustfmt::skip]
        let m: Matrix = SquareMatrix::from_nested_vec(vec![
            vec![1., 2., 3., 4.],
            vec![2., 4., 4., 2.],
            vec![8., 6., 4., 1.],
            vec![0., 0., 0., 1.],
        ]);

        #[rustfmt::skip]
        let b = Vector::new(1., 2., 3.);

        assert_eq!(m * b, Vector::new(14., 22., 32.));
    }
}
