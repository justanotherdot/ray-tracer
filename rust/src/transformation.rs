use crate::matrix::*;

pub fn translation(x: f64, y: f64, z: f64) -> Matrix {
    let mut m = Matrix::empty(4, 4).identity();

    m[(0, 3)] = x;
    m[(1, 3)] = y;
    m[(2, 3)] = z;

    m
}

pub fn scaling(x: f64, y: f64, z: f64) -> Matrix {
    let mut m = Matrix::empty(4, 4).identity();

    m[(0, 0)] = x;
    m[(1, 1)] = y;
    m[(2, 2)] = z;

    m
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::coordinate::*;

    #[test]
    fn multiplying_by_a_translation_matrix() {
        let transform = translation(5., -3., 2.);
        let p = Point::new(-3., 4., 5.);
        assert_eq!(transform * p, Point::new(2., 1., 7.));
    }

    #[test]
    fn multiplying_by_the_inverse_of_a_translation_matrix() {
        let transform = translation(5., -3., 2.);
        let p = Point::new(-3., 4., 5.);
        assert_eq!(transform.inverse() * p, Point::new(-8., 7., 3.));
    }

    #[test]
    fn translation_does_not_affect_vectors() {
        let transform = translation(5., -3., 2.);
        let v = Vector::new(-3., 4., 5.);
        assert_eq!(transform * v, v);
    }

    #[test]
    fn a_scaling_matrix_applied_to_a_point() {
        let transform = scaling(2., 3., 4.);
        let p = Point::new(-4., 6., 8.);
        assert_eq!(transform * p, Point::new(-8., 18., 32.));
    }

    #[test]
    fn a_scaling_matrix_applied_to_a_vector() {
        let transform = scaling(2., 3., 4.);
        let v = Vector::new(-4., 6., 8.);
        assert_eq!(transform * v, Vector::new(-8., 18., 32.));
    }

    #[test]
    fn multiplying_by_the_inverse_of_a_scaling_matrix() {
        let transform = scaling(2., 3., 4.);
        let v = Vector::new(-4., 6., 8.);
        assert_eq!(transform.inverse() * v, Vector::new(-2., 2., 2.));
    }

    #[test]
    fn reflection_is_scaling_by_a_negative_value() {
        let transform = scaling(-1., 1., 1.);
        let p = Point::new(2., 3., 4.);
        assert_eq!(transform.inverse() * p, Point::new(-2., 3., 4.));
    }
}
