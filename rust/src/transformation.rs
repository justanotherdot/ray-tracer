use crate::matrix::*;

pub fn translation(x: f64, y: f64, z: f64) -> Matrix {
    let mut m = Matrix::empty(4, 4).identity();

    m[(0, 3)] = x;
    m[(1, 3)] = y;
    m[(2, 3)] = z;

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
}
