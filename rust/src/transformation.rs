use crate::matrix::*;

// Part of the problem here is we have an implicit assumption that matrices
// that perform transformations in this module are all 4x4 matrices but we have
// no nice way of ensuring that this bound is checked.

// TODO Might be worth renaming this to MatrixTransformation
// so that it's clearer.
pub struct Transformation(Vec<Matrix>);

// Using this API may, unfortunately, be a bit costly.
// But that hasn't been checked in any sane way (no benchmarks) so it might be fine!
impl Transformation {
    pub fn new() -> Self {
        Transformation(vec![])
    }

    pub fn build(self) -> Matrix {
        self.0
            .into_iter()
            .fold(Matrix::empty(4, 4).identity(), |m, t| t * m)
    }

    pub fn translate(self, x: f64, y: f64, z: f64) -> Self {
        let mut ts = self.0;
        ts.push(translation(x, y, z));
        Transformation(ts)
    }

    pub fn scale(self, x: f64, y: f64, z: f64) -> Self {
        let mut ts = self.0;
        ts.push(scaling(x, y, z));
        Transformation(ts)
    }

    pub fn rotate_x(self, r: f64) -> Self {
        let mut ts = self.0;
        ts.push(rotation_x(r));
        Transformation(ts)
    }

    pub fn rotate_y(self, r: f64) -> Self {
        let mut ts = self.0;
        ts.push(rotation_y(r));
        Transformation(ts)
    }

    pub fn rotate_z(self, r: f64) -> Self {
        let mut ts = self.0;
        ts.push(rotation_z(r));
        Transformation(ts)
    }

    pub fn shear(self, xy: f64, xz: f64, yx: f64, yz: f64, zx: f64, zy: f64) -> Self {
        let mut ts = self.0;
        ts.push(shearing(xy, xz, yx, yz, zx, zy));
        Transformation(ts)
    }
}

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

pub fn rotation_x(r: f64) -> Matrix {
    let mut m = Matrix::empty(4, 4).identity();

    let cos_r = r.cos();
    let sin_r = r.sin();

    m[(1, 1)] = cos_r;
    m[(1, 2)] = -sin_r;
    m[(2, 1)] = sin_r;
    m[(2, 2)] = cos_r;

    m
}

pub fn rotation_y(r: f64) -> Matrix {
    let mut m = Matrix::empty(4, 4).identity();

    let cos_r = r.cos();
    let sin_r = r.sin();

    m[(0, 0)] = cos_r;
    m[(0, 2)] = sin_r;
    m[(2, 0)] = -sin_r;
    m[(2, 2)] = cos_r;

    m
}

pub fn rotation_z(r: f64) -> Matrix {
    let mut m = Matrix::empty(4, 4).identity();

    let cos_r = r.cos();
    let sin_r = r.sin();

    m[(0, 0)] = cos_r;
    m[(0, 1)] = -sin_r;
    m[(1, 0)] = sin_r;
    m[(1, 1)] = cos_r;

    m
}

pub fn shearing(xy: f64, xz: f64, yx: f64, yz: f64, zx: f64, zy: f64) -> Matrix {
    let mut m = Matrix::empty(4, 4).identity();

    m[(0, 1)] = xy;
    m[(0, 2)] = xz;
    m[(1, 0)] = yx;
    m[(1, 2)] = yz;
    m[(2, 0)] = zx;
    m[(2, 1)] = zy;

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

    #[test]
    fn rotating_a_point_around_the_x_axis() {
        let p = Point::new(0., 1., 0.);
        let half_quarter = rotation_x(std::f64::consts::PI / 4.);
        let full_quarter = rotation_x(std::f64::consts::PI / 2.);
        assert_eq!(
            half_quarter * p,
            Point::new(0., 2_f64.sqrt() / 2., 2_f64.sqrt() / 2.)
        );
        assert_eq!(full_quarter * p, Point::new(0., 0., 1.));
    }

    #[test]
    fn the_inverse_of_an_x_rotation_rotates_in_the_opposite_direction() {
        let p = Point::new(0., 1., 0.);
        let half_quarter_inverse = rotation_x(std::f64::consts::PI / 4.).inverse();
        assert_eq!(
            half_quarter_inverse * p,
            Point::new(0., 2_f64.sqrt() / 2., -2_f64.sqrt() / 2.)
        );
    }

    #[test]
    fn rotating_a_point_around_the_y_axis() {
        let p = Point::new(0., 0., 1.);
        let half_quarter = rotation_y(std::f64::consts::PI / 4.);
        let full_quarter = rotation_y(std::f64::consts::PI / 2.);
        assert_eq!(
            half_quarter * p,
            Point::new(2_f64.sqrt() / 2., 0., 2_f64.sqrt() / 2.)
        );
        assert_eq!(full_quarter * p, Point::new(1., 0., 0.));
    }

    #[test]
    fn rotating_a_point_around_the_z_axis() {
        let p = Point::new(0., 1., 0.);
        let half_quarter = rotation_z(std::f64::consts::PI / 4.);
        let full_quarter = rotation_z(std::f64::consts::PI / 2.);
        assert_eq!(
            half_quarter * p,
            Point::new(-2_f64.sqrt() / 2., 2_f64.sqrt() / 2., 0.)
        );
        assert_eq!(full_quarter * p, Point::new(-1., 0., 0.));
    }

    #[test]
    fn a_shearing_transformation_moves_x_in_proportion_to_y() {
        let transform = shearing(1., 0., 0., 0., 0., 0.);
        let p = Point::new(2., 3., 4.);

        assert_eq!(transform * p, Point::new(5., 3., 4.));
    }

    #[test]
    fn a_shearing_transformation_moves_x_in_proportion_to_z() {
        let transform = shearing(0., 1., 0., 0., 0., 0.);
        let p = Point::new(2., 3., 4.);

        assert_eq!(transform * p, Point::new(6., 3., 4.));
    }

    #[test]
    fn a_shearing_transformation_moves_y_in_proportion_to_x() {
        let transform = shearing(0., 0., 1., 0., 0., 0.);
        let p = Point::new(2., 3., 4.);

        assert_eq!(transform * p, Point::new(2., 5., 4.));
    }

    #[test]
    fn a_shearing_transformation_moves_y_in_proportion_to_z() {
        let transform = shearing(0., 0., 0., 1., 0., 0.);
        let p = Point::new(2., 3., 4.);

        assert_eq!(transform * p, Point::new(2., 7., 4.));
    }

    #[test]
    fn a_shearing_transformation_moves_z_in_proportion_to_x() {
        let transform = shearing(0., 0., 0., 0., 1., 0.);
        let p = Point::new(2., 3., 4.);

        assert_eq!(transform * p, Point::new(2., 3., 6.));
    }

    #[test]
    fn a_shearing_transformation_moves_z_in_proportion_to_y() {
        let transform = shearing(0., 0., 0., 0., 0., 1.);
        let p = Point::new(2., 3., 4.);

        assert_eq!(transform * p, Point::new(2., 3., 7.));
    }

    #[test]
    fn individual_transformations_are_applied_in_sequence() {
        let p = Point::new(1., 0., 1.);
        let a = rotation_x(std::f64::consts::PI / 2.);
        let b = scaling(5., 5., 5.);
        let c = translation(10., 5., 7.);

        let p2 = a * p;
        assert_eq!(p2, Point::new(1., -1., 0.));

        let p3 = b * p2;
        assert_eq!(p3, Point::new(5., -5., 0.));

        let p4 = c * p3;
        assert_eq!(p4, Point::new(15., 0., 7.));
    }

    #[test]
    fn chained_transformations_must_be_applied_in_reverse_order() {
        let p = Point::new(1., 0., 1.);
        let a = rotation_x(std::f64::consts::PI / 2.);
        let b = scaling(5., 5., 5.);
        let c = translation(10., 5., 7.);

        let t = c * b * a;
        let p1 = t * p;
        assert_eq!(&p1, &Point::new(15., 0., 7.));

        // FIXME must be in reverse order and not here.
        let t1 = Transformation::new()
            .rotate_x(std::f64::consts::PI / 2.)
            .scale(5., 5., 5.)
            .translate(10., 5., 7.)
            .build();

        let p2 = t1 * p;
        assert_eq!(&p2, &p1);
    }
}
