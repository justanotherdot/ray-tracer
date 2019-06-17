use crate::coordinate::{Point, Vector};
use crate::matrix::SquareMatrix;
use crate::ray::Sphere;

pub fn normal_at(s: Sphere, world_point: Point) -> Vector {
    let subm = s.transform.submatrix(3, 3);
    let object_point = s.transform.inverse() * world_point;
    let object_normal = object_point - Point::new(0., 0., 0.);
    let world_normal = subm.inverse().transpose() * object_normal;
    world_normal.normalize()
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::coordinate::{Point, Vector};
    use crate::ray::Sphere;
    use crate::transformation::Transformation;

    #[test]
    fn the_normal_on_a_sphere_at_a_point_on_the_x_axis() {
        let s = Sphere::new(0);
        let n = normal_at(s, Point::new(1., 0., 0.));
        assert_eq!(n, Vector::new(1., 0., 0.));
    }

    #[test]
    fn the_normal_on_a_sphere_at_a_point_on_the_y_axis() {
        let s = Sphere::new(0);
        let n = normal_at(s, Point::new(0., 1., 0.));
        assert_eq!(n, Vector::new(0., 1., 0.));
    }

    #[test]
    fn the_normal_on_a_sphere_at_a_point_on_the_z_axis() {
        let s = Sphere::new(0);
        let n = normal_at(s, Point::new(0., 0., 1.));
        assert_eq!(n, Vector::new(0., 0., 1.));
    }

    #[test]
    fn the_normal_on_a_sphere_at_a_nonaxial_point() {
        let s = Sphere::new(0);
        let n = normal_at(
            s,
            Point::new(
                (3.0 as f64).sqrt() / 3.,
                (3.0 as f64).sqrt() / 3.,
                (3.0 as f64).sqrt() / 3.,
            ),
        );
        let v = Vector::new(
            (3.0 as f64).sqrt() / 3.,
            (3.0 as f64).sqrt() / 3.,
            (3.0 as f64).sqrt() / 3.,
        );
        assert_eq!(n, v);
    }

    #[test]
    fn the_normal_is_a_normalized_vector() {
        let s = Sphere::new(0);
        let n = normal_at(
            s,
            Point::new(
                (3.0 as f64).sqrt() / 3.,
                (3.0 as f64).sqrt() / 3.,
                (3.0 as f64).sqrt() / 3.,
            ),
        );
        let v = Vector::new(
            (3.0 as f64).sqrt() / 3.,
            (3.0 as f64).sqrt() / 3.,
            (3.0 as f64).sqrt() / 3.,
        );
        assert_eq!(n, v.normalize());
    }

    #[test]
    fn computing_the_normal_on_a_translated_sphere() {
        let mut s = Sphere::new(0);
        s.set_transform(Transformation::new().translate(0., 1., 0.).build());
        let n = normal_at(s, Point::new(0., 1.70711, -0.70711));
        let v = Vector::new(0., 0.70711, -0.70711);
        assert_eq!(n, v);
    }

    #[test]
    fn computing_the_normal_on_a_transformed_sphere() {
        let mut s = Sphere::new(0);
        s.set_transform(
            Transformation::new()
                .rotate_z(std::f64::consts::PI / 5.)
                .scale(1., 0.5, 1.)
                .build(),
        );
        let n = normal_at(
            s,
            Point::new(0., (2. as f64).sqrt() / 2., -((2. as f64).sqrt() / 2.)),
        );
        let v = Vector::new(0., 0.97014, -0.24254);
        assert_eq!(n, v);
    }
}
