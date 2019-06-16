
#[cfg(test)]
mod test {
    use crate::ray::Sphere;
    use crate::coodinate::{Point, Sphere};
    use super::*;

    #[test]
    fn the_normal_on_a_sphere_at_a_point_on_the_x_axis() {
        let s = Sphere::new(0);
        let n = normal_at(s, Point::new(1., 0. 0.));
        assert_eq!(n, Vector::new(1., 0., 0.));
    }
}
