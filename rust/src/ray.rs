use crate::coordinate::{Point, Vector};

pub struct Ray {
    origin: Point,
    direction: Vector,
}

impl Ray {
    pub fn new(origin: Point, direction: Vector) -> Self {
        Self { origin, direction }
    }

    pub fn origin(&self) -> &Point {
        &self.origin
    }

    pub fn direction(&self) -> &Vector {
        &self.direction
    }

    pub fn position(&self, time: f64) -> Point {
        // TODO It would be worth using direct functions here
        // that take refs instead of the move semantics that
        // operations take.
        let dir2 = self.direction.clone() * time;
        self.origin.clone() + dir2
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::coordinate::*;

    #[test]
    fn creating_and_querying_a_ray() {
        let origin = Point::new(1., 2., 3.);
        let direction = Vector::new(4., 5., 6.);
        let r = Ray::new(origin, direction);
        assert_eq!(r.origin(), &origin);
        assert_eq!(r.direction(), &direction);
    }

    #[test]
    fn computing_a_point_from_a_distance() {
        let r = Ray::new(Point::new(2., 3., 4.), Vector::new(1., 0., 0.));
        assert_eq!(r.position(0.), Point::new(2., 3., 4.));
        assert_eq!(r.position(1.), Point::new(3., 3., 4.));
        assert_eq!(r.position(-1.), Point::new(1., 3., 4.));
        assert_eq!(r.position(2.5), Point::new(4.5, 3., 4.));
    }

    //#[test]
    //fn a_ray_intersects_a_sphere_at_two_points() {
    //let r = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
    //let s = Sphere::new();
    //let xs = s.intersect(r);
    //assert_eq!(xs.count(), 2);
    //assert_eq!(xs.get(0), Some(4.0));
    //assert_eq!(xs.get(1), Some(6.0));
    //}
}
