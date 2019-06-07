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

// TODO: Possibly a `Shape` struct that shapes can be added onto?
// that way we wind up with something a bit nicer for tracking distinct
// ids. A static hashmap is a horrible idea.
// TODO: This needs a custom PartialEq with our naive cmp fn.
// Since we will directly or indirectly compare against f64 again.
#[derive(PartialEq)]
pub struct Sphere {
    id: u64,
}

#[derive(PartialEq)]
pub struct Intersection {
    // TODO: Should this just be a two-tuple?
    //       Is the points of intersection always 0-2?
    points: Vec<f64>,
}

impl Sphere {
    pub fn new(id: u64) -> Self {
        Sphere { id }
    }

    pub fn intersect(self, r: Ray) -> Intersection {
        let sphere_to_ray = r.origin().clone() - Point::new(0., 0., 0.);

        let a = r.direction().clone().dot(&r.direction().clone());
        let b = 2. * r.direction().clone().dot(&sphere_to_ray);
        let c = sphere_to_ray.dot(&sphere_to_ray) - 1.;

        let discriminant = b.powf(2.) - 4. * a * c;

        let points = vec![];
        if discriminant < 0.0 {
            Intersection { points }
        } else {
            let t1 = ((-b) - discriminant.sqrt()) / (2. * a);
            let t2 = ((-b) + discriminant.sqrt()) / (2. * a);
            let points = vec![t1, t2];
            Intersection { points }
        }
    }
}

impl Intersection {
    pub fn count(&self) -> usize {
        self.points.len()
    }

    pub fn get(&self, ix: usize) -> Option<f64> {
        self.points.get(ix).map(|v| *v)
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

    #[test]
    fn a_ray_intersects_a_sphere_at_two_points() {
        let r = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
        // TODO This needs a better way to ensure distinctness.
        let s = Sphere::new(0);
        let xs = s.intersect(r);
        assert_eq!(xs.count(), 2);
        assert_eq!(xs.get(0), Some(4.0));
        assert_eq!(xs.get(1), Some(6.0));
    }

    #[test]
    fn a_ray_intersects_a_sphere_at_a_tangent() {
        let r = Ray::new(Point::new(0., 1., -5.), Vector::new(0., 0., 1.));
        // TODO This needs a better way to ensure distinctness.
        let s = Sphere::new(0);
        let xs = s.intersect(r);
        assert_eq!(xs.count(), 2);
        assert_eq!(xs.get(0), Some(5.0));
        assert_eq!(xs.get(1), Some(5.0));
    }

    #[test]
    fn a_ray_misses_a_sphere() {
        let r = Ray::new(Point::new(0., 2., -5.), Vector::new(0., 0., 1.));
        // TODO This needs a better way to ensure distinctness.
        let s = Sphere::new(0);
        let xs = s.intersect(r);
        assert_eq!(xs.count(), 0);
        assert_eq!(xs.get(0), None);
    }

    #[test]
    fn a_ray_originates_inside_a_sphere() {
        let r = Ray::new(Point::new(0., 0., 0.), Vector::new(0., 0., 1.));
        // TODO This needs a better way to ensure distinctness.
        let s = Sphere::new(0);
        let xs = s.intersect(r);
        assert_eq!(xs.count(), 2);
        assert_eq!(xs.get(0), Some(-1.0));
        assert_eq!(xs.get(1), Some(1.0));
    }

    #[test]
    fn a_sphere_is_behind_the_ray() {
        let r = Ray::new(Point::new(0., 0., 5.), Vector::new(0., 0., 1.));
        // TODO This needs a better way to ensure distinctness.
        let s = Sphere::new(0);
        let xs = s.intersect(r);
        assert_eq!(xs.count(), 2);
        assert_eq!(xs.get(0), Some(-6.0));
        assert_eq!(xs.get(1), Some(-4.0));
    }
}
