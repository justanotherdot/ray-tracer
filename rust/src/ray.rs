use crate::coordinate::{Point, Vector};
use crate::naive_cmp::naive_approx_equal_float;
use std::cmp::PartialEq;
use std::ops::Index;
use std::rc::Rc;

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
        self.origin + self.direction * time
    }
}

// TODO: Possibly a `Shape` struct that shapes can be added onto?
// that way we wind up with something a bit nicer for tracking distinct
// ids. A static hashmap is a horrible idea.
// TODO: This needs a custom PartialEq with our naive cmp fn.
// Since we will directly or indirectly compare against f64 again.
// TODO: Remove this clone?
#[derive(PartialEq, Debug, Clone)]
pub struct Sphere {
    id: u64,
}

#[allow(unused_macros)]
macro_rules! intersections {
    ($($e:expr),*) => {{
        {
            let xs = vec![
                $($e),*
            ];
            Intersections(xs)
        }
    }};
}

// TODO This ought to be called Intersction and Intersection should
// be called something else.
#[derive(Debug)]
pub struct Intersection {
    pub t: f64,
    pub object: Rc<Sphere>, // TODO: ought to be Shape.
}

impl PartialEq for Intersection {
    fn eq(&self, other: &Self) -> bool {
        self.object == other.object && naive_approx_equal_float(&self.t, &other.t)
    }
}

impl Sphere {
    pub fn new(id: u64) -> Self {
        Sphere { id }
    }

    pub fn intersect(&self, r: Ray) -> Intersections {
        let sphere_to_ray = r.origin().clone() - Point::new(0., 0., 0.);

        let a = r.direction().clone().dot(&r.direction().clone());
        let b = 2. * r.direction().clone().dot(&sphere_to_ray);
        let c = sphere_to_ray.dot(&sphere_to_ray) - 1.;

        let discriminant = b.powf(2.) - 4. * a * c;

        if discriminant < 0.0 {
            Intersections(vec![])
        } else {
            let t1 = ((-b) - discriminant.sqrt()) / (2. * a);
            let t2 = ((-b) + discriminant.sqrt()) / (2. * a);
            let i1 = Intersection::new(t1, &self);
            let i2 = Intersection::new(t2, &self);
            intersections![i1, i2]
        }
    }
}

impl Intersection {
    // TODO Generalise object to Shape eventually.
    pub fn new(t: f64, object: &Sphere) -> Self {
        let object = Rc::new(object.clone());
        Intersection { t, object }
    }
}

pub struct Intersections(Vec<Intersection>);

impl Intersections {
    pub fn count(&self) -> usize {
        self.0.len()
    }
}

// TODO: The book uses index syntax toget the fst and snd elements
// of the intersections but from what I can tell there is only two
// intersections.
impl Index<usize> for Intersections {
    type Output = Intersection;

    fn index(&self, ix: usize) -> &Self::Output {
        &self.0[ix]
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
        assert_eq!(xs[0].t, 4.0);
        assert_eq!(xs[1].t, 6.0);
    }

    #[test]
    fn a_ray_intersects_a_sphere_at_a_tangent() {
        let r = Ray::new(Point::new(0., 1., -5.), Vector::new(0., 0., 1.));
        // TODO This needs a better way to ensure distinctness.
        let s = Sphere::new(0);
        let xs = s.intersect(r);
        assert_eq!(xs.count(), 2);
        assert_eq!(xs[0].t, 5.0);
        assert_eq!(xs[1].t, 5.0);
    }

    #[test]
    fn a_ray_misses_a_sphere() {
        let r = Ray::new(Point::new(0., 2., -5.), Vector::new(0., 0., 1.));
        // TODO This needs a better way to ensure distinctness.
        let s = Sphere::new(0);
        let xs = s.intersect(r);
        assert_eq!(xs.count(), 0);
    }

    #[test]
    fn a_ray_originates_inside_a_sphere() {
        let r = Ray::new(Point::new(0., 0., 0.), Vector::new(0., 0., 1.));
        // TODO This needs a better way to ensure distinctness.
        let s = Sphere::new(0);
        let xs = s.intersect(r);
        assert_eq!(xs.count(), 2);
        assert_eq!(xs[0].t, -1.0);
        assert_eq!(xs[1].t, 1.0);
    }

    #[test]
    fn a_sphere_is_behind_the_ray() {
        let r = Ray::new(Point::new(0., 0., 5.), Vector::new(0., 0., 1.));
        // TODO This needs a better way to ensure distinctness.
        let s = Sphere::new(0);
        let xs = s.intersect(r);
        assert_eq!(xs.count(), 2);
        assert_eq!(xs[0].t, -6.0);
        assert_eq!(xs[1].t, -4.0);
    }

    #[test]
    fn an_intersectoin_encapsulates_t_and_object() {
        let s = Sphere::new(0);
        let i = Intersection::new(3.5, &s);
        assert_eq!(i.t, 3.5);
        assert_eq!(*i.object, s);
    }

    #[test]
    fn aggregating_intersections() {
        let s = Sphere::new(0);
        let i1 = Intersection::new(1., &s);
        let i2 = Intersection::new(2., &s);
        let xs = intersections![i1, i2];
        assert_eq!(xs.count(), 2);
        assert_eq!(xs[0].t, 1.);
        assert_eq!(xs[1].t, 2.);
    }

    #[test]
    fn intersect_sets_the_object_on_the_intersection() {
        let r = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
        let s = Sphere::new(0);
        let xs = s.intersect(r);
        let rc = Rc::new(s.clone());
        assert_eq!(xs.count(), 2);
        assert_eq!(xs[0].object, rc);
        assert_eq!(xs[1].object, rc);
    }
}
