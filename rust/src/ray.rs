use crate::coordinate::{Point, Vector};
use crate::matrix::{IdentityMatrix, Matrix, SquareMatrix};
use crate::naive_cmp;
use crate::shader::Material;
use smallvec::*;
use std::cmp::{Eq, Ord, Ordering, PartialEq};
use std::default::Default;
use std::rc::Rc;

#[derive(Clone)]
pub struct Ray {
    pub origin: Point,
    pub direction: Vector,
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

    pub fn transform(&self, m: Matrix) -> Self {
        Self {
            origin: m.clone() * self.origin,
            direction: m.clone() * self.direction,
        }
    }
}

// TODO: This needs some way ensuring that we don't have overlapping
// ids on shapes. i.e. We should never have the same shape after `new`.
// TODO: Possibly a `Shape` struct that shapes can be added onto?
// that way we wind up with something a bit nicer for tracking distinct
// ids. A static hashmap is a horrible idea.
// TODO: This needs a custom PartialEq with our naive cmp fn.
// Since we will directly or indirectly compare against f64 again.
// TODO: Remove this clone?
#[derive(PartialEq, Debug, Clone)]
pub struct Sphere {
    pub id: u64,
    pub transform: Matrix,
    pub material: Material,
}

#[allow(unused_macros)]
macro_rules! intersections {
    ($($e:expr),*) => {
        {
            let mut xs = smallvec![$($e),*];
            xs.sort();
            Intersections(xs)
        }
    };
}

// TODO This ought to be called Intersction and Intersection should
// be called something else.
#[derive(Debug, Clone)]
pub struct Intersection {
    pub t: f64,
    pub object: Rc<Sphere>, // TODO: ought to be Shape.
}

impl PartialEq for Intersection {
    fn eq(&self, other: &Self) -> bool {
        self.object == other.object && naive_cmp::naive_approx_equal_float(&self.t, &other.t)
    }
}

impl Eq for Intersection {}

impl PartialOrd for Intersection {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(naive_cmp::naive_approx_float_cmp(&self.t, &other.t))
    }
}

impl Ord for Intersection {
    fn cmp(&self, other: &Self) -> Ordering {
        naive_cmp::naive_approx_float_cmp(&self.t, &other.t)
    }
}

impl Sphere {
    pub fn new(id: u64) -> Self {
        let transform = Matrix::empty(4, 4).identity();
        let material = Default::default();
        Sphere {
            id,
            transform,
            material,
        }
    }

    pub fn intersect(&self, r: Ray) -> Intersections {
        let r = r.transform(self.transform.inverse());
        let sphere_to_ray = r.origin().clone() - Point::new(0., 0., 0.);

        let a = r.direction().clone().dot(&r.direction().clone());
        let b = 2. * r.direction().clone().dot(&sphere_to_ray);
        let c = sphere_to_ray.dot(&sphere_to_ray) - 1.;

        let discriminant = b.powf(2.) - 4. * a * c;

        if discriminant < 0.0 {
            intersections![]
        } else {
            let t1 = ((-b) - discriminant.sqrt()) / (2. * a);
            let t2 = ((-b) + discriminant.sqrt()) / (2. * a);
            let i1 = Intersection::new(t1, &self);
            let i2 = Intersection::new(t2, &self);
            intersections![i1, i2]
        }
    }

    pub fn set_transform(&mut self, m: Matrix) {
        self.transform = m;
    }
}

impl Intersection {
    // TODO Generalise object to Shape eventually.
    pub fn new(t: f64, object: &Sphere) -> Self {
        let object = Rc::new(object.clone());
        Intersection { t, object }
    }
}

pub struct Intersections(SmallVec<[Intersection; 2]>);

impl Intersections {
    pub fn count(&self) -> usize {
        self.0.len()
    }

    pub fn hit(&self) -> Option<&Intersection> {
        self.0.iter().filter(|i| i.t >= 0.).nth(0)
    }

    pub fn get(&self, ix: usize) -> Option<&Intersection> {
        self.0.iter().nth(ix)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::coordinate::*;
    use crate::transformation::Transformation;

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
        assert_eq!(xs.get(0).unwrap().t, 4.0);
        assert_eq!(xs.get(1).unwrap().t, 6.0);
    }

    #[test]
    fn a_ray_intersects_a_sphere_at_a_tangent() {
        let r = Ray::new(Point::new(0., 1., -5.), Vector::new(0., 0., 1.));
        // TODO This needs a better way to ensure distinctness.
        let s = Sphere::new(0);
        let xs = s.intersect(r);
        assert_eq!(xs.count(), 2);
        assert_eq!(xs.get(0).unwrap().t, 5.0);
        assert_eq!(xs.get(1).unwrap().t, 5.0);
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
        assert_eq!(xs.get(0).unwrap().t, -1.0);
        assert_eq!(xs.get(1).unwrap().t, 1.0);
    }

    #[test]
    fn a_sphere_is_behind_the_ray() {
        let r = Ray::new(Point::new(0., 0., 5.), Vector::new(0., 0., 1.));
        // TODO This needs a better way to ensure distinctness.
        let s = Sphere::new(0);
        let xs = s.intersect(r);
        assert_eq!(xs.count(), 2);
        assert_eq!(xs.get(0).unwrap().t, -6.0);
        assert_eq!(xs.get(1).unwrap().t, -4.0);
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
        assert_eq!(xs.get(0).unwrap().t, 1.);
        assert_eq!(xs.get(1).unwrap().t, 2.);
    }

    #[test]
    fn intersect_sets_the_object_on_the_intersection() {
        let r = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
        let s = Sphere::new(0);
        let xs = s.intersect(r);
        let rc = Rc::new(s.clone());
        assert_eq!(xs.count(), 2);
        assert_eq!(xs.get(0).unwrap().object, rc);
        assert_eq!(xs.get(1).unwrap().object, rc);
    }

    #[test]
    fn the_hit_when_all_intersections_have_positive_t() {
        let s = Sphere::new(0);
        let i1 = Intersection::new(1., &s);
        let i2 = Intersection::new(2., &s);
        let xs = intersections![i1.clone(), i2.clone()];
        let i = xs.hit();
        assert_eq!(i, Some(&i1));
    }

    #[test]
    fn the_hit_when_some_intersections_have_negative_t() {
        let s = Sphere::new(0);
        let i1 = Intersection::new(-1., &s);
        let i2 = Intersection::new(1., &s);
        let xs = intersections![i2.clone(), i1.clone()];
        let i = xs.hit();
        assert_eq!(i, Some(&i2));
    }

    #[test]
    fn the_hit_when_all_intersections_have_negative_t() {
        let s = Sphere::new(0);
        let i1 = Intersection::new(-2., &s);
        let i2 = Intersection::new(-1., &s);
        let xs = intersections![i2.clone(), i1.clone()];
        let i = xs.hit();
        assert_eq!(i, None);
    }

    #[test]
    fn the_hit_is_always_the_lowest_nonnegative_insersection() {
        let s = Sphere::new(0);
        let i1 = Intersection::new(5., &s);
        let i2 = Intersection::new(7., &s);
        let i3 = Intersection::new(-3., &s);
        let i4 = Intersection::new(2., &s);
        let xs = intersections![i1.clone(), i2.clone(), i3.clone(), i4.clone()];
        let i = xs.hit();
        assert_eq!(i, Some(&i4));
    }

    #[test]
    fn translating_a_ray() {
        let r = Ray::new(Point::new(1., 2., 3.), Vector::new(0., 1., 0.));
        let m = Transformation::new().translate(3., 4., 5.).build();
        let r2 = r.transform(m);
        assert_eq!(r2.origin, Point::new(4., 6., 8.));
        assert_eq!(r2.direction, Vector::new(0., 1., 0.));
    }

    #[test]
    fn scaling_a_ray() {
        let r = Ray::new(Point::new(1., 2., 3.), Vector::new(0., 1., 0.));
        let m = Transformation::new().scale(2., 3., 4.).build();
        let r2 = r.transform(m);
        assert_eq!(r2.origin, Point::new(2., 6., 12.));
        assert_eq!(r2.direction, Vector::new(0., 3., 0.));
    }

    #[test]
    fn a_spheres_default_transformation() {
        let s = Sphere::new(0);
        let identity = Matrix::empty(4, 4).identity();
        assert_eq!(s.transform, identity);
    }

    #[test]
    fn changing_a_spheres_transformation() {
        let mut s = Sphere::new(0);
        let t = Transformation::new().translate(2., 3., 4.).build();
        s.set_transform(t.clone());
        assert_eq!(s.transform, t);
    }

    #[test]
    fn intersecting_a_scaled_sphere_with_a_ray() {
        let r = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
        let mut s = Sphere::new(0);
        let t = Transformation::new().scale(2., 2., 2.).build();
        s.set_transform(t.clone());
        let xs = s.intersect(r);
        assert_eq!(xs.count(), 2);
        assert_eq!(xs.get(0).unwrap().t, 3.);
        assert_eq!(xs.get(1).unwrap().t, 7.);
    }

    #[test]
    fn intersecting_a_translated_sphere_with_a_ray() {
        let r = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
        let mut s = Sphere::new(0);
        let t = Transformation::new().translate(5., 0., 0.).build();
        s.set_transform(t.clone());
        let xs = s.intersect(r);
        assert_eq!(xs.count(), 0);
    }
}
