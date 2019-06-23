use crate::color::Color;
use crate::coordinate::{Point, Vector};
use crate::ray::{Intersection, Intersections, Ray, Sphere};
use crate::shader::PointLight;
use crate::transformation::Transformation;
use smallvec::*;
use std::default::Default;
use std::rc::Rc;

pub struct World {
    objects: SmallVec<[Sphere; 64]>,
    light: Option<PointLight>,
}

impl World {
    pub fn new() -> World {
        let objects = smallvec![];
        let light = None;
        World { objects, light }
    }
}

impl Default for World {
    fn default() -> Self {
        let mut w = World::new();

        let light = PointLight::new(Point::new(-10., 10., -10.), Color::new(1., 1., 1.));
        let mut s1 = Sphere::new(0);
        s1.material.color = Color::new(0.8, 1.0, 0.6);
        s1.material.diffuse = 0.7;
        s1.material.specular = 0.2;

        let mut s2 = Sphere::new(1);
        s2.set_transform(Transformation::new().scale(0.5, 0.5, 0.5).build());

        w.objects = smallvec![s1, s2];
        w.light = Some(light);

        w
    }
}

pub fn intersect_world(w: World, r: Ray) -> Intersections {
    let sv = smallvec![];
    let mut is = w.objects.iter().fold(sv, |mut acc, o| {
        o.intersect(&r).into_iter().for_each(|intersection| {
            acc.push(intersection.clone());
        });
        acc
    });

    is.sort();

    Intersections::from_smallvec(is)
}

pub struct PreComp {
    pub t: f64,
    pub object: Rc<Sphere>, // TODO: Should be Shape.
    pub point: Point,
    pub eyev: Vector,
    pub normalv: Vector,
    pub inside: bool,
}

pub fn prepare_computations(intersection: Intersection, ray: Ray) -> PreComp {
    let t = intersection.t;
    let object = intersection.object;
    let point = ray.position(t);
    let eyev = -ray.direction;
    let mut normalv = object.normal_at(point);

    let inside = if normalv.dot(&eyev) < 0.0 {
        normalv = -normalv;
        true
    } else {
        false
    };

    PreComp {
        t,
        object,
        point,
        eyev,
        normalv,
        inside,
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::coordinate::Vector;
    use crate::ray::Ray;

    #[test]
    fn creating_a_world() {
        let w = World::new();
        assert!(w.objects.is_empty());
        assert_eq!(w.light, None);
    }

    #[test]
    fn the_default_world() {
        let light = PointLight::new(Point::new(-10., 10., -10.), Color::new(1., 1., 1.));

        let mut s1 = Sphere::new(0);
        s1.material.color = Color::new(0.8, 1.0, 0.6);
        s1.material.diffuse = 0.7;
        s1.material.specular = 0.2;

        let mut s2 = Sphere::new(1);
        s2.set_transform(Transformation::new().scale(0.5, 0.5, 0.5).build());

        let w: World = Default::default();

        assert_eq!(w.light, Some(light));
        assert!(w.objects.iter().any(|x| *x == s1));
        assert!(w.objects.iter().any(|x| *x == s2));
    }

    #[test]
    fn intersect_a_world_with_a_ray() {
        let w: World = Default::default();

        let r = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
        let xs = intersect_world(w, r);

        assert_eq!(xs.count(), 4);
        assert_eq!(xs.get(0).unwrap().t, 4.0);
        assert_eq!(xs.get(1).unwrap().t, 4.5);
        assert_eq!(xs.get(2).unwrap().t, 5.5);
        assert_eq!(xs.get(3).unwrap().t, 6.0);
    }

    #[test]
    fn precomputing_the_state_of_an_intersection() {
        let r = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
        let shape = Sphere::new(0);
        let i = Intersection::new(4., &shape);
        let comps = prepare_computations(i.clone(), r);

        assert_eq!(comps.t, i.t);
        assert_eq!(comps.object, i.object);
        assert_eq!(comps.point, Point::new(0., 0., -1.));
        assert_eq!(comps.eyev, Vector::new(0., 0., -1.));
        assert_eq!(comps.normalv, Vector::new(0., 0., -1.));
    }

    #[test]
    fn the_hit_when_an_intersection_occurs_on_the_outside() {
        let r = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
        let shape = Sphere::new(0);
        let i = Intersection::new(4., &shape);
        let comps = prepare_computations(i, r);

        assert_eq!(comps.inside, false);
    }

    #[test]
    fn the_hit_when_an_intersection_occurs_on_the_inside() {
        let r = Ray::new(Point::new(0., 0., 0.), Vector::new(0., 0., 1.));
        let shape = Sphere::new(0);
        let i = Intersection::new(1., &shape);
        let comps = prepare_computations(i, r);

        assert_eq!(comps.point, Point::new(0., 0., 1.));
        assert_eq!(comps.eyev, Vector::new(0., 0., -1.));
        assert_eq!(comps.inside, true);
        assert_eq!(comps.normalv, Vector::new(0., 0., -1.));
    }
}
