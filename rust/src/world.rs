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

    pub fn intersect(&self, r: &Ray) -> Intersections {
        intersect_world(self, r)
    }

    pub fn shade_hit(&self, c: &PreComp) -> Color {
        shade_hit(self, c)
    }

    pub fn color_at(&self, r: &Ray) -> Color {
        color_at(self, r)
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

// TODO: Make this a method off World.
pub fn intersect_world(w: &World, r: &Ray) -> Intersections {
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

pub fn color_at(w: &World, r: &Ray) -> Color {
    let is = w.intersect(r);
    match is.hit() {
        Some(hit) => shade_hit(w, &prepare_computations(&hit, &r)),
        None => Color::new(0., 0., 0.),
    }
}

pub struct PreComp {
    pub t: f64,
    pub object: Rc<Sphere>, // TODO: Should be Shape.
    pub point: Point,
    pub eyev: Vector,
    pub normalv: Vector,
    pub inside: bool,
}

pub fn prepare_computations(intersection: &Intersection, ray: &Ray) -> PreComp {
    let t = intersection.t;
    let object = intersection.object.clone();
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

pub fn shade_hit(w: &World, c: &PreComp) -> Color {
    // N.B. If we wanted to, we could support multiple light
    // sources by summing each `lighting` result.
    match w.light {
        Some(ref light) => c
            .object
            .material
            .lighting(light, &c.point, &c.eyev, &c.normalv),
        None => panic!("error: shade_hit called but no light source found"),
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::coordinate::Vector;
    use crate::ray::Ray;
    use crate::shader::PointLight;

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
        let xs = intersect_world(&w, &r);

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
        let comps = prepare_computations(&i, &r);

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
        let comps = prepare_computations(&i, &r);

        assert_eq!(comps.inside, false);
    }

    #[test]
    fn the_hit_when_an_intersection_occurs_on_the_inside() {
        let r = Ray::new(Point::new(0., 0., 0.), Vector::new(0., 0., 1.));
        let shape = Sphere::new(0);
        let i = Intersection::new(1., &shape);
        let comps = prepare_computations(&i, &r);

        assert_eq!(comps.point, Point::new(0., 0., 1.));
        assert_eq!(comps.eyev, Vector::new(0., 0., -1.));
        assert_eq!(comps.inside, true);
        assert_eq!(comps.normalv, Vector::new(0., 0., -1.));
    }

    #[test]
    fn shading_an_intersection() {
        let w: World = Default::default();
        let r = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
        let shape = w.objects.get(0).unwrap();
        let i = Intersection::new(4., shape);
        let comps = prepare_computations(&i, &r);
        let c = shade_hit(&w, &comps);

        assert_eq!(c, Color::new(0.38066, 0.47583, 0.2855));
    }

    #[test]
    fn shading_an_intersection_from_the_inside() {
        let mut w: World = Default::default();
        w.light = Some(PointLight::new(
            Point::new(0., 0.25, 0.),
            Color::new(1., 1., 1.),
        ));
        let r = Ray::new(Point::new(0., 0., 0.), Vector::new(0., 0., 1.));
        let shape = w.objects.get(1).unwrap();
        let i = Intersection::new(0.5, shape);
        let comps = prepare_computations(&i, &r);
        let c = shade_hit(&w, &comps);

        assert_eq!(c, Color::new(0.90498, 0.90498, 0.90498));
    }

    #[test]
    fn the_color_when_a_ray_misses() {
        let w: World = Default::default();
        let r = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 1., 0.));
        let c = w.color_at(&r);

        assert_eq!(c, Color::new(0., 0., 0.));
    }

    #[test]
    fn the_color_when_a_ray_hits() {
        let w: World = Default::default();
        let r = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
        let c = w.color_at(&r);

        assert_eq!(c, Color::new(0.38066, 0.47583, 0.2855));
    }

    #[test]
    fn the_color_with_an_intersection_behind_the_ray() {
        let w: World = Default::default();
        let mut outer = w.objects.get(0).unwrap().clone();
        outer.material.ambient = 1.;
        let inner = w.objects.get(1).unwrap().clone();
        outer.material.ambient = 1.;
        let r = Ray::new(Point::new(0., 0., 0.75), Vector::new(0., 0., -1.));
        let c = w.color_at(&r);

        assert_eq!(&c, &inner.material.color);
    }
}
