use crate::canvas::Canvas;
use crate::color::Color;
use crate::coordinate::{Point, Vector};
use crate::matrix;
use crate::matrix::Matrix;
use crate::naive_cmp::F64_EPSILON;
use crate::ray::{Intersection, Intersections, Ray, Sphere};
use crate::shader::{is_shadowed, PointLight};
use crate::transformation::Transformation;
use smallvec::*;
use std::default::Default;
use std::rc::Rc;

#[derive(Debug)]
pub struct World {
    pub objects: SmallVec<[Sphere; 64]>,
    pub light: Option<PointLight>,
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

    pub fn intersect_world(&self, r: &Ray) -> Intersections {
        intersect_world(self, r)
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

pub fn intersect_world(w: &World, r: &Ray) -> Intersections {
    let mut sv = SmallVec::new();
    w.objects
        .iter()
        .filter_map(|o| o.intersect(&r))
        .for_each(|(i1, i2)| {
            sv.push(i1);
            sv.push(i2);
        });
    Intersections::from_smallvec(sv)
}

pub fn color_at(w: &World, r: &Ray) -> Color {
    let is = w.intersect(r);
    match is.hit() {
        Some(hit) => shade_hit(w, &prepare_computations(&hit, &r)),
        None => Color::new(0., 0., 0.),
    }
}

#[derive(Debug)]
pub struct PreComp {
    pub t: f64,
    pub object: Rc<Sphere>, // TODO: Should be Shape.
    pub point: Point,
    pub eyev: Vector,
    pub normalv: Vector,
    pub inside: bool,
    pub over_point: Point,
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
    let over_point = point + normalv * F64_EPSILON;
    PreComp {
        t,
        object,
        point,
        eyev,
        normalv,
        inside,
        over_point,
    }
}

// TODO turn this into Result.
pub fn shade_hit(w: &World, c: &PreComp) -> Color {
    // N.B. If we wanted to, we could support multiple light
    // sources by summing each `lighting` result.
    let shadowed = is_shadowed(w, &c.over_point);
    let light = w.light.as_ref().unwrap();
    c.object
        .material
        .lighting(&light, &c.over_point, &c.eyev, &c.normalv, shadowed)
}

pub fn view_transform(from: Point, to: Point, up: Vector) -> Matrix {
    let forward = (to - from).normalize();
    let upn = up.normalize();
    let left = forward.cross(&upn);
    let true_up = left.cross(&forward);
    #[rustfmt::skip]
    let orientation: Matrix = Matrix::from_nested_vec(vec![
        vec![left.x,       left.y,     left.z,    0.],
        vec![true_up.x,    true_up.y,  true_up.z, 0.],
        vec![-forward.x,  -forward.y, -forward.z, 0.],
        vec![0.,           0.,         0.,        1.],
    ]);

    let translation: Matrix = Transformation::new()
        .translate(-from.x, -from.y, -from.z)
        .build();

    matrix::matrix_mul(&orientation, &translation)
}

pub struct Camera {
    pub hsize: usize,
    pub vsize: usize,
    pub field_of_view: f64,
    pub transform: Matrix,
    pub pixel_size: f64,
    pub half_height: f64,
    pub half_width: f64,
}

impl Camera {
    pub fn new(hsize: usize, vsize: usize, field_of_view: f64) -> Self {
        let transform = Transformation::new().build();

        // TODO: The abs delta of tan(pi/4) has rounding errors on
        // f64, so we go to f32 to flub it to a lower precision.
        let half_view = (field_of_view as f32 / 2.0 as f32).tan() as f64;

        let aspect = hsize as f64 / vsize as f64;

        let (half_width, half_height) = if aspect >= 1. {
            (half_view, half_view / aspect)
        } else {
            (half_view * aspect, half_view)
        };
        let pixel_size = (half_width * 2.) / hsize as f64;

        Camera {
            hsize,
            vsize,
            field_of_view,
            transform,
            pixel_size,
            half_width,
            half_height,
        }
    }

    pub fn ray_for_pixel(&self, px: usize, py: usize) -> Ray {
        let xoffset = (px as f64 + 0.5) * self.pixel_size;
        let yoffset = (py as f64 + 0.5) * self.pixel_size;

        let world_x = self.half_width - xoffset;
        let world_y = self.half_height - yoffset;

        let pixel = self.transform.inverse() * Point::new(world_x, world_y, -1.);
        let origin = self.transform.inverse() * Point::new(0., 0., 0.);
        let direction = (pixel - origin).normalize();

        Ray::new(origin, direction)
    }

    pub fn render(&self, world: &World) -> Canvas {
        let mut image = Canvas::new(self.hsize, self.vsize);
        for y in 0..self.vsize - 1 {
            for x in 0..self.hsize - 1 {
                let ray = self.ray_for_pixel(x, y);
                let color = world.color_at(&ray);
                image.write_pixel(x, y, color);
            }
        }

        image
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::coordinate::Vector;
    use crate::matrix::{IdentityMatrix, Matrix};
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
        let mut w: World = Default::default();

        {
            let outer = &mut w.objects[0];
            outer.material.ambient = 1.;
        }
        {
            let inner = &mut w.objects[1];
            inner.material.ambient = 1.;
        }

        let r = Ray::new(Point::new(0., 0., 0.75), Vector::new(0., 0., -1.));
        let c = w.color_at(&r);

        let inner = &w.objects[1];

        assert_eq!(&c, &inner.material.color);
    }

    #[test]
    fn the_transformation_matrix_for_the_default_orientation() {
        let from = Point::new(0., 0., 0.);
        let to = Point::new(0., 0., -1.);
        let up = Vector::new(0., 1., 0.);
        let t = view_transform(from, to, up);
        let m = Matrix::empty(4, 4).identity();
        assert_eq!(t, m);
    }

    #[test]
    fn a_view_transformation_matrix_looking_in_positive_z_direction() {
        let from = Point::new(0., 0., 0.);
        let to = Point::new(0., 0., 1.);
        let up = Vector::new(0., 1., 0.);
        let t = view_transform(from, to, up);
        let m = Transformation::new().scale(-1., 1., -1.).build();
        assert_eq!(t, m);
    }

    #[test]
    fn the_view_transformation_moves_the_world() {
        let from = Point::new(0., 0., 8.);
        let to = Point::new(0., 0., 0.);
        let up = Vector::new(0., 1., 0.);
        let t = view_transform(from, to, up);
        let m = Transformation::new().translate(0., 0., -8.).build();
        assert_eq!(t, m);
    }

    #[test]
    fn an_arbitrary_view_transformation() {
        let from = Point::new(1., 3., 2.);
        let to = Point::new(4., -2., 8.);
        let up = Vector::new(1., 1., 0.);
        let t = view_transform(from, to, up);
        #[rustfmt::skip]
        let m = Matrix::from_nested_vec(vec![
            vec![-0.50709, 0.50709,  0.67612, -2.36643],
            vec![ 0.76772, 0.60609,  0.12122, -2.82843],
            vec![-0.35857, 0.59761, -0.71714,  0.00000],
            vec![ 0.00000, 0.00000,  0.00000,  1.00000],
        ]);
        assert_eq!(t, m);
    }

    #[test]
    fn constructing_a_camera() {
        let hsize = 160;
        let vsize = 120;
        let field_of_view = std::f64::consts::PI / 2.0;
        let c = Camera::new(hsize, vsize, field_of_view);
        assert_eq!(c.hsize, 160);
        assert_eq!(c.vsize, 120);
        assert_eq!(c.field_of_view, std::f64::consts::PI / 2.0);
        assert_eq!(c.transform, Matrix::empty(4, 4).identity());
    }

    #[test]
    fn the_pixel_size_for_a_horizontal_canvas() {
        let c = Camera::new(200, 125, std::f64::consts::PI / 2.0);
        assert_eq!(&c.pixel_size, &0.01);
    }

    #[test]
    fn the_pixel_size_for_a_vertical_canvas() {
        let c = Camera::new(125, 200, std::f64::consts::PI / 2.0);
        assert_eq!(&c.pixel_size, &0.01);
    }

    #[test]
    fn constructing_a_ray_through_the_center_of_the_canvas() {
        let c = Camera::new(201, 101, std::f64::consts::PI / 2.0);
        let r = c.ray_for_pixel(100, 50);
        assert_eq!(r.origin, Point::new(0., 0., 0.));
        assert_eq!(r.direction, Vector::new(0., 0., -1.));
    }

    #[test]
    fn constructing_a_ray_through_the_corner_of_the_canvas() {
        let c = Camera::new(201, 101, std::f64::consts::PI / 2.0);
        let r = c.ray_for_pixel(0, 0);
        assert_eq!(r.origin, Point::new(0., 0., 0.));
        assert_eq!(r.direction, Vector::new(0.66519, 0.33259, -0.66851));
    }

    #[test]
    fn constructing_a_ray_when_the_camera_is_transformed() {
        let mut c = Camera::new(201, 101, std::f64::consts::PI / 2.0);
        c.transform = Transformation::new()
            .translate(0., -2., 5.)
            .rotate_y(std::f64::consts::PI / 4.)
            .build();
        let r = c.ray_for_pixel(100, 50);
        assert_eq!(r.origin, Point::new(0., 2., -5.));
        assert_eq!(
            r.direction,
            Vector::new((2.0 as f64).sqrt() / 2., 0., -((2.0 as f64).sqrt() / 2.0))
        );
    }

    #[test]
    fn rendering_a_world_with_a_camera() {
        let w: World = Default::default();
        let mut c = Camera::new(11, 11, std::f64::consts::PI / 2.0);
        let from = Point::new(0., 0., -5.);
        let to = Point::new(0., 0., 0.);
        let up = Vector::new(0., 1., 0.);
        c.transform = view_transform(from, to, up);
        let image = c.render(&w);
        assert_eq!(image.pixel_at(5, 5), &Color::new(0.38066, 0.47583, 0.2855));
    }

    #[test]
    fn shade_hit_is_given_an_intersection_in_shadow() {
        let mut w: World = World::new();
        w.light = Some(PointLight::new(
            Point::new(0., 0., -10.),
            Color::new(1., 1., 1.),
        ));
        let s1 = Sphere::new(0);
        let mut s2 = Sphere::new(1);
        s2.set_transform(Transformation::new().translate(0., 0., 10.).build());
        w.objects = smallvec![s1, s2.clone()];
        let r = Ray::new(Point::new(0., 0., 5.), Vector::new(0., 0., 1.));
        let i = Intersection::new(4., &s2);
        let comps = prepare_computations(&i, &r);
        let c = w.shade_hit(&comps);
        assert_eq!(c, Color::new(0.1, 0.1, 0.1));
    }
}
