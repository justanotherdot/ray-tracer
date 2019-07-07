use ray_tracer::color::Color;
use ray_tracer::coordinate::{Point, Vector};
use ray_tracer::ppm::Ppm;
use ray_tracer::ray::Sphere;
use ray_tracer::shader::PointLight;
use ray_tracer::world;
use ray_tracer::world::{Camera, World};
use std::f64::consts::PI;
use std::fs::File;
use std::io::prelude::*;

fn trace() -> Ppm {
    // TODO: Try with default world, what does it look like when rendered?
    //let w: World = Default::default();
    let mut w = World::new();
    let mut s = Sphere::new(0);
    s.material.color = Color::new(1., 0.2, 1.);
    w.objects.push(s);

    let light_position = Point::new(-10., 10., -10.);
    let light_color = Color::new(1., 1., 1.);
    let light = PointLight::new(light_position, light_color);

    w.light = Some(light);

    let mut c = Camera::new(200, 200, PI / 2.0);
    let from = Point::new(0., 0., -5.);
    let to = Point::new(0., 0., 0.);
    let up = Vector::new(0., 1., 0.);
    c.transform = world::view_transform(from, to, up);

    let canvas = c.render(w);

    canvas.to_ppm()
}

pub fn main() -> std::io::Result<()> {
    let ppm = trace();
    let mut file = File::create("sphere_compact.ppm")?;
    file.write_all(ppm.blob().as_bytes())?;
    Ok(())
}
