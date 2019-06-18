use ray_tracer::canvas::Canvas;
use ray_tracer::color::Color;
use ray_tracer::coordinate::Point;
use ray_tracer::ppm::Ppm;
use ray_tracer::ray::{Ray, Sphere};
use ray_tracer::shader;
use ray_tracer::shader::PointLight;
use std::borrow::Borrow;
use std::fs::File;
use std::io::prelude::*;

fn trace() -> Ppm {
    let canvas_pixels = 200;
    let wall_size = 7.0;
    let wall_z = 10.;
    let pixel_size = wall_size / canvas_pixels as f64;
    let half_wall_size = wall_size / 2.;
    let mut canvas = Canvas::new(canvas_pixels, canvas_pixels);

    let mut s = Sphere::new(0);
    s.material.color = Color::new(1., 0.2, 1.);

    let light_position = Point::new(-10., 10., -10.);
    let light_color = Color::new(1., 1., 1.);
    let light = PointLight::new(light_position, light_color);

    let ray_origin = Point::new(0., 0., -5.);
    for y in 0..canvas_pixels {
        let world_y = half_wall_size - pixel_size * y as f64;
        for x in 0..canvas_pixels {
            let world_x = -half_wall_size + pixel_size * x as f64;
            let pos = Point::new(world_x, world_y, wall_z);
            let r = Ray::new(ray_origin, (pos - ray_origin).normalize());
            let xs = s.intersect(r.clone());

            if xs.hit().is_some() {
                let hit = xs.hit().unwrap();
                let point = r.clone().position(hit.t);
                let hit_object: &Sphere = hit.object.borrow();
                let normal = shader::normal_at(hit_object.clone(), point);
                let eye = -(r).direction;

                let color = shader::lighting(s.material.clone(), light.clone(), point, eye, normal);
                canvas.write_pixel(x, y, color);
            }
        }
    }

    canvas.to_ppm()
}

pub fn main() -> std::io::Result<()> {
    let ppm = trace();
    let mut file = File::create("sphere.ppm")?;
    file.write_all(ppm.blob().as_bytes())?;
    Ok(())
}
