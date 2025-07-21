use ray_tracer::canvas::Canvas;
use ray_tracer::color::Color;
use ray_tracer::coordinate::Point;
use ray_tracer::ppm::Ppm;
use ray_tracer::ray::{Ray, Sphere};
use ray_tracer::transformation::Transformation;
use std::fs::File;
use std::io::prelude::*;

fn trace() -> Ppm {
    let canvas_pixels = 100;
    let wall_size = 7.0;
    let wall_z = 10.;
    let pixel_size = wall_size / canvas_pixels as f64;
    let half_wall_size = wall_size / 2.;
    let mut canvas = Canvas::new(canvas_pixels, canvas_pixels);

    let ray_strike_color = Color::new(1., 1., 1.);
    let mut s = Sphere::new(0);
    s.set_transform(
        Transformation::new()
            .translate(0., 0.0, 0.)
            //.scale(0.5, 1., 1.)
            //.rotate_z(std::f64::consts::PI / 4.)
            .build(),
    );

    let ray_origin = Point::new(0., 0., -5.);
    for y in 0..canvas_pixels {
        let world_y = half_wall_size - pixel_size * y as f64;
        for x in 0..canvas_pixels {
            let world_x = -half_wall_size + pixel_size * x as f64;
            let pos = Point::new(world_x, world_y, wall_z);
            let r = Ray::new(ray_origin, (pos - ray_origin).normalize());
            let xs = s.intersect(&r);

            if xs.hit().is_some() {
                canvas.write_pixel(x, y, ray_strike_color.clone());
            }
        }
    }

    canvas.to_ppm()
}

pub fn main() -> std::io::Result<()> {
    let ppm = trace();
    let mut file = File::create("circle.ppm")?;
    file.write_all(ppm.blob().as_bytes())?;
    Ok(())
}
