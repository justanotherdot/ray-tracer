use rust_ray_tracer::canvas::Canvas;
use rust_ray_tracer::color::Color;
use rust_ray_tracer::coordinate::Point;
use rust_ray_tracer::ppm::Ppm;
use rust_ray_tracer::transformation::Transformation;
use std::fs::File;
use std::io::prelude::*;

fn plot() -> Ppm {
    let dim = 500;
    let width = dim;
    let height = dim;
    let mut canvas = Canvas::new(width, height);
    canvas.fill(Color::new(1.0, 1.0, 1.0));

    let p = Point::new(0., 0., 1.) * 125.;
    let slice_deg = std::f64::consts::PI / 6.;
    (0..12).for_each(|i| {
        let t = Transformation::new()
            .rotate_y((i as f64) * slice_deg)
            .build();
        let p1 = t * p;
        canvas.write_pixel(
            (250. + p1.x) as usize,
            (500. - (250. + p1.z)) as usize,
            Color::new(0., 0., 0.),
        );
    });

    canvas.to_ppm()
}

pub fn main() -> std::io::Result<()> {
    let ppm = plot();
    let mut file = File::create("clock.ppm")?;
    file.write_all(ppm.blob().as_bytes())?;
    Ok(())
}
