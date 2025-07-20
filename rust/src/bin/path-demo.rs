use ray_tracer::canvas::Canvas;
use ray_tracer::color::Color;
use ray_tracer::coordinate::{Point, Vector};
use ray_tracer::ppm::Ppm;
use std::fs::File;
use std::io::prelude::*;

struct Projectile {
    position: Point,
    velocity: Vector,
}

struct Environment {
    gravity: Vector,
    wind: Vector,
}

fn tick(env: &Environment, proj: Projectile) -> Projectile {
    let position = proj.position + proj.velocity;
    let velocity = proj.velocity + env.gravity + env.wind;
    Projectile { position, velocity }
}

fn simulate_and_plot() -> Ppm {
    let position = Point::new(0., 1., 0.);
    let velocity = Vector::new(1., 1.8, 0.).normalize() * 11.25;
    let mut p = Projectile { position, velocity };

    let mut canvas = Canvas::new(900, 553);
    canvas.fill(Color::new(1.0, 1.0, 1.0));

    let gravity = Vector::new(0., -0.1, 0.);
    let wind = Vector::new(-0.01, 0., 0.);
    let e = Environment { gravity, wind };

    loop {
        p = tick(&e, p);

        let x = p.position.x.max(0.) as usize;
        let y = (550. - (p.position.y.max(0.))).min(549.) as usize;
        canvas.write_pixel(x, y, Color::new(0.0, 0.0, 0.0));

        if p.position.y <= 0. {
            break;
        }
    }

    canvas.to_ppm()
}

pub fn main() -> std::io::Result<()> {
    let ppm = simulate_and_plot();
    let mut file = File::create("plot.ppm")?;
    file.write_all(ppm.blob().as_bytes())?;
    Ok(())
}
