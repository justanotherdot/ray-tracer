use rust_ray_tracer::coordinate::{Point, Vector};

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

fn simulate() {
    let position = Point::new(0., 1., 0.);
    let velocity = Vector::new(1., 1., 0.).normalize();
    let mut p = Projectile { position, velocity };

    let gravity = Vector::new(0., -0.1, 0.);
    let wind = Vector::new(-0.01, 0., 0.);
    let e = Environment { gravity, wind };

    println!("({}, {})", p.position.x, p.position.y);
    loop {
        p = tick(&e, p);
        println!("({}, {})", p.position.x, p.position.y);

        if p.position.y <= 0. {
            break;
        }
    }
}

pub fn main() {
    simulate();
}
