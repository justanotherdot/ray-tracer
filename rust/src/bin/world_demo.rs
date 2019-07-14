use ray_tracer::color::Color;
use ray_tracer::coordinate::{Point, Vector};
use ray_tracer::ppm::Ppm;
use ray_tracer::ray::Sphere;
use ray_tracer::shader::Material;
use ray_tracer::shader::PointLight;
use ray_tracer::transformation::Transformation;
use ray_tracer::world;
use ray_tracer::world::{Camera, World};
use std::f64::consts::PI;
use std::fs::File;
use std::io::prelude::*;

fn trace() -> Ppm {
    // TODO: Try with default world, what does it look like when rendered?
    //let w: World = Default::default();
    let mut w = World::new();

    let mut floor = Sphere::new(0);
    floor.transform = Transformation::new().scale(10., 0.01, 10.).build();
    floor.material = Material::new();
    floor.material.color = Color::new(1., 0.9, 0.9);
    floor.material.specular = 0.;

    let mut left_wall = Sphere::new(1);
    left_wall.transform = Transformation::new()
        .scale(10., 0.01, 10.)
        .rotate_x(PI / 2.)
        .rotate_y(-PI / 4.)
        .translate(0., 0., 5.)
        .build();
    left_wall.material = floor.material.clone();

    let mut right_wall = Sphere::new(2);
    right_wall.transform = Transformation::new()
        .scale(10., 0.01, 10.)
        .rotate_x(PI / 2.)
        .rotate_y(PI / 4.)
        .translate(0., 0., 5.)
        .build();
    right_wall.material = floor.material.clone();

    let mut middle = Sphere::new(3);
    middle.transform = Transformation::new().translate(-0.5, 1., 0.5).build();
    middle.material = Material::new();
    middle.material.color = Color::new(0.1, 1., 0.5);
    middle.material.diffuse = 0.7;
    middle.material.specular = 0.3;

    let mut right = Sphere::new(4);
    right.transform = Transformation::new()
        .scale(0.5, 0.5, 0.5)
        .translate(1.5, 0.5, -0.5)
        .build();
    right.material = Material::new();
    right.material.color = Color::new(0.5, 1., 0.1);
    right.material.diffuse = 0.7;
    right.material.specular = 0.3;

    let mut left = Sphere::new(5);
    left.transform = Transformation::new()
        .scale(0.33, 0.33, 0.33)
        .translate(-1.5, 0.33, -0.75)
        .build();
    left.material = Material::new();
    left.material.color = Color::new(1., 0.8, 0.1);
    left.material.diffuse = 0.7;
    left.material.specular = 0.3;

    w.objects.push(floor);
    w.objects.push(left_wall);
    w.objects.push(right_wall);
    w.objects.push(middle);
    w.objects.push(right);
    w.objects.push(left);

    w.light = Some(PointLight::new(
        Point::new(-10., 10., -10.),
        Color::new(1., 1., 1.),
    ));

    let mut c = Camera::new(800, 400, PI / 3.0);
    c.transform = world::view_transform(
        Point::new(0., 1.5, -5.),
        Point::new(0., 1., 0.),
        Vector::new(0., 1., 0.),
    );

    let canvas = c.render(w);

    canvas.to_ppm()
}

pub fn main() -> std::io::Result<()> {
    let ppm = trace();
    let mut file = File::create("world_demo.ppm")?;
    file.write_all(ppm.blob().as_bytes())?;
    Ok(())
}
