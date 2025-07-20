use criterion::{black_box, criterion_group, criterion_main, Criterion};

use ray_tracer::{
    coordinate::{Point, Vector},
    ray::{Ray, Sphere},
    transformation::Transformation,
    world::{Camera, World},
};
use std::f64::consts::PI;

// Benchmark core operations that we're optimizing

fn benchmark_ray_sphere_intersection(c: &mut Criterion) {
    let sphere = Sphere::new(0);
    let ray = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));

    c.bench_function("ray_sphere_intersection", |b| {
        b.iter(|| black_box(sphere.intersect(black_box(&ray))))
    });
}

fn benchmark_world_intersect(c: &mut Criterion) {
    let w: World = Default::default();
    let ray = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));

    c.bench_function("world_intersect", |b| {
        b.iter(|| black_box(w.intersect(black_box(&ray))))
    });
}

fn benchmark_matrix_inverse(c: &mut Criterion) {
    let transform = Transformation::new()
        .translate(1., 2., 3.)
        .rotate_x(PI / 4.)
        .scale(2., 2., 2.)
        .build();

    c.bench_function("matrix_inverse", |b| {
        b.iter(|| black_box(transform.inverse()))
    });
}

fn benchmark_vector_operations(c: &mut Criterion) {
    let v1 = Vector::new(1., 2., 3.);
    let v2 = Vector::new(4., 5., 6.);

    c.bench_function("vector_dot", |b| {
        b.iter(|| black_box(v1.dot(black_box(&v2))))
    });

    c.bench_function("vector_normalize", |b| b.iter(|| black_box(v1.normalize())));

    c.bench_function("vector_cross", |b| {
        b.iter(|| black_box(v1.cross(black_box(&v2))))
    });
}

fn benchmark_small_render(c: &mut Criterion) {
    let mut camera = Camera::new(50, 50, PI / 3.0);
    camera.transform = ray_tracer::world::view_transform(
        Point::new(0., 1.5, -5.),
        Point::new(0., 1., 0.),
        Vector::new(0., 1., 0.),
    );

    c.bench_function("render_50x50", |b| {
        b.iter(|| {
            let w: World = Default::default();
            black_box(camera.render(black_box(w)))
        })
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = benchmark_ray_sphere_intersection,
             benchmark_world_intersect,
             benchmark_matrix_inverse,
             benchmark_vector_operations,
             benchmark_small_render
);
criterion_main!(benches);
