use dhat::Profiler;
use ray_tracer::{
    color::Color,
    coordinate::{Point, Vector},
    ray::{Ray, Sphere},
    shader::{Material, PointLight},
    transformation::Transformation,
    world::{Camera, World},
};
use std::f64::consts::PI;

#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

fn produce_world() -> World {
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

    w.add_object(floor);
    w.add_object(left_wall);
    w.add_object(right_wall);
    w.add_object(middle);
    w.add_object(right);
    w.add_object(left);

    w.light = Some(PointLight::new(
        Point::new(-10., 10., -10.),
        Color::new(1., 1., 1.),
    ));

    w
}

fn memory_test_intersection_collection() {
    println!("=== Memory test: Intersection collection ===");
    let _profiler = Profiler::new_heap();

    let sphere = Sphere::new(0);

    // Simulate many ray-sphere intersections (hot path)
    for _ in 0..1000 {
        let ray = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
        let _intersections = sphere.intersect(&ray);
        // This creates and destroys SmallVec many times
    }
}

fn memory_test_world_intersect() {
    println!("=== Memory test: World intersection ===");
    let _profiler = Profiler::new_heap();

    let world = produce_world();

    // Simulate many world intersections
    for i in 0..100 {
        let x_offset = (i as f64) * 0.1;
        let ray = Ray::new(Point::new(x_offset, 0., -5.), Vector::new(0., 0., 1.));
        let _intersections = world.intersect(&ray);
        // This exercises the SmallVec collection and extension
    }
}

fn memory_test_matrix_operations() {
    println!("=== Memory test: Matrix operations ===");
    let _profiler = Profiler::new_heap();

    // Test matrix inverse computation (expensive operation)
    for _ in 0..100 {
        let transform = Transformation::new()
            .translate(1., 2., 3.)
            .rotate_x(PI / 4.)
            .scale(2., 2., 2.)
            .build();
        let _inverse = transform.inverse();
        // Check if this allocates temporary matrices
    }
}

fn memory_test_small_render() {
    println!("=== Memory test: Small render ===");
    let _profiler = Profiler::new_heap();

    let world = produce_world();
    let mut camera = Camera::new(50, 50, PI / 3.0);
    camera.transform = ray_tracer::world::view_transform(
        Point::new(0., 1.5, -5.),
        Point::new(0., 1., 0.),
        Vector::new(0., 1., 0.),
    );

    let _canvas = camera.render(world);
    // This exercises the full rendering pipeline
}

fn memory_test_ppm_generation() {
    println!("=== Memory test: PPM generation ===");
    let _profiler = Profiler::new_heap();

    let world = produce_world();
    let mut camera = Camera::new(100, 75, PI / 3.0);
    camera.transform = ray_tracer::world::view_transform(
        Point::new(0., 1.5, -5.),
        Point::new(0., 1., 0.),
        Vector::new(0., 1., 0.),
    );

    let canvas = camera.render(world);
    let _ppm = canvas.to_ppm();
    // This likely allocates large strings
}

fn main() {
    println!("Ray tracer memory profiling");
    println!("===========================");

    // Test individual components
    memory_test_intersection_collection();
    memory_test_world_intersect();
    memory_test_matrix_operations();
    memory_test_small_render();
    memory_test_ppm_generation();

    println!("\nMemory profiling complete. Check dhat-heap.json for detailed results.");
    println!("View with: dh_view.py dhat-heap.json");
}
