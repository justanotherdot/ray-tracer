# Core ray tracer API

This document defines the absolute minimum API needed to build a working ray tracer from scratch. This is the essential "Ray Tracer in 100 Lines" - the core functions you need to render your first sphere.

## The key insight

Ray tracing boils down to just 5 essential components:

1. **Vector math** (dot product, normalize)
2. **Ray-sphere intersection** (quadratic formula)  
3. **Simple lighting** (diffuse shading)
4. **Camera ray generation** (perspective projection)
5. **Pixel iteration** (render loop)

Everything else is optimization, additional features, or architectural improvements built on top of these fundamentals.

## Step 1: Basic geometry (20 lines)

```rust
// 3D point in space
#[derive(Copy, Clone, Debug)]
struct Point {
    x: f64,
    y: f64, 
    z: f64,
}

// 3D direction vector
#[derive(Copy, Clone, Debug)]  
struct Vector {
    x: f64,
    y: f64,
    z: f64,
}

// Essential vector operations
impl Vector {
    fn dot(self, other: Vector) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
    
    fn length(self) -> f64 {
        self.dot(self).sqrt()
    }
    
    fn normalize(self) -> Vector {
        let len = self.length();
        Vector { x: self.x / len, y: self.y / len, z: self.z / len }
    }
}

// Point and vector arithmetic
impl std::ops::Sub for Point {
    type Output = Vector;
    fn sub(self, other: Point) -> Vector {
        Vector { x: self.x - other.x, y: self.y - other.y, z: self.z - other.z }
    }
}

impl std::ops::Add<Vector> for Point {
    type Output = Point;
    fn add(self, vec: Vector) -> Point {
        Point { x: self.x + vec.x, y: self.y + vec.y, z: self.z + vec.z }
    }
}

impl std::ops::Mul<f64> for Vector {
    type Output = Vector;
    fn mul(self, scalar: f64) -> Vector {
        Vector { x: self.x * scalar, y: self.y * scalar, z: self.z * scalar }
    }
}
```

## Step 2: Ray definition (5 lines)

```rust
// Ray: origin + t * direction
#[derive(Debug)]
struct Ray {
    origin: Point,
    direction: Vector, // Should be normalized
}

impl Ray {
    fn at(self, t: f64) -> Point {
        self.origin + self.direction * t
    }
}
```

## Step 3: Sphere intersection (15 lines)

```rust
// Simple sphere at origin with radius
#[derive(Debug)]
struct Sphere {
    center: Point,
    radius: f64,
}

// Ray-sphere intersection using quadratic formula
fn intersect_sphere(ray: &Ray, sphere: &Sphere) -> Option<f64> {
    let oc = ray.origin - sphere.center;
    
    let a = ray.direction.dot(ray.direction);
    let b = 2.0 * oc.dot(ray.direction);
    let c = oc.dot(oc) - sphere.radius * sphere.radius;
    
    let discriminant = b * b - 4.0 * a * c;
    
    if discriminant < 0.0 {
        None
    } else {
        let t1 = (-b - discriminant.sqrt()) / (2.0 * a);
        let t2 = (-b + discriminant.sqrt()) / (2.0 * a);
        
        // Return closest positive intersection
        if t1 > 0.0 { Some(t1) } 
        else if t2 > 0.0 { Some(t2) } 
        else { None }
    }
}
```

## Step 4: Color and lighting (10 lines)

```rust
// RGB color
#[derive(Copy, Clone, Debug)]
struct Color {
    r: f64,
    g: f64, 
    b: f64,
}

// Simple lighting calculation
fn compute_color(hit_point: Point, sphere: &Sphere, light_pos: Point) -> Color {
    // Surface normal (outward from sphere center)
    let normal = (hit_point - sphere.center).normalize();
    
    // Light direction
    let light_dir = (light_pos - hit_point).normalize();
    
    // Simple diffuse lighting (Lambertian)
    let brightness = normal.dot(light_dir).max(0.0);
    
    Color { r: brightness, g: brightness, b: brightness }
}
```

## Step 5: Camera and rendering (30 lines)

```rust
// Simple camera
struct Camera {
    origin: Point,
    lower_left: Point,
    horizontal: Vector,
    vertical: Vector,
}

impl Camera {
    fn new(width: f64, height: f64) -> Camera {
        let aspect_ratio = width / height;
        let viewport_height = 2.0;
        let viewport_width = aspect_ratio * viewport_height;
        
        let origin = Point { x: 0.0, y: 0.0, z: 0.0 };
        let horizontal = Vector { x: viewport_width, y: 0.0, z: 0.0 };
        let vertical = Vector { x: 0.0, y: viewport_height, z: 0.0 };
        let lower_left = origin 
            - horizontal * 0.5 
            - vertical * 0.5 
            - Vector { x: 0.0, y: 0.0, z: 1.0 };
            
        Camera { origin, lower_left, horizontal, vertical }
    }
    
    fn get_ray(&self, u: f64, v: f64) -> Ray {
        Ray {
            origin: self.origin,
            direction: (self.lower_left + self.horizontal * u + self.vertical * v - self.origin).normalize()
        }
    }
}

// Main render function
fn render_pixel(x: usize, y: usize, width: usize, height: usize, 
                camera: &Camera, sphere: &Sphere, light: Point) -> Color {
    let u = x as f64 / width as f64;
    let v = y as f64 / height as f64;
    
    let ray = camera.get_ray(u, v);
    
    if let Some(t) = intersect_sphere(&ray, sphere) {
        let hit_point = ray.at(t);
        compute_color(hit_point, sphere, light)
    } else {
        // Background color
        Color { r: 0.2, g: 0.3, b: 1.0 }
    }
}
```

## Step 6: Complete minimal ray tracer (20 lines)

```rust
fn main() {
    let width = 400;
    let height = 300;
    
    // Scene setup
    let camera = Camera::new(width as f64, height as f64);
    let sphere = Sphere { 
        center: Point { x: 0.0, y: 0.0, z: -1.0 }, 
        radius: 0.5 
    };
    let light = Point { x: 2.0, y: 2.0, z: 0.0 };
    
    // Render to PPM format
    println!("P3\n{} {}\n255", width, height);
    
    for y in (0..height).rev() {
        for x in 0..width {
            let color = render_pixel(x, y, width, height, &camera, &sphere, light);
            
            let r = (255.0 * color.r.clamp(0.0, 1.0)) as u8;
            let g = (255.0 * color.g.clamp(0.0, 1.0)) as u8;
            let b = (255.0 * color.b.clamp(0.0, 1.0)) as u8;
            
            println!("{} {} {}", r, g, b);
        }
    }
}
```

## Essential API summary (what you actually need)

### Core types (4 types)
```rust
struct Point { x: f64, y: f64, z: f64 }
struct Vector { x: f64, y: f64, z: f64 }  
struct Ray { origin: Point, direction: Vector }
struct Color { r: f64, g: f64, b: f64 }
```

### Core functions (6 functions)
```rust
// Vector math
fn dot(a: Vector, b: Vector) -> f64
fn normalize(v: Vector) -> Vector

// Ray tracing
fn intersect_sphere(ray: &Ray, sphere: &Sphere) -> Option<f64>
fn compute_color(hit_point: Point, normal: Vector, light_dir: Vector) -> Color

// Rendering
fn get_ray_for_pixel(x: usize, y: usize, camera: &Camera) -> Ray
fn render_pixel(ray: Ray, scene: &Scene) -> Color
```

That's it! Everything else is optimization, additional features, or convenience functions.

## Extension points for full ray tracer

Once you have this core working, you extend by:

1. **Multiple objects**: `Vec<Sphere>` and loop through intersections
2. **Transformations**: Add `Matrix` and transform rays into object space
3. **Materials**: Add `Material` struct with surface properties  
4. **Multiple lights**: `Vec<Light>` and sum contributions
5. **Shadows**: Cast ray from hit point to light
6. **Reflection**: Recursively trace reflected rays
7. **Performance**: Replace `Vec` with arrays, add spatial acceleration

But the core above renders a lit sphere - a complete ray tracer in ~100 lines of essential code.