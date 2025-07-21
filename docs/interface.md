# Core ray tracer interfaces

This document defines the essential data types and function signatures needed to build an extensible, maintainable ray tracer from scratch. These interfaces represent the minimal abstractions required while allowing for future extensibility.

## Fundamental geometric types

### Point and Vector
```rust
// 3D point in space with homogeneous coordinates
struct Point {
    x: f64,
    y: f64, 
    z: f64,
    w: f64, // Always 1.0 for points
}

// 3D direction vector with homogeneous coordinates  
struct Vector {
    x: f64,
    y: f64,
    z: f64,
    w: f64, // Always 0.0 for vectors
}

// Essential operations
impl Vector {
    fn dot(&self, other: &Vector) -> f64;
    fn cross(&self, other: &Vector) -> Vector;
    fn magnitude(&self) -> f64;
    fn normalize(&self) -> Vector;
    fn reflect(&self, normal: &Vector) -> Vector;
}

impl Point {
    // Point - Point = Vector
    // Point + Vector = Point
    // Point - Vector = Point
}
```

### Ray
```rust
// Parametric ray: origin + t * direction
struct Ray {
    origin: Point,
    direction: Vector, // Should be normalized
}

impl Ray {
    fn new(origin: Point, direction: Vector) -> Self;
    fn position(&self, t: f64) -> Point;
    fn transform(&self, matrix: &Matrix) -> Ray;
}
```

### Matrix and transformations
```rust
// 4x4 transformation matrix using fixed arrays for performance
struct Matrix {
    data: [f64; 16],
}

impl Matrix {
    fn identity() -> Self;
    fn translate(x: f64, y: f64, z: f64) -> Self;
    fn scale(x: f64, y: f64, z: f64) -> Self;
    fn rotate_x(radians: f64) -> Self;
    fn rotate_y(radians: f64) -> Self;
    fn rotate_z(radians: f64) -> Self;
    
    fn multiply(&self, other: &Matrix) -> Matrix;
    fn inverse(&self) -> Matrix;
    fn transpose(&self) -> Matrix;
    
    // Transform points and vectors
    fn transform_point(&self, point: &Point) -> Point;
    fn transform_vector(&self, vector: &Vector) -> Vector;
}
```

## Core shape abstraction

### Shape trait
```rust
// Intersection result from ray-shape test
struct Intersection {
    t: f64,              // Distance along ray
    point: Point,        // World-space intersection point
    normal: Vector,      // Surface normal at intersection
    material_id: usize,  // Reference to material
    inside: bool,        // Ray hit from inside shape
}

// All renderable objects implement this trait
trait Shape {
    // Test ray intersection, return sorted intersections
    fn intersect(&self, ray: &Ray) -> Vec<Intersection>;
    
    // Get surface normal at world point
    fn normal_at(&self, world_point: &Point) -> Vector;
    
    // Get transformation matrix
    fn transform(&self) -> &Matrix;
    fn set_transform(&mut self, transform: Matrix);
    
    // Material assignment
    fn material_id(&self) -> usize;
    fn set_material_id(&mut self, id: usize);
    
    // Bounding box for acceleration structures
    fn bounds(&self) -> BoundingBox;
}

// Concrete shape implementations
struct Sphere {
    transform: Matrix,
    material_id: usize,
}

struct Plane {
    transform: Matrix, 
    material_id: usize,
}

struct Triangle {
    p1: Point,
    p2: Point, 
    p3: Point,
    transform: Matrix,
    material_id: usize,
}
```

## Material and lighting system

### Color
```rust
// RGB color with floating point components
struct Color {
    r: f64,
    g: f64, 
    b: f64,
}

impl Color {
    fn new(r: f64, g: f64, b: f64) -> Self;
    fn black() -> Self;
    fn white() -> Self;
    
    // Color arithmetic
    fn add(&self, other: &Color) -> Color;
    fn multiply(&self, other: &Color) -> Color; // Component-wise
    fn scale(&self, scalar: f64) -> Color;
}
```

### Material system
```rust
// Surface material properties
struct Material {
    color: Color,
    ambient: f64,     // Ambient reflection [0,1]
    diffuse: f64,     // Diffuse reflection [0,1] 
    specular: f64,    // Specular reflection [0,1]
    shininess: f64,   // Specular exponent
    reflective: f64,  // Mirror reflection [0,1]
    transparency: f64, // Transparency [0,1]
    refractive_index: f64, // For refraction
}

// Future: Pattern and texture support
trait Pattern {
    fn color_at(&self, point: &Point) -> Color;
}

// Light sources
struct PointLight {
    position: Point,
    intensity: Color,
}

// Future: Area lights, directional lights
trait Light {
    fn illuminate(&self, point: &Point) -> (Vector, Color, f64); // direction, color, distance
}
```

### Lighting computation
```rust
// Phong lighting model
fn compute_lighting(
    material: &Material,
    light: &PointLight, 
    point: &Point,
    eye_vector: &Vector,
    normal_vector: &Vector,
    in_shadow: bool,
) -> Color;

// Shadow testing
fn is_shadowed(world: &World, point: &Point, light: &PointLight) -> bool;
```

## Scene management

### World and scene graph
```rust
// Scene container with objects and lights
struct World {
    shapes: Vec<Box<dyn Shape>>,
    lights: Vec<Box<dyn Light>>, 
    materials: Vec<Material>,
}

impl World {
    fn new() -> Self;
    fn add_shape(&mut self, shape: Box<dyn Shape>);
    fn add_light(&mut self, light: Box<dyn Light>);
    fn add_material(&mut self, material: Material) -> usize; // Returns material ID
    
    // Ray intersection against all objects
    fn intersect(&self, ray: &Ray) -> Vec<Intersection>;
    
    // Compute color for ray
    fn color_at(&self, ray: &Ray, remaining_bounces: u32) -> Color;
    
    // Shade intersection point
    fn shade_hit(&self, intersection: &Intersection, ray: &Ray, remaining_bounces: u32) -> Color;
}

// For complex scenes: spatial acceleration
trait AccelerationStructure {
    fn intersect(&self, ray: &Ray) -> Vec<&dyn Shape>;
    fn add_shape(&mut self, shape: &dyn Shape);
}

struct BoundingVolumeHierarchy {
    // BVH implementation
}
```

## Camera and rendering

### Camera system
```rust
struct Camera {
    hsize: usize,           // Horizontal resolution
    vsize: usize,           // Vertical resolution  
    field_of_view: f64,     // Vertical FOV in radians
    transform: Matrix,      // Camera position/orientation
    
    // Computed values
    pixel_size: f64,
    half_width: f64,
    half_height: f64,
}

impl Camera {
    fn new(hsize: usize, vsize: usize, field_of_view: f64) -> Self;
    
    // Generate ray for pixel coordinates
    fn ray_for_pixel(&self, x: usize, y: usize) -> Ray;
    
    // Camera positioning helper
    fn look_at(&mut self, from: Point, to: Point, up: Vector);
    
    // Render world to image
    fn render(&self, world: &World) -> Image;
}
```

### Image output
```rust
// Image buffer and output
struct Image {
    width: usize,
    height: usize,
    pixels: Vec<Color>, // Row-major order
}

impl Image {
    fn new(width: usize, height: usize) -> Self;
    fn set_pixel(&mut self, x: usize, y: usize, color: Color);
    fn get_pixel(&self, x: usize, y: usize) -> &Color;
    
    // Output formats
    fn to_ppm(&self) -> String;
    fn to_png(&self, path: &str) -> Result<(), Box<dyn std::error::Error>>;
}
```

## Error handling

### Result types for robustness
```rust
// Ray tracer specific errors
#[derive(Debug)]
enum RayTracerError {
    InvalidMatrix,
    InvalidTransformation, 
    InvalidMaterial,
    InvalidScene,
    RenderError(String),
    IoError(std::io::Error),
}

type Result<T> = std::result::Result<T, RayTracerError>;

// Critical operations should return Results
impl Matrix {
    fn try_inverse(&self) -> Result<Matrix>;
}

impl Camera {
    fn try_render(&self, world: &World) -> Result<Image>;
}
```

## Performance considerations

### Memory layout optimization
```rust
// Use fixed-size arrays instead of Vec where possible
const MAX_INTERSECTIONS: usize = 64;
type IntersectionArray = [Option<Intersection>; MAX_INTERSECTIONS];

// Structure of arrays for better cache locality
struct ShapeArray {
    transforms: Vec<Matrix>,
    material_ids: Vec<usize>,
    shape_types: Vec<ShapeType>,
    shape_data: Vec<ShapeData>, // Union-like data
}
```

### Parallel processing interfaces
```rust
// Tile-based rendering for parallelization
struct RenderTile {
    x_start: usize,
    y_start: usize, 
    width: usize,
    height: usize,
}

impl Camera {
    fn render_tile(&self, world: &World, tile: RenderTile) -> Vec<Color>;
    fn render_parallel(&self, world: &World, num_threads: usize) -> Image;
}
```

## Extensibility points

### Plugin architecture
```rust
// Allow custom shapes, materials, lights
trait ShapeFactory {
    fn create_shape(&self, params: &ShapeParams) -> Box<dyn Shape>;
}

trait MaterialFactory {
    fn create_material(&self, params: &MaterialParams) -> Material;
}

// Scene loading
trait SceneLoader {
    fn load_scene(&self, path: &str) -> Result<World>;
}

// Custom sampling and filtering
trait Sampler {
    fn sample_pixel(&self, x: f64, y: f64) -> Vec<(f64, f64)>; // Sub-pixel samples
}

trait Filter {
    fn filter_color(&self, colors: &[Color]) -> Color;
}
```

## Usage examples

### Basic scene setup
```rust
// Create world
let mut world = World::new();

// Add material
let red_material = Material {
    color: Color::new(1.0, 0.2, 0.2),
    ambient: 0.1,
    diffuse: 0.9,
    specular: 0.9,
    shininess: 200.0,
    ..Default::default()
};
let material_id = world.add_material(red_material);

// Add sphere
let mut sphere = Sphere::new();
sphere.set_material_id(material_id);
sphere.set_transform(Matrix::translate(0.0, 0.0, -5.0));
world.add_shape(Box::new(sphere));

// Add light
let light = PointLight {
    position: Point::new(-10.0, 10.0, -10.0),
    intensity: Color::white(),
};
world.add_light(Box::new(light));

// Setup camera
let mut camera = Camera::new(800, 600, std::f64::consts::PI / 3.0);
camera.look_at(
    Point::new(0.0, 0.0, 0.0),   // from
    Point::new(0.0, 0.0, -1.0),  // to
    Vector::new(0.0, 1.0, 0.0),  // up
);

// Render
let image = camera.render(&world)?;
image.to_png("output.png")?;
```

This interface design prioritizes:
- **Clear separation of concerns** between geometry, materials, lighting, and rendering
- **Extensibility** through traits and plugin architecture
- **Performance** with fixed arrays and parallel processing support
- **Maintainability** with proper error handling and type safety
- **Future growth** with hooks for advanced features like global illumination