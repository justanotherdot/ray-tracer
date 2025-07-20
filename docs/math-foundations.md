# Mathematical foundations

This document explains the core mathematical concepts underlying ray tracing, providing the foundation needed to understand and implement the ray tracer from scratch.

## Basic vector mathematics

### Vectors and points

In 3D graphics, we distinguish between vectors and points:

- **Points** represent positions in 3D space (x, y, z, w=1)
- **Vectors** represent directions and magnitudes (x, y, z, w=0)

The `w` component is used for homogeneous coordinates to enable matrix transformations:
```
Point + Vector = Point (translation)
Point - Point = Vector (direction from one point to another)
Vector + Vector = Vector (addition of directions)
```

### Vector operations

**Magnitude (length)**: √(x² + y² + z²)

**Normalization**: Converting a vector to unit length while preserving direction
```
normalized_vector = vector / magnitude(vector)
```

**Dot product**: Measures how aligned two vectors are
```
a · b = ax*bx + ay*by + az*bz
```
- Result > 0: vectors point in similar directions
- Result = 0: vectors are perpendicular  
- Result < 0: vectors point in opposite directions

**Cross product**: Produces a vector perpendicular to both input vectors
```
a × b = (ay*bz - az*by, az*bx - ax*bz, ax*by - ay*bx)
```

**Reflection**: Used for computing light bounces
```
reflected = incident - 2 * (incident · normal) * normal
```

## Matrix mathematics

### 4×4 transformation matrices

All spatial transformations are represented as 4×4 matrices for homogeneous coordinates:

**Translation matrix**:
```
[1  0  0  tx]
[0  1  0  ty]
[0  0  1  tz]
[0  0  0  1 ]
```

**Scaling matrix**:
```
[sx 0  0  0]
[0  sy 0  0]
[0  0  sz 0]
[0  0  0  1]
```

**Rotation around X-axis**:
```
[1  0       0      0]
[0  cos(θ) -sin(θ) 0]
[0  sin(θ)  cos(θ) 0]
[0  0       0      1]
```

### Matrix operations

**Matrix multiplication**: Used to combine transformations
```
(A × B)[i,j] = Σ(k) A[i,k] * B[k,j]
```

**Matrix inverse**: Undoes a transformation
- Required for transforming rays into object space
- Computed using cofactor method for 4×4 matrices

**Transpose**: Flips matrix across diagonal
- Used in normal transformation: (M⁻¹)ᵀ

## Ray mathematics

### Ray definition

A ray is defined parametrically:
```
point(t) = origin + t * direction
```
Where `t` is a scalar parameter (t ≥ 0 for forward direction).

### Ray-sphere intersection

For a unit sphere at origin, solve:
```
(ray.origin + t * ray.direction) · (ray.origin + t * ray.direction) = 1
```

This expands to a quadratic equation:
```
at² + bt + c = 0
```
Where:
- a = direction · direction
- b = 2 * direction · (origin - sphere_center)  
- c = (origin - sphere_center) · (origin - sphere_center) - radius²

Solutions using quadratic formula:
```
t = (-b ± √(b² - 4ac)) / 2a
```

**Discriminant interpretation**:
- < 0: ray misses sphere
- = 0: ray touches sphere (tangent)
- > 0: ray intersects sphere at two points

### Normal computation

For a sphere, the normal at any surface point is:
```
normal = (surface_point - sphere_center) / radius
```

For transformed objects:
1. Transform point to object space: `object_point = inverse(transform) * world_point`
2. Compute object-space normal: `object_normal = object_point - center`
3. Transform normal to world space: `world_normal = transpose(inverse(transform)) * object_normal`
4. Normalize: `final_normal = normalize(world_normal)`

## Lighting mathematics

### Phong reflection model

The Phong model combines three lighting components:

**Ambient lighting**: Uniform background illumination
```
ambient = material.ambient * light.intensity
```

**Diffuse lighting**: Surface roughness (Lambertian reflection)
```
diffuse = material.diffuse * light.intensity * max(0, normal · light_direction)
```

**Specular lighting**: Shiny highlights
```
specular = material.specular * light.intensity * max(0, reflection · eye_direction)^shininess
```

**Total lighting**:
```
color = ambient + diffuse + specular
```

### Shadow computation

To determine if a point is in shadow:
1. Cast ray from surface point toward light source
2. Check for intersections between point and light
3. If intersection exists with t < distance_to_light, point is shadowed

## Camera mathematics

### Perspective projection

The camera uses a perspective projection defined by:
- Field of view (FOV): Angular width of view
- Aspect ratio: width/height
- View transform: Positions and orients camera

**Pixel-to-world transformation**:
1. Convert pixel coordinates to normalized device coordinates (-1 to 1)
2. Apply inverse perspective projection
3. Transform from camera space to world space

**View transform computation**:
```
forward = normalize(to - from)
right = normalize(cross(forward, up))
true_up = cross(right, forward)

orientation = [right.x    right.y    right.z    0]
              [true_up.x  true_up.y  true_up.z  0] 
              [-forward.x -forward.y -forward.z 0]
              [0          0          0          1]

view_transform = orientation * translation(-from.x, -from.y, -from.z)
```

## Coordinate systems

### Object vs world space

- **Object space**: Local coordinate system where object is defined
- **World space**: Global coordinate system where scene is assembled
- **Camera space**: Coordinate system relative to camera position

**Transformation pipeline**:
1. Object space → World space (model transform)
2. World space → Camera space (view transform)  
3. Camera space → Screen space (projection transform)

### Homogeneous coordinates

Using 4D coordinates (x, y, z, w) enables:
- Distinction between points (w=1) and vectors (w=0)
- Uniform handling of translation via matrix multiplication
- Perspective projection through perspective division (x/w, y/w, z/w)

## Color mathematics

### Color representation

Colors represented as RGB triplets (red, green, blue) with values typically in [0,1].

**Color operations**:
- Addition: (r1+r2, g1+g2, b1+b2)
- Scalar multiplication: (s*r, s*g, s*b)  
- Component multiplication: (r1*r2, g1*g2, b1*b2)

### Gamma correction

Linear color values must be gamma-corrected for display:
```
display_color = pow(linear_color, 1/2.2)
```

Most image formats expect gamma-corrected values.