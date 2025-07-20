# Code refactoring plan

This document outlines the proposed reorganization of the ray tracer codebase for better maintainability, extensibility, and performance.

## Current code structure issues

### File organization problems
1. **`ray.rs` is doing too much** - Contains `Ray`, `Sphere`, `Intersection`, and `Intersections`. Should be split up.
2. **`world.rs` mixing concerns** - Contains `World`, `Camera`, view transforms, and rendering logic
3. **Missing shape abstraction** - Only spheres exist, no trait for extensibility
4. **No clear error handling strategy** - Mix of panics, unwraps, and TODOs about Result types

### Performance issues
5. **SmallVec needs replacement** - User analysis confirmed arrays provide substantial performance gains
6. **Missing spatial organization** - No clear place for future acceleration structures

## Proposed new structure

```
src/
├── geometry/           # Core geometric types
│   ├── mod.rs
│   ├── point.rs        # Point type and operations
│   ├── vector.rs       # Vector type and operations  
│   ├── ray.rs          # Ray type and operations
│   └── matrix.rs       # Matrix and transformations
├── shapes/             # Shape implementations
│   ├── mod.rs          # Shape trait definition
│   ├── sphere.rs       # Sphere implementation
│   ├── plane.rs        # Future: Plane implementation
│   └── intersection.rs # Intersection types
├── materials/          # Material and lighting system
│   ├── mod.rs
│   ├── material.rs     # Material properties
│   ├── lighting.rs     # Phong lighting implementation
│   └── color.rs        # Color type and operations
├── scene/              # Scene management
│   ├── mod.rs
│   ├── world.rs        # World container
│   ├── camera.rs       # Camera and view transforms
│   └── lights.rs       # Light sources
├── rendering/          # Rendering pipeline
│   ├── mod.rs
│   ├── renderer.rs     # Main rendering logic
│   └── image.rs        # Image output (canvas/ppm)
├── errors.rs           # Centralized error handling
└── lib.rs             # Public API exports
```

## Core API guidance

The `docs/core-api.md` document defines the essential 100-line ray tracer that can render a lit sphere. This serves as our north star for refactoring - everything in the current codebase is either:

- **Essential core** (vector math, ray-sphere intersection, basic lighting, camera rays, render loop)  
- **Optimization** (arrays vs SmallVec, matrix caching, performance improvements)
- **Features** (transformations, multiple objects, shadows, materials)
- **Architecture** (traits, error handling, modularity, extensibility)

During refactoring, we must ensure the core functions remain simple and fast, while organizing the optimizations and features around them.

## Refactoring phases

### Phase 1: Performance optimizations (immediate wins)
**Priority**: High  
**Timeline**: 1-2 weeks

1. **Replace SmallVec with arrays**
   - Convert `SmallVec<[Intersection; 64]>` to fixed arrays
   - Convert `SmallVec<[Sphere; 64]>` to fixed arrays
   - Confirmed substantial performance gain

2. **Fix critical bugs**
   - Render bounds error (`world.rs:205-206`)
   - Vector magnitude calculation (`coordinate.rs:78`)
   - Point/Vector arithmetic consistency

### Phase 2: Core abstraction (architectural foundation)
**Priority**: High  
**Timeline**: 2-3 weeks

3. **Extract Shape trait**
   - Create `shapes/mod.rs` with `Shape` trait
   - Move `Sphere` to `shapes/sphere.rs`
   - Move intersection types to `shapes/intersection.rs`
   - Update all references

4. **Split geometry module**
   - Extract `Point` and `Vector` to separate files
   - Move `Ray` to `geometry/ray.rs`
   - Keep `matrix.rs` as is (already well-contained)

### Phase 3: Scene management cleanup
**Priority**: Medium  
**Timeline**: 2-3 weeks

5. **Split world.rs**
   - Move `Camera` to `scene/camera.rs`
   - Move view transform functions with camera
   - Keep `World` in `scene/world.rs`
   - Extract lighting to `scene/lights.rs`

6. **Reorganize materials**
   - Move `Color` to `materials/color.rs`
   - Move lighting calculations to `materials/lighting.rs`
   - Create proper `Material` abstraction

### Phase 4: Rendering pipeline
**Priority**: Medium  
**Timeline**: 1-2 weeks

7. **Extract rendering logic**
   - Create `rendering/renderer.rs` for main render loop
   - Move image output to `rendering/image.rs`
   - Clean separation between scene and rendering

8. **Add error handling**
   - Create centralized `errors.rs`
   - Convert panics to Results where appropriate
   - Implement proper error propagation

## Migration strategy

### Backward compatibility approach
- Keep existing public API during transition
- Use `pub use` re-exports in `lib.rs` to maintain compatibility
- Migrate internal code first, then clean up exports

### Testing strategy
- Run full test suite after each phase
- Add integration tests for major refactoring steps
- Benchmark performance after SmallVec replacement

### File movement order
1. Start with new modules (create empty structures)
2. Move code in dependency order (geometry first, then shapes, etc.)
3. Update imports incrementally
4. Clean up old files last

## Benefits of new structure

### Maintainability
- **Single responsibility**: Each module has clear purpose
- **Dependency clarity**: Clean import hierarchy
- **Future growth**: Clear places for new features

### Extensibility  
- **Shape system**: Easy to add new primitives
- **Material system**: Pluggable material types
- **Rendering pipeline**: Swappable renderers and filters

### Performance
- **Arrays over SmallVec**: Confirmed performance gains
- **Better cache locality**: Related code grouped together
- **Optimization opportunities**: Clear bottleneck identification

### Testing
- **Unit testing**: Easier to test isolated components
- **Integration testing**: Clear API boundaries
- **Performance testing**: Isolated benchmarking targets

## Implementation notes

### Critical considerations
- **SmallVec arrays**: Use `[Option<T>; N]` pattern for variable-length collections
- **Error handling**: Prefer `Result<T, E>` over panics for user-facing operations
- **Memory layout**: Consider structure-of-arrays for hot paths
- **Public API**: Minimize breaking changes during transition

### Future extensibility hooks
- **Shape trait**: Foundation for planes, triangles, meshes
- **Material trait**: Support for textures, patterns, PBR
- **Acceleration structures**: Spatial partitioning, BVH
- **Parallel rendering**: Thread-safe scene traversal