# Ray tracer roadmap

This document outlines the bugs, optimizations, remaining work, and future enhancements for the ray tracer.

## Critical bugs to fix

### 1. Render bounds error
**File**: `world.rs:205-206`  
**Issue**: Loop bounds use `< hsize-1` and `< vsize-1`, missing the final row and column of pixels.  
**Fix**: Change to `< hsize` and `< vsize`  
**Impact**: Missing pixels in rendered output

### 2. Vector magnitude calculation
**File**: `coordinate.rs:78`  
**Issue**: Magnitude includes `w` component, but for 3D vectors should only use x,y,z.  
**Fix**: Change to `v.x.powf(2.) + v.y.powf(2.) + v.z.powf(2.)`  
**Impact**: Incorrect vector lengths affecting lighting and normalization

### 3. Point/Vector arithmetic consistency
**Files**: Various arithmetic implementations  
**Issue**: Some operations incorrectly modify the `w` component.  
**Fix**: Ensure Point operations preserve `w=1`, Vector operations preserve `w=0`  
**Impact**: Transformation correctness

## Performance optimizations

### Immediate wins (low effort, high impact)

1. **Replace SmallVec with arrays**
   - Convert `SmallVec<[Intersection; 64]>` to fixed arrays
   - Convert `SmallVec<[Sphere; 64]>` to fixed arrays  
   - Eliminates allocation overhead and improves cache locality
   - **Confirmed substantial performance gain from your analysis**

2. **Cache matrix inverses**
   - Store inverse alongside transform in Sphere
   - Lazy computation and caching
   - Avoids repeated expensive inverse calculations

3. **Parallelize rendering**
   - Use rayon for parallel pixel computation
   - Chunk-based work distribution
   - Should scale with available CPU cores

4. **Optimize intersection sorting**
   - Pre-allocate intersection vectors
   - Use partial sorting for hit detection

### Medium effort optimizations

5. **SIMD vector operations**
   - Use platform SIMD for basic vector math
   - Batch operations where possible

6. **Reduce remaining allocations**
   - Object pooling for intersections
   - Stack-allocated temporary vectors
   - Eliminate String allocations in PPM generation

7. **Spatial acceleration**
   - Implement bounding volume hierarchy (BVH)
   - Grid-based spatial partitioning
   - Critical for scenes with many objects

### Advanced optimizations

8. **GPU acceleration**
   - Port core intersection kernels to GPU
   - Use compute shaders or CUDA
   - Most beneficial for compute-heavy scenes

9. **Memory layout optimization**
   - Structure of arrays for better cache locality
   - Custom allocators for frequent objects
   - Profile-guided optimization based on actual usage

## Remaining work to complete ray tracer

### Core features needed

1. **Additional primitive shapes**
   - Planes, cylinders, cones
   - Triangle meshes for arbitrary geometry
   - Bounding boxes for acceleration

2. **Material system improvements**
   - Texture mapping and UV coordinates
   - Procedural patterns and noise
   - Transparency and refraction
   - Reflective materials

3. **Advanced lighting**
   - Multiple light sources
   - Area lights and soft shadows
   - Ambient occlusion
   - Global illumination basics

4. **Scene management**
   - Hierarchical scene graph
   - Object instancing
   - Transform hierarchies

### Quality of life improvements

5. **Scene loading**
   - OBJ file parser for meshes
   - Scene description format (JSON/YAML)
   - Asset management system

6. **Output improvements**
   - Multiple image formats (PNG, JPEG)
   - HDR output support
   - Gamma correction and tone mapping

7. **Development tools**
   - Interactive preview window
   - Performance profiling tools
   - Debug visualization modes

## Future enhancements

### Advanced rendering techniques

1. **Global illumination**
   - Path tracing for realistic lighting
   - Bidirectional path tracing
   - Photon mapping

2. **Advanced materials**
   - Physically-based rendering (PBR)
   - Subsurface scattering
   - Volumetric rendering

3. **Motion and animation**
   - Motion blur support
   - Keyframe animation system
   - Physics integration

### Performance and scalability

4. **Distributed rendering**
   - Network-based render farms
   - Cloud rendering support
   - Progressive rendering

5. **Adaptive sampling**
   - Importance sampling
   - Adaptive anti-aliasing
   - Denoising algorithms

6. **Real-time features**
   - Temporal reprojection
   - Level-of-detail systems
   - Streaming asset loading

### Developer experience

7. **Language bindings**
   - C API for integration
   - Python bindings for scripting
   - WebAssembly for web demos

8. **Ecosystem integration**
   - Blender plugin
   - USD format support
   - Integration with 3D pipelines

## Implementation priority

### Phase 1: Bug fixes and basic optimizations (1-2 weeks)
- Fix render bounds error
- Fix vector magnitude calculation
- Add parallel rendering
- Cache matrix inverses

### Phase 2: Core features completion (4-6 weeks)  
- Additional primitive shapes
- Multiple light sources
- Basic transparency and reflection
- Scene loading format

### Phase 3: Advanced optimization (6-8 weeks)
- Spatial acceleration structures
- SIMD optimization
- Memory layout improvements
- GPU acceleration research

### Phase 4: Advanced features (8-12 weeks)
- Path tracing implementation
- PBR material system
- Advanced lighting techniques
- Production-ready output pipeline

## Success metrics

### Performance targets
- Profile current performance to establish baseline metrics
- Support 100k+ primitives efficiently  
- Memory usage optimization for complex scenes
- 90%+ CPU utilization on available cores

### Quality targets
- Physically accurate lighting model
- Support for industry-standard scene formats
- Professional-quality output formats
- Comprehensive test suite with >95% coverage

### Usability targets
- Simple scene description language
- Interactive preview capabilities  
- Comprehensive documentation
- Example scenes and tutorials