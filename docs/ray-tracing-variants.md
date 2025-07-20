# Ray tracing variants

This document explains the different types of ray tracers and how they compare, providing context for this project's approach and potential future directions.

## Current implementation: Whitted ray tracing

This project currently implements **Whitted-style ray tracing** (also called classic or traditional ray tracing), which:

- Shoots primary rays from camera through each pixel
- Handles recursive reflection and refraction rays
- Uses simple lighting models (Phong/Blinn-Phong shading)
- Supports hard shadows via shadow rays
- Fast and predictable performance
- Limited to local illumination (no light bouncing between surfaces)

This approach produces clean, sharp images with perfect reflections and refractions, but lacks the subtle lighting effects of global illumination.

## Other ray tracing approaches

### Path tracing (Monte Carlo ray tracing)
**What it is**: Shoots many random rays per pixel using Monte Carlo integration to solve the rendering equation.

**Advantages**:
- Physically accurate global illumination
- Natural soft shadows, color bleeding, ambient occlusion
- Handles complex lighting phenomena (caustics, subsurface scattering)
- Converges to the mathematically correct lighting solution

**Disadvantages**:
- Much slower (hundreds to thousands of samples per pixel)
- Noisy output that requires denoising or many samples
- More complex to implement correctly

**Use cases**: Film rendering, architectural visualization, product visualization where accuracy matters more than speed.

### Bidirectional path tracing
**What it is**: Shoots rays from both camera and light sources, then connects the paths.

**Advantages**:
- More efficient than basic path tracing for certain lighting scenarios
- Better at handling "difficult" lighting (indoor scenes, small light sources)
- Still physically accurate like path tracing

**Disadvantages**:
- Even more complex to implement
- Not always faster than path tracing
- Memory overhead for storing light paths

**Use cases**: Research, high-end offline rendering where path tracing struggles.

### Photon mapping
**What it is**: Two-pass algorithm that first shoots photons from lights to build a spatial map, then uses this during rendering.

**Advantages**:
- Excellent for caustics (light focused through glass/water)
- Good for subsurface scattering effects
- Can handle specular-diffuse-specular light paths efficiently

**Disadvantages**:
- Complex implementation with many parameters to tune
- Memory intensive (storing millions of photons)
- Bias-variance tradeoffs in the photon map

**Use cases**: Scenes with prominent caustics, subsurface materials (skin, marble, candles).

### Real-time ray tracing (RTX/hardware accelerated)
**What it is**: Uses dedicated GPU hardware for ray-triangle intersection and BVH traversal.

**Advantages**:
- Interactive frame rates (30-60+ FPS)
- Combines with rasterization for hybrid rendering
- Hardware-accelerated BVH traversal

**Disadvantages**:
- Limited ray bounces due to performance constraints
- Requires modern RTX/RDNA2+ GPUs
- Still typically uses simplified lighting models

**Use cases**: Video games, interactive applications, real-time previews.

### Metropolis light transport (MLT)
**What it is**: Advanced Monte Carlo method using Markov Chain Monte Carlo (MCMC) to sample light paths.

**Advantages**:
- Excellent for difficult lighting scenarios
- Can find important light paths that path tracing misses
- Mathematically unbiased

**Disadvantages**:
- Very slow to converge
- Complex implementation
- Can exhibit temporal artifacts in animations

**Use cases**: Research, extremely challenging lighting scenarios.

### Instant radiosity
**What it is**: Approximates global illumination by placing virtual point lights at first-bounce intersections.

**Advantages**:
- Much faster than path tracing
- Provides global illumination effects
- Relatively simple to implement

**Disadvantages**:
- Approximation, not physically accurate
- Artifacts from point light representation
- Struggles with glossy surfaces

**Use cases**: Real-time global illumination approximation.

## Hybrid approaches

Many modern renderers combine multiple techniques:
- **Primary visibility**: Ray tracing for reflections, rasterization for primary surfaces
- **Lighting**: Path tracing for global illumination, cached for performance  
- **Materials**: Different algorithms for different material types
- **Level of detail**: Simpler methods for distant objects

## Performance characteristics

| Method | Speed | Accuracy | Implementation | Memory |
|--------|-------|----------|----------------|---------|
| Whitted | Fast | Low | Simple | Low |
| Path tracing | Slow | High | Medium | Low |
| Bidirectional | Slower | High | Hard | Medium |
| Photon mapping | Medium | High | Hard | High |
| RTX real-time | Fast | Medium | Medium | Low |
| MLT | Very slow | High | Very hard | Low |

## Future directions for this project

This project could evolve to support multiple ray tracing methods:

1. **Path tracing module**: Add Monte Carlo integration for global illumination
2. **GPU acceleration**: Port kernels to CUDA/OpenCL/compute shaders  
3. **Hybrid rendering**: Combine Whitted + path tracing for best of both
4. **Material improvements**: PBR materials, subsurface scattering
5. **Performance**: BVH acceleration, spatial data structures
6. **Advanced features**: Motion blur, depth of field, volumetrics

The current Whitted implementation provides an excellent foundation - it has all the core ray-geometry intersection code, matrix math, and scene management that other methods build upon.