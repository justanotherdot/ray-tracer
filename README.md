# Ray Tracer

> Multi-language implementations of ray tracing algorithms.

A collection of ray tracer implementations across different programming languages, currently featuring a **Whitted-style ray tracer** in Rust. This classic ray tracing approach handles recursive reflections, refractions, and sharp shadows with predictable performance.

For an overview of different ray tracing approaches and how they compare, see [docs/ray-tracing-variants.md](docs/ray-tracing-variants.md).

![CI](https://github.com/justanotherdot/ray-tracer/workflows/CI/badge.svg)

## Language Implementations

### Rust (`rust/`)
- **Status**: Active development
- **Type**: Whitted-style ray tracer following "The Ray Tracer Challenge"
- **Features**: Spheres, Phong lighting, shadows, reflections, transformations
- **Performance**: Optimized with arrays, SIMD-ready math operations
- **Documentation**: Comprehensive docs in `docs/` covering architecture and optimization

## Project Structure

```
ray-tracer/
├── rust/           # Rust implementation
├── docs/           # Shared documentation
│   ├── ray-tracing-variants.md    # Overview of ray tracing approaches
│   ├── math-foundations.md        # Core mathematical concepts
│   ├── core-api.md               # Essential 100-line ray tracer
│   └── ...
└── README.md       # This file
```

Future language implementations will follow the same pattern with their own subdirectories.
