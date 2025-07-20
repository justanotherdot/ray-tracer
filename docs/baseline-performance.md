# Baseline performance metrics

Established baseline performance for tracking optimization progress.

## Benchmark environment
- **Rust**: 1.88.0 (6b00bc388 2025-06-23) 
- **Platform**: macOS 15.5 (Darwin 24.5.0)
- **Hardware**: MacBook Pro (Mac15,6) with 36 GB memory, ARM64 (Apple Silicon)
- **Compiler flags**: Release mode with LTO enabled, debug symbols retained
- **Date**: 2025-07-21

## Baseline measurements (before optimizations)

### Core operations
- **Ray-sphere intersection**: 1.06 μs per operation
- **World intersect** (2 spheres): 2.27 μs per operation  
- **Matrix inverse**: 926 ns per operation
- **Vector dot product**: 869 ps per operation
- **Vector normalize**: 885 ps per operation
- **Vector cross product**: 1.58 ns per operation

### Rendering performance
- **50x50 render**: 10.25 ms (2,500 pixels)
- **Per-pixel cost**: ~4.1 μs per pixel

### Full scene rendering
- **World demo (200x100)**: ~1.6 seconds (20,000 pixels)
- **Per-pixel cost (full scene)**: ~80 μs per pixel (much higher due to 6-sphere scene complexity)

## Performance analysis

The significant difference between simple render (4.1μs/pixel) and full scene (80μs/pixel) indicates:
- **Scene complexity matters**: 6 spheres vs 2 spheres = ~20x slower per pixel
- **Intersection cost scales**: More objects = more intersection tests per ray
- **SmallVec overhead**: Likely allocating/reallocating for intersection collections

## Optimization targets

Based on measurements:
1. **SmallVec → Arrays**: Your confirmed substantial gain should improve world intersect
2. **Matrix inverse caching**: 926ns per inverse, likely called frequently during rendering
3. **Intersection sorting**: Part of the 2.27μs world intersect time
4. **Shadow ray optimization**: Each pixel may cast shadow rays to all lights

## Tracking progress

Run benchmarks after each optimization:
```bash
cargo bench --bench simple_benchmark
time cargo run --release --bin world_demo
```

## Historical tracking

| Date | Change | Ray-Sphere | World Intersect | Render 50x50 | World Demo | Notes |
|------|---------|------------|-----------------|--------------|------------|-------|
| 2025-07-21 | Baseline (SmallVec) | 1.06μs | 2.27μs | 10.25ms | ~1.6s | Initial measurements |
| | | | | | | |

Update this table after each optimization to track progress.