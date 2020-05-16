use criterion::{black_box, criterion_group, criterion_main, Criterion};

use ray_tracer::matrix::Matrix;

// Order of exploration:
//   1. inverse
//   2. determinant
//   3. cofactor
//   4. minor

fn criterion_benchmark(c: &mut Criterion) {
    #[rustfmt::skip]
    let m: Matrix = Matrix::from_vec(vec![
        -2., 1., 2., 3.,
        3., 2., 1., -1.,
        4., 3., 6., 5.,
        1., 2., 7., 8.,
    ]);
    c.bench_function("empty", |b| b.iter(|| black_box(Matrix::empty(4, 4))));
    c.bench_function("dim", |b| b.iter(|| black_box(m.dim())));
    c.bench_function("transpose", |b| b.iter(|| black_box(m.transpose())));
    c.bench_function("determinant", |b| b.iter(|| black_box(m.determinant())));
    c.bench_function("submatrix", |b| b.iter(|| black_box(m.submatrix(2, 3))));
    c.bench_function("minor", |b| b.iter(|| black_box(m.minor(2, 3))));
    c.bench_function("cofactor", |b| b.iter(|| black_box(m.cofactor(2, 3))));
    c.bench_function("inverse", |b| b.iter(|| black_box(m.inverse())));
    c.bench_function("is_invertible", |b| b.iter(|| black_box(m.is_invertible())));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
