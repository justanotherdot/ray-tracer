use crate::naive_cmp::naive_approx_equal_float;
use std::ops::{Index, IndexMut};

// These really ought to be enums and probably mean we don't need `w`, but we'll see how it pans
// out in the book.
#[allow(dead_code)]
const POINT_MAGIC: f64 = 1.0;
#[allow(dead_code)]
const VECTOR_MAGIC: f64 = 0.0;

#[allow(dead_code)]
#[derive(Debug, Copy, Clone)]
pub struct Point {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

#[allow(dead_code)]
impl Point {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Point {
            x,
            y,
            z,
            w: POINT_MAGIC,
        }
    }

    pub fn len(&self) -> usize {
        4
    }

    pub fn mul_f64(&self, rhs: f64) -> Self {
        mul_point_f64(self, rhs)
    }
}

impl std::cmp::PartialEq<Vector> for Point {
    fn eq(&self, other: &Vector) -> bool {
        naive_approx_equal_float(&self.x, &other.x)
            && naive_approx_equal_float(&self.y, &other.y)
            && naive_approx_equal_float(&self.z, &other.z)
            && naive_approx_equal_float(&self.w, &other.w)
    }
}

impl std::cmp::PartialEq<Point> for Vector {
    fn eq(&self, other: &Point) -> bool {
        naive_approx_equal_float(&self.x, &other.x)
            && naive_approx_equal_float(&self.y, &other.y)
            && naive_approx_equal_float(&self.z, &other.z)
            && naive_approx_equal_float(&self.w, &other.w)
    }
}

impl PartialEq for Point {
    fn eq(&self, other: &Self) -> bool {
        naive_approx_equal_float(&self.x, &other.x)
            && naive_approx_equal_float(&self.y, &other.y)
            && naive_approx_equal_float(&self.z, &other.z)
            && naive_approx_equal_float(&self.w, &other.w)
    }
}

#[allow(dead_code)]
#[derive(Debug, Copy, Clone)]
pub struct Vector {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

fn magnitude(v: &Vector) -> f64 {
    // TODO Ought this not include `w`?
    let sum_of_squares = v.x.powf(2.) + v.y.powf(2.) + v.z.powf(2.) + v.w.powf(2.);
    sum_of_squares.sqrt()
}

fn normalize(v: &Vector) -> Vector {
    let mag = v.magnitude();
    Vector {
        x: v.x / mag,
        y: v.y / mag,
        z: v.z / mag,
        w: v.w / mag,
    }
}

fn dot(a: &Vector, b: &Vector) -> f64 {
    a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w
}

fn cross(a: &Vector, b: &Vector) -> Vector {
    Vector::new(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )
}

// TODO A macro could easily impl Vector and Point here.
impl Index<usize> for Vector {
    type Output = f64;

    fn index(&self, ix: usize) -> &f64 {
        match ix {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            3 => &self.w,
            _ => panic!("index out of bounds: the len is 4 but the index is {}",
                ix
            ),
        }
    }
}

// TODO A macro could easily impl Vector and Point here.
impl IndexMut<usize> for Vector {
    fn index_mut(&mut self, ix: usize) -> &mut f64 {
        match ix {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            3 => &mut self.w,
            _ => panic!("index out of bounds: the len is 4 but the index is {}",
                ix
            ),
        }
    }
}

// TODO A macro could easily impl Vector and Point here.
impl Index<usize> for Point {
    type Output = f64;

    fn index(&self, ix: usize) -> &f64 {
        match ix {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            3 => &self.w,
            _ => panic!(
                "index out of bounds: the len is 4 but the index is {}",
                ix
            ),
        }
    }
}

// TODO A macro could easily impl Vector and Point here.
impl IndexMut<usize> for Point {
    fn index_mut(&mut self, ix: usize) -> &mut f64 {
        match ix {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            3 => &mut self.w,
            _ => panic!(
                "index out of bounds: the len is 4 but the index is {}",
                ix
            ),
        }
    }
}

pub fn reflect(incidence: &Vector, normal: &Vector) -> Vector {
    incidence.sub(&(normal.mul_f64(2.) * incidence.dot(&normal)))
}

#[allow(dead_code)]
impl Vector {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Vector {
            x,
            y,
            z,
            w: VECTOR_MAGIC,
        }
    }

    pub fn reflect(&self, normal: &Vector) -> Vector {
        reflect(self, normal)
    }

    pub fn magnitude(&self) -> f64 {
        magnitude(self)
    }

    pub fn normalize(&self) -> Self {
        normalize(self)
    }

    pub fn dot(&self, rhs: &Self) -> f64 {
        dot(self, rhs)
    }

    pub fn cross(&self, rhs: &Self) -> Self {
        cross(self, rhs)
    }

    pub fn len(&self) -> usize {
        4
    }

    pub fn sub(&self, rhs: &Self) -> Self {
        sub_vector_by_ref(self, rhs)
    }

    pub fn mul_f64(&self, rhs: f64) -> Self {
        mul_vec_f64_by_ref(self, rhs)
    }
}

impl PartialEq for Vector {
    fn eq(&self, other: &Self) -> bool {
        naive_approx_equal_float(&self.x, &other.x)
            && naive_approx_equal_float(&self.y, &other.y)
            && naive_approx_equal_float(&self.z, &other.z)
            && naive_approx_equal_float(&self.w, &other.w)
    }
}

impl std::ops::Add<Vector> for Point {
    type Output = Point;

    fn add(self, rhs: Vector) -> Point {
        Point {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
            w: self.w + rhs.w,
        }
    }
}

impl std::ops::Add<Point> for Vector {
    type Output = Point;

    fn add(self, rhs: Point) -> Point {
        Point {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
            w: self.w + rhs.w,
        }
    }
}

impl std::ops::Add for Vector {
    type Output = Vector;

    fn add(self, rhs: Vector) -> Vector {
        Vector {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
            w: self.w + rhs.w,
        }
    }
}

pub fn sub_point_by_ref(lhs: &Point, rhs: &Point) -> Vector {
    Vector {
        x: lhs.x - rhs.x,
        y: lhs.y - rhs.y,
        z: lhs.z - rhs.z,
        w: lhs.w - rhs.w,
    }
}

impl std::ops::Sub for Point {
    type Output = Vector;

    fn sub(self, rhs: Point) -> Self::Output {
        sub_point_by_ref(&self, &rhs)
    }
}

impl std::ops::Sub<Vector> for Point {
    type Output = Vector;

    fn sub(self, rhs: Vector) -> Self::Output {
        Vector {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
            w: self.w - rhs.w,
        }
    }
}

pub fn sub_vector_by_ref(lhs: &Vector, rhs: &Vector) -> Vector {
    Vector {
        x: lhs.x - rhs.x,
        y: lhs.y - rhs.y,
        z: lhs.z - rhs.z,
        w: lhs.w - rhs.w,
    }
}

impl std::ops::Sub for Vector {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        sub_vector_by_ref(&self, &rhs)
    }
}

impl std::ops::Neg for Point {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: -self.w,
        }
    }
}

pub fn mul_vec_f64_by_ref(lhs: &Vector, rhs: f64) -> Vector {
    Vector {
        x: lhs.x * rhs,
        y: lhs.y * rhs,
        z: lhs.z * rhs,
        w: lhs.w * rhs,
    }
}

impl std::ops::Mul<f64> for Vector {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        mul_vec_f64_by_ref(&self, rhs)
    }
}

impl std::ops::Mul<Vector> for f64 {
    type Output = Vector;

    fn mul(self, rhs: Vector) -> Self::Output {
        mul_vec_f64_by_ref(&rhs, self)
    }
}

pub fn mul_point_f64(lhs: &Point, rhs: f64) -> Point {
    Point {
        x: lhs.x * rhs,
        y: lhs.y * rhs,
        z: lhs.z * rhs,
        w: lhs.w * rhs,
    }
}

impl std::ops::Mul<f64> for Point {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        mul_point_f64(&self, rhs)
    }
}

impl std::ops::Mul<Point> for f64 {
    type Output = Point;

    fn mul(self, rhs: Point) -> Self::Output {
        Point {
            x: rhs.x * self,
            y: rhs.y * self,
            z: rhs.z * self,
            w: rhs.w * self,
        }
    }
}

impl std::ops::Div<Vector> for f64 {
    type Output = Vector;

    fn div(self, rhs: Vector) -> Self::Output {
        Vector {
            x: rhs.x / self,
            y: rhs.y / self,
            z: rhs.z / self,
            w: rhs.w / self,
        }
    }
}

impl std::ops::Div<f64> for Vector {
    type Output = Vector;

    fn div(self, rhs: f64) -> Self::Output {
        Vector {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
            w: self.w / rhs,
        }
    }
}

impl std::ops::Div<Point> for f64 {
    type Output = Point;

    fn div(self, rhs: Point) -> Self::Output {
        Point {
            x: rhs.x / self,
            y: rhs.y / self,
            z: rhs.z / self,
            w: rhs.w / self,
        }
    }
}

impl std::ops::Div<f64> for Point {
    type Output = Point;

    fn div(self, rhs: f64) -> Self::Output {
        Point {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
            w: self.w / rhs,
        }
    }
}

impl std::ops::Neg for Vector {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: -self.w,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn points_are_the_right_shape() {
        let a = Point::new(4.3, -4.2, 3.1);
        assert!(a.x == 4.3);
        assert!(a.y == -4.2);
        assert!(a.z == 3.1);
        assert!(a.w == POINT_MAGIC);
    }

    #[test]
    fn vectors_are_the_right_shape() {
        let a = Vector::new(4.3, -4.2, 3.1);
        assert!(a.x == 4.3);
        assert!(a.y == -4.2);
        assert!(a.z == 3.1);
        assert!(a.w == VECTOR_MAGIC);
    }

    #[test]
    fn add_vectors_and_points() {
        let a1 = Point::new(3., -2., 5.);
        let a2 = Vector::new(-2., 3., 1.0);
        let a3 = a1 + a2;

        assert_eq!(&a3, &Point::new(1., 1., 6.));
        assert_eq!(&Point::new(1., 1., 6.), &a3);
    }

    #[test]
    fn add_vectors() {
        let a1 = Vector::new(3., -2., 5.);
        let a2 = Vector::new(-2., 3., 1.0);
        let a3 = a1 + a2;

        assert_eq!(&a3, &Vector::new(1., 1., 6.));
        assert_eq!(&Vector::new(1., 1., 6.), &a3);
    }

    #[test]
    fn subtracting_two_points() {
        let a1 = Point::new(3., 2., 1.);
        let a2 = Point::new(5., 6., 7.);
        let a3 = a1 - a2;

        assert_eq!(&a3, &Vector::new(-2., -4., -6.));
    }

    #[test]
    fn subtracting_a_vector_from_a_point() {
        let a1 = Point::new(3., 2., 1.);
        let a2 = Vector::new(5., 6., 7.);
        let a3 = a1 - a2;

        assert_eq!(&a3, &Point::new(-2., -4., -6.));
    }

    #[test]
    fn subtracting_two_vectors() {
        let a1 = Vector::new(3., 2., 1.);
        let a2 = Vector::new(5., 6., 7.);
        let a3 = a1 - a2;

        assert_eq!(&a3, &Vector::new(-2., -4., -6.));
    }

    #[test]
    fn subtracting_a_vector_from_the_zero_vector() {
        let zero = Vector::new(0., 0., 0.);
        let v1 = Vector::new(1., -2., 3.);
        let v2 = zero - v1;

        assert_eq!(&v2, &Vector::new(-1., 2., -3.));
    }

    #[test]
    fn negating_a_point() {
        let p = Point::new(1., -2., 3.);
        // TODO It's worth considering if `new` on Point and Vector is worth it
        // if we're doing things like this here.
        let expected = Point {
            x: -1.,
            y: 2.,
            z: -3.,
            w: -1.,
        };

        // The test in the book shows w = 4 and represents this test with tuples but since we have
        // two specific structs we simply show the negated Point.
        assert_eq!(&(-p), &expected);
    }

    #[test]
    fn negating_a_vector() {
        let v = Vector::new(1., -2., 3.);
        // TODO It's worth considering if `new` on Vector and Vector is worth it
        // if we're doing things like this here.
        let expected = Vector {
            x: -1.,
            y: 2.,
            z: -3.,
            w: -0.,
        };

        // The test in the book shows w = 4 and represents this test with tuples but since we have
        // two specific structs we simply show the negated Point.
        assert_eq!(&(-v), &expected);
    }

    #[test]
    fn multiplying_a_vector_by_a_scalar() {
        let v = Vector::new(1., -2., 3.);
        let expected = Vector {
            x: 3.5,
            y: -7.,
            z: 10.5,
            w: 0.,
        };

        // Again, we might have a w here that abuses 1 or 0
        // so it may make sense to have assertions that panic
        // so we can catch bugs early.
        //
        // Technically this will only effect Point, as
        // x * 0 = 0, but x * 1 = x
        assert_eq!(&(v * 3.5), &expected);
    }

    #[test]
    fn multiplying_a_vector_by_a_fraction() {
        let v = Vector::new(1., -2., 3.);
        let expected = Vector {
            x: 0.5,
            y: -1.,
            z: 1.5,
            w: 0.,
        };

        // Again, we might have a w here that abuses 1 or 0
        // so it may make sense to have assertions that panic
        // so we can catch bugs early.
        //
        // Technically this will only effect Point, as
        // x * 0 = 0, but x * 1 = x
        assert_eq!(&(v * 0.5), &expected);
    }

    #[test]
    fn multiplying_a_vector_by_a_scalar_identity() {
        let v = Vector::new(1., -2., 3.);
        let expected = Vector {
            x: 1.,
            y: -2.,
            z: 3.,
            w: 0.,
        };

        // Again, we might have a w here that abuses 1 or 0
        // so it may make sense to have assertions that panic
        // so we can catch bugs early.
        //
        // Technically this will only effect Point, as
        // x * 0 = 0, but x * 1 = x
        assert_eq!(&(v * 1.), &expected);
    }

    #[test]
    fn multiplying_a_point_by_a_scalar() {
        let p = Point::new(1., -2., 3.);
        let expected = Point {
            x: 3.5,
            y: -7.,
            z: 10.5,
            w: 3.5,
        };

        // Again, we might have a w here that abuses 1 or 0
        // so it may make sense to have assertions that panic
        // so we can catch bugs early.
        //
        // Technically this will only effect Point, as
        // x * 0 = 0, but x * 1 = x
        assert_eq!(&(p * 3.5), &expected);
    }

    #[test]
    fn multiplying_a_point_by_a_fraction() {
        let p = Point::new(1., -2., 3.);
        let expected = Point {
            x: 0.5,
            y: -1.,
            z: 1.5,
            w: 0.5,
        };

        // Again, we might have a w here that abuses 1 or 0
        // so it may make sense to have assertions that panic
        // so we can catch bugs early.
        //
        // Technically this will only effect Point, as
        // x * 0 = 0, but x * 1 = x
        assert_eq!(&(p * 0.5), &expected);
    }

    #[test]
    fn multiplying_a_point_by_a_scalar_identity() {
        let p = Point::new(1., -2., 3.);
        let expected = Point {
            x: 3.5,
            y: -7.,
            z: 10.5,
            w: 3.5,
        };

        // Again, we might have a w here that abuses 1 or 0
        // so it may make sense to have assertions that panic
        // so we can catch bugs early.
        //
        // Technically this will only effect Point, as
        // x * 0 = 0, but x * 1 = x
        assert_eq!(&(p * 3.5), &expected);
    }

    #[test]
    fn dividing_a_point_by_a_scalar() {
        let p = Point::new(1., -2., 3.);
        let expected = Point {
            x: 0.5,
            y: -1.,
            z: 1.5,
            w: 0.5,
        };

        // Again, we might have a w here that abuses 1 or 0
        // so it may make sense to have assertions that panic
        // so we can catch bugs early.
        //
        // Technically this will only effect Point, as
        // x * 0 = 0, but x * 1 = x
        assert_eq!(&(p / 2.), &expected);
    }

    #[test]
    fn dividing_a_vector_by_a_scalar() {
        let v = Vector::new(1., -2., 3.);
        let expected = Vector {
            x: 0.5,
            y: -1.,
            z: 1.5,
            w: 0.0,
        };

        // Again, we might have a w here that abuses 1 or 0
        // so it may make sense to have assertions that panic
        // so we can catch bugs early.
        //
        // Technically this will only effect Point, as
        // x * 0 = 0, but x * 1 = x
        assert_eq!(&(v / 2.), &expected);
    }

    #[test]
    fn computing_the_magnitude_of_unit_vector_01() {
        let v = Vector::new(1., 0., 0.);
        assert_eq!(v.magnitude(), 1.);
    }

    #[test]
    fn computing_the_magnitude_of_unit_vector_02() {
        let v = Vector::new(0., 1., 0.);
        assert_eq!(v.magnitude(), 1.);
    }

    #[test]
    fn computing_the_magnitude_of_unit_vector_03() {
        let v = Vector::new(0., 0., 1.);
        assert_eq!(v.magnitude(), 1.);
    }

    #[test]
    fn computing_the_magnitude_of_unit_vector_04() {
        let v = Vector::new(1., 2., 3.);
        assert_eq!(v.magnitude(), f64::sqrt(14.0));
    }

    #[test]
    fn computing_the_magnitude_of_unit_vector_05() {
        let v = Vector::new(-1., -2., -3.);
        assert_eq!(v.magnitude(), f64::sqrt(14.0));
    }

    #[test]
    fn normalizing_vector_01() {
        let v = Vector::new(4.0, 0.0, 0.0);
        let expected = Vector::new(1.0, 0.0, 0.0);
        assert_eq!(v.normalize(), expected);
    }

    #[test]
    fn normalizing_vector_02() {
        let v = Vector::new(1.0, 2.0, 3.0);
        let expected = Vector::new(0.26726, 0.53452, 0.80178);
        assert_eq!(v.normalize(), expected);
    }

    #[test]
    fn normalizing_vector_03() {
        let v = Vector::new(1.0, 2.0, 3.0);
        assert_eq!(v.normalize().magnitude(), 1.0);
    }

    #[test]
    fn dot_product_of_two_points() {
        let a = Vector::new(1.0, 2.0, 3.0);
        let b = &Vector::new(2.0, 3.0, 4.0);
        assert_eq!(a.dot(b), 20.0);
    }

    #[test]
    fn cross_product_of_two_vectors() {
        let a = Vector::new(1.0, 2.0, 3.0);
        let b = Vector::new(2.0, 3.0, 4.0);
        let exp1 = Vector::new(-1.0, 2.0, -1.0);
        let exp2 = Vector::new(1.0, -2.0, 1.0);
        assert_eq!(a.cross(&b), exp1);
        assert_eq!(b.cross(&a), exp2);
    }

    #[test]
    fn vectors_can_be_indexed() {
        let a = Vector::new(1.0, 2.0, 3.0);
        assert_eq!(a[0], 1.0);
        assert_eq!(a[1], 2.0);
        assert_eq!(a[2], 3.0);
        assert_eq!(a[3], 0.0);
    }

    #[test]
    fn points_can_be_indexed() {
        let a = Point::new(1.0, 2.0, 3.0);
        assert_eq!(a[0], 1.0);
        assert_eq!(a[1], 2.0);
        assert_eq!(a[2], 3.0);
        assert_eq!(a[3], 1.0);
    }

    #[test]
    fn points_have_a_length() {
        let a = Point::new(1.0, 2.0, 3.0);
        assert_eq!(a.len(), 4);
    }

    #[test]
    fn vectors_have_a_length() {
        let a = Vector::new(1.0, 2.0, 3.0);
        assert_eq!(a.len(), 4);
    }
}
