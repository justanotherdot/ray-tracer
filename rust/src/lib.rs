// These really ought to be enums and probably mean we don't need `w`, but we'll see how it pans
// out in the book.
#[allow(dead_code)]
const POINT_MAGIC: f64 = 1.0;
#[allow(dead_code)]
const VECTOR_MAGIC: f64 = 0.0;

#[allow(dead_code)]
#[derive(Debug)]
pub struct Point {
    x: f64,
    y: f64,
    z: f64,
    w: f64,
}

#[allow(dead_code)]
impl Point {
    fn new(x: f64, y: f64, z: f64) -> Self {
        Point {
            x,
            y,
            z,
            w: POINT_MAGIC,
        }
    }
}

impl std::cmp::PartialEq<Vector> for Point {
    fn eq(&self, other: &Vector) -> bool {
        naive_approx_equal_float(self.x, other.x)
            && naive_approx_equal_float(self.y, other.y)
            && naive_approx_equal_float(self.z, other.z)
            && naive_approx_equal_float(self.w, other.w)
    }
}

impl std::cmp::PartialEq<Point> for Vector {
    fn eq(&self, other: &Point) -> bool {
        naive_approx_equal_float(self.x, other.x)
            && naive_approx_equal_float(self.y, other.y)
            && naive_approx_equal_float(self.z, other.z)
            && naive_approx_equal_float(self.w, other.w)
    }
}

impl PartialEq for Point {
    fn eq(&self, other: &Self) -> bool {
        naive_approx_equal_float(self.x, other.x)
            && naive_approx_equal_float(self.y, other.y)
            && naive_approx_equal_float(self.z, other.z)
            && naive_approx_equal_float(self.w, other.w)
    }
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct Vector {
    x: f64,
    y: f64,
    z: f64,
    w: f64,
}

#[allow(dead_code)]
impl Vector {
    fn new(x: f64, y: f64, z: f64) -> Self {
        Vector {
            x,
            y,
            z,
            w: VECTOR_MAGIC,
        }
    }

    fn magnitude(&self) -> f64 {
        // TODO Ought this not include `w`?
        let sum_of_squares = self.x.powf(2.) + self.y.powf(2.) + self.z.powf(2.) + self.w.powf(2.);
        sum_of_squares.sqrt()
    }
}

impl PartialEq for Vector {
    fn eq(&self, other: &Self) -> bool {
        naive_approx_equal_float(self.x, other.x)
            && naive_approx_equal_float(self.y, other.y)
            && naive_approx_equal_float(self.z, other.z)
            && naive_approx_equal_float(self.w, other.w)
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

impl std::ops::Sub for Point {
    type Output = Vector;

    fn sub(self, rhs: Point) -> Self::Output {
        Vector {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
            w: self.w - rhs.w,
        }
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

impl std::ops::Sub for Vector {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Vector {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
            w: self.w - rhs.w,
        }
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

impl std::ops::Mul<f64> for Vector {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Vector {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
            w: self.w * rhs,
        }
    }
}

impl std::ops::Mul<Vector> for f64 {
    type Output = Vector;

    fn mul(self, rhs: Vector) -> Self::Output {
        Vector {
            x: rhs.x * self,
            y: rhs.y * self,
            z: rhs.z * self,
            w: rhs.w * self,
        }
    }
}

impl std::ops::Mul<f64> for Point {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Point {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
            w: self.w * rhs,
        }
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

fn naive_approx_equal_float(x: f64, y: f64) -> bool {
    const F64_EPSILON: f64 = 0.00001;
    // TODO Needs checks for NaN and ±∞ etc.
    if x == std::f64::NAN && y == std::f64::NAN {
        return false;
    }

    (x - y).abs() < F64_EPSILON
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
    fn naive_approx_equal_float_works() {
        assert!(naive_approx_equal_float(0.15 + 0.15, 0.1 + 0.2));
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
}
