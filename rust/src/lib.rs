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

#[allow(dead_code)]
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
}
