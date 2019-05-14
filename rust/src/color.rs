#[derive(Debug)]
pub struct Color {
    pub r: f64,
    pub g: f64,
    pub b: f64,
}

impl std::cmp::PartialEq for Color {
    fn eq(&self, other: &Color) -> bool {
        naive_approx_equal_float(self.r, other.r)
            && naive_approx_equal_float(self.g, other.g)
            && naive_approx_equal_float(self.b, other.b)
    }
}

// n.b.
// This approximate equality is needed for Color, too,
// In order to make the tests pass given in the book.
fn naive_approx_equal_float(x: f64, y: f64) -> bool {
    const F64_EPSILON: f64 = 0.00001;
    // TODO Needs checks for NaN and ±∞ etc.
    if x == std::f64::NAN && y == std::f64::NAN {
        return false;
    }

    (x - y).abs() < F64_EPSILON
}

impl Color {
    pub fn new(r: f64, g: f64, b: f64) -> Self {
        Color { r, g, b }
    }
}

impl std::ops::Add for Color {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let r = self.r + rhs.r;
        let g = self.g + rhs.g;
        let b = self.b + rhs.b;
        Color { r, g, b }
    }
}

impl std::ops::Sub for Color {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        let r = self.r - rhs.r;
        let g = self.g - rhs.g;
        let b = self.b - rhs.b;
        Color { r, g, b }
    }
}

impl std::ops::Mul<f64> for Color {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self {
        let r = self.r * rhs;
        let g = self.g * rhs;
        let b = self.b * rhs;
        Color { r, g, b }
    }
}

impl std::ops::Mul for Color {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        let r = self.r * rhs.r;
        let g = self.g * rhs.g;
        let b = self.b * rhs.b;
        Color { r, g, b }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn colors_have_red_green_blue_fields() {
        let c = Color::new(-0.5, 0.4, 1.7);
        assert_eq!(c.r, -0.5);
        assert_eq!(c.g, 0.4);
        assert_eq!(c.b, 1.7);
    }

    #[test]
    fn adding_colors() {
        let c1 = Color::new(0.9, 0.6, 0.75);
        let c2 = Color::new(0.7, 0.1, 0.25);
        assert_eq!(c1 + c2, Color::new(1.6, 0.7, 1.0));
    }

    #[test]
    fn subtracting_colors() {
        let c1 = Color::new(0.9, 0.6, 0.75);
        let c2 = Color::new(0.7, 0.1, 0.25);
        assert_eq!(c1 - c2, Color::new(0.2, 0.5, 0.5));
    }

    #[test]
    fn multiplying_a_color_by_a_scalar() {
        let c = Color::new(0.2, 0.3, 0.4);
        assert_eq!(c * 2., Color::new(0.4, 0.6, 0.8));
    }

    #[test]
    fn multiplying_colors() {
        let c1 = Color::new(1., 0.2, 0.4);
        let c2 = Color::new(0.9, 1., 0.1);
        assert_eq!(c1 * c2, Color::new(0.9, 0.2, 0.04));
    }

}
