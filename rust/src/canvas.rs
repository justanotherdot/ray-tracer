use crate::color::Color;

#[derive(Debug)]
pub struct Canvas {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<Color>, // TODO &'a [Color] as opposed to Vec.
}

impl Canvas {
    pub fn new(width: usize, height: usize) -> Self {
        let black = Color::new(0., 0., 0.);
        let pixels = vec![black; width * height];
        Canvas {
            width,
            height,
            pixels,
        }
    }

    // Should we consumer color here?
    pub fn write_pixel(&mut self, x: usize, y: usize, color: Color) {
        // TODO Bounds check.
        assert!(x < self.width - 1);
        assert!(y < self.height - 1);
        self.pixels[(x * y) + x] = color;
    }

    pub fn pixel_at(&self, x: usize, y: usize) -> &Color {
        &self.pixels[(x * y) + x]
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn create_a_canvas() {
        let c = Canvas::new(10, 20);
        let expected_pixel = &Color::new(0., 0., 0.);
        // TODO Canvas should have an iterator so as to
        // avoid people digging directly into `pixels`.
        for pixel in c.pixels.iter() {
            assert_eq!(pixel, expected_pixel);
        }
    }

    #[test]
    fn writing_pixels_to_a_canvas() {
        let mut c = Canvas::new(10, 20);
        let red = Color::new(1., 0., 0.);
        c.write_pixel(2, 3, red);
        assert_eq!(c.pixel_at(2, 3), &Color::new(1., 0., 0.));
    }
}
