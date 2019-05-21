use crate::color::Color;
use crate::ppm;
use crate::ppm::Ppm;

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

    // Should we be consuming color here?
    pub fn write_pixel(&mut self, x: usize, y: usize, color: Color) {
        // TODO Bounds check.
        assert!(x < self.width);
        assert!(y < self.height);

        //     A 3x3 canvas
        //
        // logical  --  physical
        //
        // 2 x x x      0 1 2
        // 1 x x x      3 4 5
        // 0 x x x      6 7 8
        //   0 1 2

        // (y * width) + x

        self.pixels[(y * self.width) + x] = color;
    }

    pub fn pixel_at(&self, x: usize, y: usize) -> &Color {
        assert!(x < self.width);
        assert!(y < self.height);
        &self.pixels[(y * self.width) + x]
    }

    // It would probably be more helpful if this returned back
    // a pair of pixels and their respective coordinates
    // although one could always reconstruct this with the canvas
    // width and height, as per `crate::ppm::canvas_to_ppm`
    pub fn pixels(&self) -> std::slice::Iter<Color> {
        self.pixels.iter()
    }

    // TODO This could use some cleanup.
    pub fn pixel_rows(&self) -> Vec<Vec<&Color>> {
        let mut rv = vec![];
        for y in 0..self.height {
            let mut cv = vec![];
            for x in 0..self.width {
                cv.push(self.pixel_at(x, y));
            }
            rv.push(cv);
        }
        rv
    }

    pub fn to_ppm(self) -> Ppm {
        ppm::canvas_to_ppm(self)
    }

    pub fn fill(&mut self, c: Color) {
        // TODO There should be an iterator that produces all (x, y) pairs for a given width and
        // height. That way `pixels` could be augmented with each pixels location.
        for i in 0..self.width {
            for j in 0..self.height {
                self.write_pixel(i, j, c.clone());
            }
        }
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
        for pixel in c.pixels() {
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
