use crate::canvas::Canvas;

pub struct Ppm(pub String);

impl Ppm {
    pub fn blob(self) -> String {
        self.0
    }
}

pub fn canvas_to_ppm(canvas: Canvas) -> Ppm {
    let lines = format!(
        "P3\n{width} {height}\n{scale}",
        width = canvas.width,
        height = canvas.height,
        scale = 255,
    );

    Ppm(lines)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::color::Color;

    #[test]
    fn constructing_the_ppm_header() {
        let c = Canvas::new(5, 3);
        let ppm = canvas_to_ppm(c);
        let expected = "P3\n5 3\n255";
        assert_eq!(ppm.blob(), expected);
    }

    #[test]
    fn contructing_the_ppm_pixel_data() {
        let mut c = Canvas::new(5, 3);
        c.write_pixel(0, 0, Color::new(1.5, 0., 0.));
        c.write_pixel(2, 1, Color::new(0., 0.5, 0.));
        c.write_pixel(4, 2, Color::new(-0.5, 0., 1.));
        let ppm = canvas_to_ppm(c);
        let rhs = vec![
            "255 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
            "0 0 0 0 0 0 0 128 0 0 0 0 0 0 0",
            "0 0 0 0 0 0 0 0 0 0 0 0 0 0 255",
        ];
        let lhs: Vec<&str> = ppm.blob().lines().collect();
        assert_eq!(rhs[3..6], rhs[..]);
    }
}
