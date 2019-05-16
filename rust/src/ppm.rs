use crate::canvas::Canvas;

// Probably ideal just to have this as a newtype-like struct.
pub struct Ppm {
    pub lines: Vec<String>,
}

pub fn canvas_to_ppm(canvas: Canvas) -> Ppm {
    let mut lines = vec![];
    lines.push("P3".to_string());
    lines.push(format!("{} {}", canvas.width, canvas.height));
    lines.push("255".to_string());

    Ppm { lines }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::color::Color;

    #[test]
    fn constructing_the_ppm_header() {
        let c = Canvas::new(5, 3);
        let ppm = canvas_to_ppm(c);
        let expected = vec!["P3", "5 3", "255"];
        assert_eq!(ppm.lines[..3], expected[..3]);
    }

    #[test]
    fn contructing_the_ppm_pixel_data() {
        let mut c = Canvas::new(5, 3);
        c.write_pixel(0, 0, Color::new(1.5, 0., 0.));
        c.write_pixel(2, 1, Color::new(0., 0.5, 0.));
        c.write_pixel(4, 2, Color::new(-0.5, 0., 1.));
        let ppm = canvas_to_ppm(c);
        let expected = vec![
            "255 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
            "0 0 0 0 0 0 0 128 0 0 0 0 0 0 0",
            "0 0 0 0 0 0 0 0 0 0 0 0 0 0 255",
        ];
        assert_eq!(ppm.lines[3..6], expected[3..6]);
    }
}
