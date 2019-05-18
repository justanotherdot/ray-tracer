use crate::canvas::Canvas;
use crate::color::Color;

pub struct Ppm(pub String);

impl Ppm {
    pub fn blob(self) -> String {
        self.0
    }
}

fn clamp(x: f64, lower_bound: f64, upper_bound: f64) -> f64 {
    f64::min(f64::max(x, lower_bound).round(), upper_bound)
}

fn clamp_color(c: Color, lower_bound: f64, upper_bound: f64) -> (f64, f64, f64) {
    let r = clamp(c.r, lower_bound, upper_bound);
    let g = clamp(c.g, lower_bound, upper_bound);
    let b = clamp(c.b, lower_bound, upper_bound);
    (r, g, b)
}

// TODO This could use a fair bit of refactoring.
// TODO Make this a method on Canvas, `to_ppm`.
pub fn canvas_to_ppm(canvas: Canvas) -> Ppm {
    // Header.
    // Would be initial string to fold.

    let mut lines = format!(
        "P3\n{width} {height}\n{scale}\n",
        width = canvas.width,
        height = canvas.height,
        scale = 255,
    );

    let len = canvas.pixels().len();
    let mut i = 0;
    let mut curr_ix = 0;
    let mut col_count = 0;
    // This is a _teensy_ bit savage.
    // TODO This needs to be a fold.
    canvas.pixels().for_each(|c| {
        let (r, g, b) = clamp_color(c.clone() * 255., 0., 255.);

        i += 1;

        for col in vec![r, g, b] {
            curr_ix += 1;

            let s = format!("{}", col);
            col_count += s.len();

            if col_count >= 70 || i > canvas.width {
                let k = lines.pop().unwrap();
                if k != ' ' {
                    lines.push(k);
                }

                lines.push_str("\n");
                lines.push_str(&s);

                col_count = s.len();

                if curr_ix < len * 3 {
                    col_count += 1;
                    lines.push_str(" ");
                }

                if i > canvas.width {
                    i = 0;
                }
            } else if curr_ix < len * 3 {
                col_count += 1;
                lines.push_str(&s);
                lines.push_str(" ");
            } else {
                lines.push_str(&s);
            }
        }
    });

    // PPM's are always terminated by a newline.
    lines.push_str("\n");

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
        let blob = ppm.blob();
        let actual: Vec<&str> = blob.lines().collect();
        let expected = vec!["P3", "5 3", "255"];
        assert_eq!(actual[..3], expected[..]);
    }

    #[test]
    fn contructing_the_ppm_pixel_data_01() {
        let mut c = Canvas::new(1, 1);
        c.write_pixel(0, 0, Color::new(1.5, 0., 0.));
        let ppm = canvas_to_ppm(c);
        let blob = ppm.blob();
        let lhs: Vec<&str> = blob.lines().collect();
        let rhs = vec!["255 0 0"];
        assert_eq!(lhs[3..], rhs[..]);
    }

    #[test]
    fn contructing_the_ppm_pixel_data_02() {
        let mut c = Canvas::new(1, 3);
        c.write_pixel(0, 0, Color::new(1.0, 0.0, 0.0));
        c.write_pixel(0, 1, Color::new(0.0, 1.0, 0.0));
        c.write_pixel(0, 2, Color::new(0.0, 0.0, 1.0));
        let ppm = canvas_to_ppm(c);
        let blob = ppm.blob();
        let lhs: Vec<&str> = blob.lines().collect();
        let rhs = vec!["255 0 0", "0 255 0", "0 0 255"];
        assert_eq!(lhs[3..], rhs[..]);
    }

    #[test]
    fn contructing_the_ppm_pixel_data() {
        let mut c = Canvas::new(5, 3);
        c.write_pixel(0, 0, Color::new(1.5, 0., 0.));
        c.write_pixel(2, 1, Color::new(0., 0.5, 0.));
        c.write_pixel(4, 2, Color::new(-0.5, 0., 1.));
        let ppm = canvas_to_ppm(c);
        let blob = ppm.blob();
        let lhs: Vec<&str> = blob.lines().collect();
        let rhs = vec![
            "255 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
            "0 0 0 0 0 0 0 128 0 0 0 0 0 0 0",
            "0 0 0 0 0 0 0 0 0 0 0 0 0 0 255",
        ];
        assert_eq!(lhs[3..6], rhs[..]);
    }

    #[test]
    fn splitting_long_lines_in_ppm_files() {
        let mut c = Canvas::new(10, 2);
        c.fill(Color::new(1.0, 0.8, 0.6));
        let ppm = canvas_to_ppm(c);
        let blob = ppm.blob();
        let lhs: Vec<&str> = blob.lines().collect();
        let rhs = vec![
            "255 204 153 255 204 153 255 204 153 255 204 153 255 204 153 255 204",
            "153 255 204 153 255 204 153 255 204 153 255 204 153",
            "255 204 153 255 204 153 255 204 153 255 204 153 255 204 153 255 204",
            "153 255 204 153 255 204 153 255 204 153 255 204 153",
        ];
        println!("{:#?}", &lhs[3..]);
        assert_eq!(lhs[3..], rhs[..]);
    }

    #[test]
    fn ppm_files_are_terminated_by_a_newline_character() {
        let c = Canvas::new(5, 3);
        let ppm = canvas_to_ppm(c);
        let last_char = ppm.blob().pop();
        assert_eq!(last_char, Some('\n'));
    }
}
