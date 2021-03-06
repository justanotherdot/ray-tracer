use crate::color::{mul_color, Color};
use crate::coordinate::{sub_point_by_ref, Point, Vector};
use crate::ray::Ray;
use crate::world::World;
use std::default::Default;

#[derive(Debug, PartialEq, Clone)]
pub struct PointLight {
    position: Point,
    intensity: Color,
}

// TODO: Rename this as Light?
impl PointLight {
    pub fn new(position: Point, intensity: Color) -> Self {
        Self {
            position,
            intensity,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Material {
    pub color: Color,
    pub ambient: f64,
    pub diffuse: f64,
    pub specular: f64,
    pub shininess: f64,
}

impl Material {
    pub fn new() -> Self {
        let color = Color::new(1., 1., 1.);
        let ambient = 0.1;
        let diffuse = 0.9;
        let specular = 0.9;
        let shininess = 200.;
        Self {
            color,
            ambient,
            diffuse,
            specular,
            shininess,
        }
    }

    // TODO Don't just take a bool and change the functionality.
    pub fn lighting(
        &self,
        light: &PointLight,
        point: &Point,
        eyev: &Vector,
        normalv: &Vector,
        in_shadow: bool,
    ) -> Color {
        lighting(self, light, point, eyev, normalv, in_shadow)
    }
}

// This really doesn't buy us anything for now, but it's nice
// to see how simple it is to implement.
impl Default for Material {
    fn default() -> Self {
        Material::new()
    }
}

// TODO Might belong in `world`, instead.
// TODO turn this into Result of bool, as there may not be a light source.
pub fn is_shadowed(world: &World, point: &Point) -> bool {
    world
        .light
        .as_ref()
        .map(|light| {
            let v = sub_point_by_ref(&light.position, point);
            let distance = v.magnitude();
            let direction = v.normalize();
            let r = Ray::new(point.clone(), direction);
            let intersections = world.intersect(&r);
            let h = intersections.hit();
            match h {
                Some(h) => h.t < distance,
                _ => false,
            }
        })
        .unwrap_or(true)
}

pub fn lighting(
    material: &Material,
    light: &PointLight,
    point: &Point,
    eyev: &Vector,
    normalv: &Vector,
    in_shadow: bool,
) -> Color {
    let effective_color = mul_color(&material.color, &light.intensity);
    let lightv = sub_point_by_ref(&light.position, &point).normalize();
    let ambient = effective_color.clone() * material.ambient;
    let light_dot_normal = lightv.dot(&normalv);
    let black = Color::new(0., 0., 0.);
    let mut diffuse;
    let mut specular;
    if light_dot_normal < 0. {
        diffuse = black.clone();
        specular = black.clone();
    } else {
        diffuse = effective_color * material.diffuse * light_dot_normal;
        let reflectv = (-lightv).reflect(normalv);
        let reflect_dot_eye = reflectv.dot(&eyev);

        if reflect_dot_eye <= 0. {
            specular = black;
        } else {
            let factor = reflect_dot_eye.powf(material.shininess);
            specular = light.intensity.mul_f64(material.specular).mul_f64(factor);
        }
    }
    if in_shadow {
        specular = Color::new(0., 0., 0.);
        diffuse = Color::new(0., 0., 0.);
    }
    ambient + diffuse + specular
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::coordinate::{Point, Vector};
    use crate::ray::Sphere;
    use crate::transformation::Transformation;

    #[test]
    fn the_normal_on_a_sphere_at_a_point_on_the_x_axis() {
        let s = Sphere::new(0);
        let n = s.normal_at(Point::new(1., 0., 0.));
        assert_eq!(n, Vector::new(1., 0., 0.));
    }

    #[test]
    fn the_normal_on_a_sphere_at_a_point_on_the_y_axis() {
        let s = Sphere::new(0);
        let n = s.normal_at(Point::new(0., 1., 0.));
        assert_eq!(n, Vector::new(0., 1., 0.));
    }

    #[test]
    fn the_normal_on_a_sphere_at_a_point_on_the_z_axis() {
        let s = Sphere::new(0);
        let n = s.normal_at(Point::new(0., 0., 1.));
        assert_eq!(n, Vector::new(0., 0., 1.));
    }

    #[test]
    fn the_normal_on_a_sphere_at_a_nonaxial_point() {
        let s = Sphere::new(0);
        let n = s.normal_at(Point::new(
            (3.0 as f64).sqrt() / 3.,
            (3.0 as f64).sqrt() / 3.,
            (3.0 as f64).sqrt() / 3.,
        ));
        let v = Vector::new(
            (3.0 as f64).sqrt() / 3.,
            (3.0 as f64).sqrt() / 3.,
            (3.0 as f64).sqrt() / 3.,
        );
        assert_eq!(n, v);
    }

    #[test]
    fn the_normal_is_a_normalized_vector() {
        let s = Sphere::new(0);
        let n = s.normal_at(Point::new(
            (3.0 as f64).sqrt() / 3.,
            (3.0 as f64).sqrt() / 3.,
            (3.0 as f64).sqrt() / 3.,
        ));
        let v = Vector::new(
            (3.0 as f64).sqrt() / 3.,
            (3.0 as f64).sqrt() / 3.,
            (3.0 as f64).sqrt() / 3.,
        );
        assert_eq!(n, v.normalize());
    }

    #[test]
    fn computing_the_normal_on_a_translated_sphere() {
        let mut s = Sphere::new(0);
        s.set_transform(Transformation::new().translate(0., 1., 0.).build());
        let n = s.normal_at(Point::new(0., 1.70711, -0.70711));
        let v = Vector::new(0., 0.70711, -0.70711);
        assert_eq!(n, v);
    }

    #[test]
    fn computing_the_normal_on_a_transformed_sphere() {
        let mut s = Sphere::new(0);
        s.set_transform(
            Transformation::new()
                .rotate_z(std::f64::consts::PI / 5.)
                .scale(1., 0.5, 1.)
                .build(),
        );
        let n = s.normal_at(Point::new(
            0.,
            (2. as f64).sqrt() / 2.,
            -((2. as f64).sqrt() / 2.),
        ));
        let v = Vector::new(0., 0.97014, -0.24254);
        assert_eq!(n, v);
    }

    #[test]
    fn reflecting_a_vector_approaching_at_45_deg() {
        let v = Vector::new(1., -1., 0.);
        let n = Vector::new(0., 1., 0.);
        let r = v.reflect(&n);
        assert_eq!(r, Vector::new(1., 1., 0.));
    }

    #[test]
    fn reflecting_a_vector_off_a_slanted_surface() {
        let v = Vector::new(0., -1., 0.);
        let n = Vector::new((2 as f64).sqrt() / 2., (2 as f64).sqrt() / 2., 0.);
        let r = v.reflect(&n);
        assert_eq!(r, Vector::new(1., 0., 0.));
    }

    #[test]
    fn a_point_light_has_a_position_and_intensity() {
        let intensity = Color::new(1., 1., 1.);
        let position = Point::new(0., 0., 0.);
        let light = PointLight::new(position.clone(), intensity.clone());
        assert_eq!(light.position, position);
        assert_eq!(light.intensity, intensity);
    }

    #[test]
    fn the_default_material() {
        let m = Material::new();
        assert_eq!(m.color, Color::new(1., 1., 1.,));
        assert_eq!(m.ambient, 0.1);
        assert_eq!(m.diffuse, 0.9);
        assert_eq!(m.specular, 0.9);
        assert_eq!(m.shininess, 200.0);
    }

    #[test]
    fn a_sphere_has_a_default_material() {
        let s = Sphere::new(0);
        let m = s.material;
        assert_eq!(m, Material::new());
    }

    #[test]
    fn lighting_with_the_eye_between_the_light_and_the_surface() {
        // preamble.
        let m = Material::new();
        let position = Point::new(0., 0., 0.);

        let eyev = Vector::new(0., 0., -1.);
        let normalv = Vector::new(0., 0., -1.);
        let light = PointLight::new(Point::new(0., 0., -10.), Color::new(1., 1., 1.));
        let result = m.lighting(&light, &position, &eyev, &normalv, false);
        assert_eq!(result, Color::new(1.9, 1.9, 1.9));
    }

    #[test]
    fn lighting_with_the_eye_between_light_and_surface_eye_offset_45_deg() {
        // preamble.
        let m = Material::new();
        let position = Point::new(0., 0., 0.);

        let root_two_on_two = (2 as f64).sqrt() / 2.;
        let eyev = Vector::new(0., root_two_on_two, -root_two_on_two);
        let normalv = Vector::new(0., 0., -1.);
        let light = PointLight::new(Point::new(0., 0., -10.), Color::new(1., 1., 1.));
        let result = m.lighting(&light, &position, &eyev, &normalv, false);
        assert_eq!(result, Color::new(1.0, 1.0, 1.0));
    }

    #[test]
    fn lghting_with_eye_opposite_surface_light_offset_45_deg() {
        // preamble.
        let m = Material::new();
        let position = Point::new(0., 0., 0.);

        let eyev = Vector::new(0., 0., -1.);
        let normalv = Vector::new(0., 0., -1.);
        let light = PointLight::new(Point::new(0., 10., -10.), Color::new(1., 1., 1.));
        let result = m.lighting(&light, &position, &eyev, &normalv, false);
        assert_eq!(result, Color::new(0.7364, 0.7364, 0.7364));
    }

    #[test]
    fn lighting_with_eye_in_the_path_of_the_reflection_vector() {
        // preamble.
        let m = Material::new();
        let position = Point::new(0., 0., 0.);

        let root_two_on_two = (2 as f64).sqrt() / 2.;
        let eyev = Vector::new(0., -root_two_on_two, -root_two_on_two);
        let normalv = Vector::new(0., 0., -1.);
        let light = PointLight::new(Point::new(0., 10., -10.), Color::new(1., 1., 1.));
        let result = m.lighting(&light, &position, &eyev, &normalv, false);
        assert_eq!(result, Color::new(1.6364, 1.6364, 1.6364));
    }

    #[test]
    fn lighting_with_the_light_behind_the_surface() {
        // preamble.
        let m = Material::new();
        let position = Point::new(0., 0., 0.);

        let eyev = Vector::new(0., 0., -1.);
        let normalv = Vector::new(0., 0., -1.);
        let light = PointLight::new(Point::new(0., 0., 10.), Color::new(1., 1., 1.));
        let result = m.lighting(&light, &position, &eyev, &normalv, false);
        assert_eq!(result, Color::new(0.1, 0.1, 0.1));
    }

    #[test]
    fn lighting_with_the_surface_in_shadow() {
        // preamble.
        let m = Material::new();
        let position = Point::new(0., 0., 0.);

        let eyev = Vector::new(0., 0., -1.);
        let normalv = Vector::new(0., 0., -1.);
        let light = PointLight::new(Point::new(0., 0., -10.), Color::new(1., 1., 1.));
        let in_shadow = true;
        let result = m.lighting(&light, &position, &eyev, &normalv, in_shadow);
        assert_eq!(result, Color::new(0.1, 0.1, 0.1));
    }

    #[test]
    fn there_is_no_shadow_when_nothing_is_collinear_with_point_and_light() {
        let w: World = Default::default();
        let p = Point::new(0., 10., 0.);

        assert_eq!(is_shadowed(&w, &p), false);
    }

    #[test]
    fn the_shadow_when_an_object_is_between_the_point_and_the_light() {
        let w: World = Default::default();
        let p = Point::new(10., -10., 10.);

        assert_eq!(is_shadowed(&w, &p), true);
    }

    #[test]
    fn there_is_no_shadow_when_an_object_is_behind_the_light() {
        let w: World = Default::default();
        let p = Point::new(-20., 20., -20.);

        assert_eq!(is_shadowed(&w, &p), false);
    }

    #[test]
    fn there_is_no_shadow_when_an_object_is_behind_the_point() {
        let w: World = Default::default();
        let p = Point::new(-2., 2., -2.);

        assert_eq!(is_shadowed(&w, &p), false);
    }
}
