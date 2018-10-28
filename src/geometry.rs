use cgmath::*;

pub type Vector = Vector3<f32>;

#[derive(Clone, Copy)]
pub struct Ray {
  pub origin:   Vector,
  pub dir:      Vector,
}
