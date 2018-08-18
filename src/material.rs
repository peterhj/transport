use cgmath::*;

pub trait Potential {
  fn eval_potential(&self, world_pos: Vector3<f32>) -> f32;
  fn eval_adj_potential(&self, world_pos: Vector3<f32>) -> Vector3<f32>;
}

pub struct SpherePotential {
  pub center:   Vector3<f32>,
  pub radius:   f32,
}

impl SpherePotential {
  fn eval_potential(&self, world_pos: Vector3<f32>) -> f32 {
    let dx = world_pos - self.center;
    let r = self.radius;
    dx.dot(dx) - r * r
  }

  fn eval_adj_potential(&self, world_pos: Vector3<f32>) -> Vector3<f32> {
    let dx = world_pos - self.center;
    2.0 * dx
  }
}

pub trait Field {
  fn eval_field(&self, world_pos: Vector3<f32>, world_dir: Vector3<f32>, nfreqs: usize) -> Vec<f32>;
}

pub trait Density {
  fn eval_density(&self, world_pos: Vector3<f32>, nfreqs: usize) -> Vec<f32>;
}

pub struct ConstantField {
  pub constant: f32,
}

impl Field for ConstantField {
  fn eval_field(&self, world_pos: Vector3<f32>, world_dir: Vector3<f32>, nfreqs: usize) -> Vec<f32> {
    let mut f = Vec::with_capacity(nfreqs);
    for _ in 0 .. nfreqs {
      f.push(self.constant)
    }
    f
  }
}

pub struct BinaryPotentialField<P: Potential> {
  pub inside:       f32,
  pub outside:      f32,
  pub potential:    P,
}

impl<P: Potential> Field for BinaryPotentialField<P> {
  fn eval_field(&self, world_pos: Vector3<f32>, world_dir: Vector3<f32>, nfreqs: usize) -> Vec<f32> {
    let mut f = Vec::with_capacity(nfreqs);
    if self.potential.eval_potential(world_pos) <= 0.0 {
      for _ in 0 .. nfreqs {
        f.push(self.inside)
      }
    } else {
      for _ in 0 .. nfreqs {
        f.push(self.outside);
      }
    }
    f
  }
}

pub struct SmoothBinaryPotentialField<P: Potential> {
  pub inside:       f32,
  pub outside:      f32,
  pub scale:        f32,
  pub potential:    P,
}

impl<P: Potential> Field for SmoothBinaryPotentialField<P> {
  fn eval_field(&self, world_pos: Vector3<f32>, world_dir: Vector3<f32>, nfreqs: usize) -> Vec<f32> {
    let mut f = Vec::with_capacity(nfreqs);
    if self.potential.eval_potential(world_pos) <= 0.0 {
      for _ in 0 .. nfreqs {
        f.push(self.inside)
      }
    } else {
      for _ in 0 .. nfreqs {
        f.push(self.outside);
      }
    }
    f
  }
}
