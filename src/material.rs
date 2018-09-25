use cgmath::*;

#[derive(Clone)]
pub struct RefractionHelper {
  pub incident_index:   f32,
  pub transmit_index:   f32,
}

impl RefractionHelper {
  pub fn calculate_transmit_dir(&self, incident_dir: Vector3<f32>, normal_dir: Vector3<f32>) -> Option<Vector3<f32>> {
    let i_idx = self.incident_index;
    let t_idx = self.transmit_index;
    let ratio = i_idx / t_idx;
    let cos_inc = incident_dir.dot(normal_dir);
    /*let sin_inc = cos_inc.acos().sin();*/
    let sin_inc = (1.0 - cos_inc * cos_inc).sqrt().min(1.0);
    let sin_tr = ratio * sin_inc;
    if sin_tr > 1.0 {
      // Total internal reflection.
      None
    } else {
      // Vector form of Snell's law.
      // FIXME: can use a more efficient formula w/out crosses and dots:
      // `t = (eta_1 / eta_2) * -i + ((eta_1 / eta_2) * cos(theta_i) - cos(theta_t)) * n`
      /*let i_cross_n = incident_dir.cross(normal_dir);
      let transmit_dir = (i_idx / t_idx) * (normal_dir.cross(i_cross_n) - normal_dir * (1.0 - (i_idx / t_idx) * (i_idx / t_idx) * i_cross_n.dot(i_cross_n)).max(0.0).sqrt());*/
      let transmit_dir = ratio * -incident_dir + (ratio * cos_inc - (1.0 - sin_tr * sin_tr).sqrt().min(1.0)) * normal_dir;
      let transmit_dir = transmit_dir.normalize();
      Some(transmit_dir)
    }
  }

  pub fn calculate_transmit_frac(&self, incident_dir: Vector3<f32>, transmit_dir: Vector3<f32>, normal_dir: Vector3<f32>) -> f32 {
    // Fresnel's laws for dielectric interfaces.
    let i_idx = self.incident_index;
    let t_idx = self.transmit_index;
    let i_dot_n = incident_dir.dot(normal_dir);
    let t_dot_n = transmit_dir.dot(normal_dir);
    let par_coef = (t_idx * i_dot_n + i_idx * t_dot_n) / (t_idx * i_dot_n - i_idx * t_dot_n);
    let orth_coef = (i_idx * i_dot_n + t_idx * t_dot_n) / (i_idx * i_dot_n - t_idx * t_dot_n);
    let reflect_coef = 0.5 * (par_coef * par_coef + orth_coef * orth_coef);
    let transmit_coef = (1.0 - reflect_coef).max(0.0).min(1.0);
    transmit_coef
  }
}

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
