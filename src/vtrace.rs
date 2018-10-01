use ::geometry::*;

use cgmath::*;
use float::ord::*;
use memarray::*;
use rand::prelude::*;
use rand::distributions::{Open01, OpenClosed01, Standard, StandardNormal, Uniform};

use std::f32::consts::{PI, FRAC_2_PI};
use std::rc::{Rc};

fn clip_radian(mut t: f32) -> f32 {
  while t >= PI {
    t -= PI;
  }
  while t < -PI {
    t += PI;
  }
  t
}

fn linear2srgb(linear: f32) -> f32 {
  // See: <https://github.com/nothings/stb/issues/588>.
  (if linear <= 0.0031308 {
    12.92 * linear
  } else {
    1.055 * linear.powf(1.0 / 2.4) - 0.055
  }).max(0.0).min(1.0)
}

#[derive(Clone, Copy)]
pub enum InterfaceEvent {
  Absorb,
  Reflect(Vector),
  Transmit(Vector),
}

pub trait InterfaceMat {
  fn scatter_surf_bwd(&self, inc_out_dir: Vector, inc_normal: Vector) -> (InterfaceEvent, Option<f32>);
}

pub fn interface_at(
    inc_surf: Rc<dyn SurfaceMat>,
    inc_vol: Rc<dyn VolumeMat>,
    ext_surf: Rc<dyn SurfaceMat>,
    ext_vol: Rc<dyn VolumeMat>,
    x: Vector)
-> Rc<dyn InterfaceMat>
{
  // TODO
  match (inc_surf.surf_mat_kind(), inc_vol.vol_mat_kind(), ext_surf.surf_mat_kind(), ext_vol.vol_mat_kind()) {
    (SurfaceMatKind::PassThrough, _, SurfaceMatKind::Lambertian, _) => {
      // TODO
      Rc::new(LambertianInterfaceMatDef{
        absorb_prob:    0.0,
      })
    }
    (SurfaceMatKind::Lambertian, _, SurfaceMatKind::PassThrough, _) => {
      // TODO
      Rc::new(LambertianInterfaceMatDef{
        absorb_prob:    0.0,
      })
    }
    (SurfaceMatKind::PassThrough, _, SurfaceMatKind::Mirror, _) => {
      // TODO
      Rc::new(MirrorInterfaceMatDef{
        absorb_prob:    0.0,
      })
    }
    (SurfaceMatKind::Mirror, _, SurfaceMatKind::PassThrough, _) => {
      // TODO
      Rc::new(MirrorInterfaceMatDef{
        absorb_prob:    0.0,
      })
    }
    (SurfaceMatKind::PassThrough, VolumeMatKind::Dielectric, SurfaceMatKind::PassThrough, VolumeMatKind::Dielectric) => {
      //panic!("DEBUG: created dielectric-dielectric interface");
      Rc::new(DielectricDielectricInterfaceMatDef{
        inc_refractive_index: match inc_vol.real_refractive_index_at(x) {
          None => panic!("dielectric interface missing incident real refractive index"),
          Some(index) => index,
        },
        ext_refractive_index: match ext_vol.real_refractive_index_at(x) {
          None => panic!("dielectric interface missing external real refractive index"),
          Some(index) => index,
        },
      })
    }
    _ => unreachable!(),
  }
}

pub struct LambertianInterfaceMatDef {
  // TODO
  absorb_prob:  f32,
}

impl InterfaceMat for LambertianInterfaceMatDef {
  fn scatter_surf_bwd(&self, inc_out_dir: Vector, inc_normal: Vector, /*epsilon: f32*/) -> (InterfaceEvent, Option<f32>) {
    // FIXME: normalize for view-dependent intensity.
    let v_x: f32 = thread_rng().sample(StandardNormal) as _;
    let v_y: f32 = thread_rng().sample(StandardNormal) as _;
    let v_z: f32 = thread_rng().sample(StandardNormal) as _;
    let mut v = Vector3::new(v_x, v_y, v_z).normalize();
    let v_dot_n = v.dot(inc_normal);
    if v_dot_n < 0.0 {
      v = -v;
    }
    /*(InterfaceEvent::Reflect(-v), Some(v_dot_n.abs()))*/
    (InterfaceEvent::Reflect(-v), Some(1.0 / v_dot_n.abs()))
  }
}

pub struct SpecularInterfaceMatDef {
  // TODO
  absorb_prob:  f32,
}

impl InterfaceMat for SpecularInterfaceMatDef {
  fn scatter_surf_bwd(&self, inc_out_dir: Vector, inc_normal: Vector, /*epsilon: f32*/) -> (InterfaceEvent, Option<f32>) {
    // TODO
    unimplemented!();
  }
}

pub struct MirrorInterfaceMatDef {
  // TODO
  absorb_prob:  f32,
}

impl InterfaceMat for MirrorInterfaceMatDef {
  fn scatter_surf_bwd(&self, inc_out_dir: Vector, inc_normal: Vector, /*epsilon: f32*/) -> (InterfaceEvent, Option<f32>) {
    // FIXME: normalize for view-dependent intensity.
    let i_cos = inc_out_dir.dot(inc_normal);
    assert!(i_cos >= 0.0);
    let inc_in_dir = -(2.0 * i_cos * inc_normal - inc_out_dir).normalize();
    (InterfaceEvent::Reflect(inc_in_dir), None)
  }
}

pub struct DielectricDielectricInterfaceMatDef {
  inc_refractive_index: f32,
  ext_refractive_index: f32,
}

impl InterfaceMat for DielectricDielectricInterfaceMatDef {
  fn scatter_surf_bwd(&self, inc_out_dir: Vector, inc_normal: Vector, /*epsilon: f32*/) -> (InterfaceEvent, Option<f32>) {
    // TODO
    // TODO: Fresnel's laws.
    // TODO: Snell's law.
    let i_idx = self.inc_refractive_index;
    let e_idx = self.ext_refractive_index;
    let ratio = i_idx / e_idx;
    let i_cos = inc_out_dir.dot(inc_normal).min(1.0);
    assert!(i_cos >= 0.0);
    let i_sin = (1.0 - i_cos * i_cos).sqrt().min(1.0);
    let e_sin = ratio * i_sin;
    if e_sin < 0.0 {
      unreachable!();
    } else if e_sin > 1.0 {
      let inc_in_dir = (inc_out_dir - 2.0 * i_cos * inc_normal).normalize();
      (InterfaceEvent::Reflect(inc_in_dir), None)
    } else {
      let e_cos = (1.0 - e_sin * e_sin).sqrt().min(1.0);
      let ext_in_dir = (ratio * inc_out_dir - (ratio * i_cos - e_cos) * inc_normal).normalize();
      (InterfaceEvent::Transmit(ext_in_dir), None)
    }
  }
}

#[derive(Clone, Copy)]
pub enum SurfaceMatKind {
  PassThrough,
  Lambertian,
  Specular,
  Mirror,
}

pub trait SurfaceMat {
  fn surf_mat_kind(&self) -> SurfaceMatKind;
  fn query_surf_emission(&self, out_ray: Ray, inc_normal: Vector) -> f32;
}

pub type InvisibleSurfaceMatDef = PassThroughSurfaceMatDef;

#[derive(Default)]
pub struct PassThroughSurfaceMatDef {
}

impl SurfaceMat for PassThroughSurfaceMatDef {
  fn surf_mat_kind(&self) -> SurfaceMatKind {
    SurfaceMatKind::PassThrough
  }

  fn query_surf_emission(&self, out_ray: Ray, inc_normal: Vector) -> f32 {
    0.0
  }
}

#[derive(Default)]
pub struct LambertianSurfaceMatDef {
  pub absorb_prob:  f32,
}

impl SurfaceMat for LambertianSurfaceMatDef {
  fn surf_mat_kind(&self) -> SurfaceMatKind {
    SurfaceMatKind::Lambertian
  }

  fn query_surf_emission(&self, out_ray: Ray, inc_normal: Vector) -> f32 {
    0.0
  }
}

#[derive(Default)]
pub struct MirrorSurfaceMatDef {
  pub absorb_prob:  f32,
}

impl SurfaceMat for MirrorSurfaceMatDef {
  fn surf_mat_kind(&self) -> SurfaceMatKind {
    SurfaceMatKind::Mirror
  }

  fn query_surf_emission(&self, out_ray: Ray, inc_normal: Vector) -> f32 {
    0.0
  }
}

pub struct SphericalLightSurfaceMatDef {
  pub emit_rad: f32,
}

impl SurfaceMat for SphericalLightSurfaceMatDef {
  fn surf_mat_kind(&self) -> SurfaceMatKind {
    SurfaceMatKind::PassThrough
  }

  fn query_surf_emission(&self, out_ray: Ray, inc_normal: Vector) -> f32 {
    self.emit_rad
  }
}

pub struct HemisphericalLightSurfaceMatDef {
  pub emit_rad: f32,
}

impl SurfaceMat for HemisphericalLightSurfaceMatDef {
  fn surf_mat_kind(&self) -> SurfaceMatKind {
    SurfaceMatKind::PassThrough
  }

  fn query_surf_emission(&self, out_ray: Ray, inc_normal: Vector) -> f32 {
    if out_ray.dir.dot(inc_normal) > 0.0 {
      self.emit_rad
    } else {
      0.0
    }
  }
}

#[derive(Clone, Copy)]
pub enum ScatterEvent {
  Absorb,
  Scatter(Ray),
}

#[derive(Clone, Copy)]
pub enum AttenuationEvent {
  NonTerm,
  Cutoff(Vector, f32),
  Attenuate(Vector, f32),
}

#[derive(Clone, Copy)]
pub enum VolumeMatKind {
  Unspecified,
  Dielectric,
}

pub trait VolumeMat {
  fn vol_mat_kind(&self) -> VolumeMatKind;
  fn vol_absorbing_coef_at(&self, x: Vector) -> f32;
  fn vol_scattering_coef_at(&self, x: Vector) -> f32;
  fn scatter_vol_bwd(&self, out_ray: Ray) -> ScatterEvent;
  fn query_vol_emission(&self, out_ray: Ray) -> f32;

  fn real_refractive_index_at(&self, x: Vector) -> Option<f32> {
    None
  }

  /*fn interface_at(&self, other: &VolumeMat, x: Vector) -> Rc<dyn InterfaceMat> {
    match (self.vol_mat_kind(), other.vol_mat_kind()) {
      (VolumeMatKind::Dielectric, VolumeMatKind::Dielectric) => {
        Rc::new(DielectricDielectricInterfaceMatDef{
          inc_refractive_index: match self.real_refractive_index_at(x) {
            None => panic!("dielectric interface missing incident real refractive index"),
            Some(index) => index,
          },
          ext_refractive_index: match other.real_refractive_index_at(x) {
            None => panic!("dielectric interface missing external real refractive index"),
            Some(index) => index,
          },
        })
      }
      _ => unreachable!(),
    }
  }*/

  fn vol_extinction_coef_at(&self, x: Vector) -> f32 {
    self.vol_absorbing_coef_at(x) + self.vol_scattering_coef_at(x)
  }

  fn max_vol_extinction_coef(&self) -> Option<f32> {
    None
  }

  fn woodcock_track_bwd(&self, out_ray: Ray, cutoff_dist: Option<f32>) -> AttenuationEvent {
    let max_coef = match self.max_vol_extinction_coef() {
      Some(coef) => coef,
      None => panic!("woodcock sampling requires upper bound on extinction coef"),
    };
    if max_coef == 0.0 && cutoff_dist.is_none() {
      return AttenuationEvent::NonTerm;
    }
    let cutoff_s = cutoff_dist.map(|d| d * max_coef);
    let mut xp = out_ray.origin;
    let mut s = 0.0;
    loop {
      let u1: f32 = thread_rng().sample(OpenClosed01);
      s -= u1.ln();
      xp = out_ray.origin - s * out_ray.dir / max_coef;
      if let Some(cutoff_s) = cutoff_s {
        if s >= cutoff_s {
          return AttenuationEvent::Cutoff(xp, s / max_coef);
        }
      }
      let u2: f32 = thread_rng().sample(Standard);
      if u2 * max_coef < self.vol_extinction_coef_at(xp) {
        break;
      }
    }
    AttenuationEvent::Attenuate(xp, s / max_coef)
  }
}

#[derive(Default)]
pub struct VacuumVolumeMatDef {
}

impl VolumeMat for VacuumVolumeMatDef {
  fn vol_mat_kind(&self) -> VolumeMatKind {
    VolumeMatKind::Dielectric
  }

  fn vol_absorbing_coef_at(&self, x: Vector) -> f32 {
    0.0
  }

  fn vol_scattering_coef_at(&self, x: Vector) -> f32 {
    0.0
  }

  fn max_vol_extinction_coef(&self) -> Option<f32> {
    Some(0.0)
  }

  fn scatter_vol_bwd(&self, out_ray: Ray) -> ScatterEvent {
    unreachable!();
  }

  fn query_vol_emission(&self, out_ray: Ray) -> f32 {
    0.0
  }

  fn real_refractive_index_at(&self, x: Vector) -> Option<f32> {
    Some(0.0)
  }
}

#[derive(Clone)]
pub struct HomogeneousDielectricVolumeMatDef {
  pub refractive_index: f32,
  pub absorb_coef:      f32,
  pub scatter_coef:     f32,
  pub scatter_dist:     Option<Rc<dyn HomogeneousScatterDist>>,
}

impl HomogeneousDielectricVolumeMatDef {
  pub fn default_hg(scatter_coef: f32, g: f32) -> Self {
    HomogeneousDielectricVolumeMatDef{
      refractive_index: 1.0,
      absorb_coef:      0.0,
      scatter_coef:     scatter_coef,
      scatter_dist:     Some(Rc::new(HGScatterDist{g})),
    }
  }
}

impl VolumeMat for HomogeneousDielectricVolumeMatDef {
  fn vol_mat_kind(&self) -> VolumeMatKind {
    VolumeMatKind::Dielectric
  }

  fn vol_absorbing_coef_at(&self, x: Vector) -> f32 {
    self.absorb_coef
  }

  fn vol_scattering_coef_at(&self, x: Vector) -> f32 {
    self.scatter_coef
  }

  fn max_vol_extinction_coef(&self) -> Option<f32> {
    Some(self.absorb_coef + self.scatter_coef)
  }

  fn scatter_vol_bwd(&self, out_ray: Ray) -> ScatterEvent {
    let a_coef = self.vol_absorbing_coef_at(out_ray.origin);
    let total_coef = self.vol_extinction_coef_at(out_ray.origin);
    let u: f32 = thread_rng().sample(Uniform::new(0.0, total_coef));
    if u < a_coef {
      ScatterEvent::Absorb
    } else if u + a_coef < total_coef {
      match self.scatter_dist {
        None => unreachable!(),
        Some(ref scatter_dist) => {
          let in_dir = scatter_dist.sample_bwd(out_ray.dir);
          ScatterEvent::Scatter(Ray{origin: out_ray.origin, dir: in_dir})
        }
      }
    } else {
      unreachable!();
    }
  }

  fn query_vol_emission(&self, out_ray: Ray) -> f32 {
    0.0
  }

  fn real_refractive_index_at(&self, x: Vector) -> Option<f32> {
    Some(self.refractive_index)
  }
}

pub trait HomogeneousScatterDist {
  fn sample_bwd(&self, out_dir: Vector) -> Vector;
}

pub struct IsotropicScatterDist;

impl HomogeneousScatterDist for IsotropicScatterDist {
  fn sample_bwd(&self, out_dir: Vector) -> Vector {
    // TODO
    unimplemented!();
  }
}

pub struct HGScatterDist {
  pub g:    f32,
}

impl HomogeneousScatterDist for HGScatterDist {
  fn sample_bwd(&self, out_dir: Vector) -> Vector {
    // TODO
    let g = self.g;
    let u1: f32 = thread_rng().sample(Standard);
    let r = (1.0 - g * g) / (1.0 + g * (2.0 * u1 - 1.0));
    let mu = 0.5 / g * (1.0 + g * g - r * r);
    let cos_th = mu.max(-1.0).min(1.0);
    let sin_th = (1.0 - cos_th * cos_th).sqrt(); // FIXME: need random sign flip?
    let u2: f32 = thread_rng().sample(Standard);
    let phi = 2.0 * PI * u2;
    let tfm = Quaternion::from_arc(Vector3::new(0.0, 0.0, 1.0), out_dir, None);
    let v = Vector3::new(
        phi.cos() * sin_th,
        phi.sin() * sin_th,
        cos_th,
    ).normalize();
    let in_dir = -tfm.rotate_vector(v);
    in_dir
  }
}

pub trait VtraceObj {
  fn intersect_bwd(&self, out_ray: Ray, epsilon: f32) -> Option<(Vector, f32)>;
  fn incident_normal(&self, inc_ray: Ray) -> Option<Vector>;
  fn project(&self, x: Vector) -> Vector;
  fn boundary_surf_mat(&self) -> Rc<dyn SurfaceMat>;
  fn interior_vol_mat(&self) -> Rc<dyn VolumeMat>;
}

pub struct SpaceObj {
  boundary_mat: Rc<PassThroughSurfaceMatDef>,
  interior_mat: Rc<dyn VolumeMat>,
}

impl SpaceObj {
  pub fn new(interior_mat: Rc<dyn VolumeMat>) -> Self {
    SpaceObj{
      boundary_mat: Rc::new(PassThroughSurfaceMatDef::default()),
      interior_mat,
    }
  }
}

impl VtraceObj for SpaceObj {
  fn intersect_bwd(&self, out_ray: Ray, epsilon: f32) -> Option<(Vector, f32)> {
    None
  }

  fn incident_normal(&self, inc_ray: Ray) -> Option<Vector> {
    None
  }

  fn project(&self, x: Vector) -> Vector {
    x
  }

  fn boundary_surf_mat(&self) -> Rc<dyn SurfaceMat> {
    self.boundary_mat.clone()
  }

  fn interior_vol_mat(&self) -> Rc<dyn VolumeMat> {
    self.interior_mat.clone()
  }
}

pub struct SphereObj {
  center:       Vector,
  radius:       f32,
  boundary_mat: Rc<dyn SurfaceMat>,
  interior_mat: Rc<VacuumVolumeMatDef>,
  //interior_mat: Rc<dyn VolumeMat>,
}

impl SphereObj {
  pub fn new(c: Vector, r: f32, boundary: Rc<dyn SurfaceMat>) -> Self {
    SphereObj{
      center:       c,
      radius:       r,
      boundary_mat: boundary,
      interior_mat: Rc::new(VacuumVolumeMatDef::default()),
    }
  }
}

impl VtraceObj for SphereObj {
  fn intersect_bwd(&self, out_ray: Ray, epsilon: f32) -> Option<(Vector, f32)> {
    // TODO
    let d = -out_ray.dir;
    let delta = out_ray.origin - self.center;
    let b = d.dot(delta);
    let determinant = b * b + self.radius * self.radius - delta.dot(delta);
    if determinant < -epsilon {
      None
    } else if determinant > epsilon {
      let t1 = -b - determinant.sqrt();
      let t2 = -b + determinant.sqrt();
      let t = match (t1 < 0.0, t2 < 0.0) {
        (false, false)  => t1.min(t2),
        (false, true)   => t1,
        (true,  false)  => t2,
        (true,  true)   => return None,
      };
      //let xp = out_ray.origin - t * out_ray.dir;
      let x = out_ray.origin - t * out_ray.dir;
      let delta = x - self.center;
      let n = delta.normalize();
      let xp = self.center + self.radius * n;
      Some((xp, t))
    } else {
      let t = -b;
      //let xp = out_ray.origin - t * out_ray.dir;
      let x = out_ray.origin - t * out_ray.dir;
      let delta = x - self.center;
      let n = delta.normalize();
      let xp = self.center + self.radius * n;
      Some((xp, t))
    }
  }

  fn incident_normal(&self, inc_ray: Ray) -> Option<Vector> {
    let delta = inc_ray.origin - self.center;
    let n = delta.normalize();
    if inc_ray.dir.dot(n) >= 0.0 {
      Some(n)
    } else {
      Some(-n)
    }
  }

  fn project(&self, x: Vector) -> Vector {
    let delta = x - self.center;
    let n = delta.normalize();
    let xp = self.center + self.radius * n;
    xp
  }

  fn boundary_surf_mat(&self) -> Rc<dyn SurfaceMat> {
    self.boundary_mat.clone()
  }

  fn interior_vol_mat(&self) -> Rc<dyn VolumeMat> {
    self.interior_mat.clone()
  }
}

pub struct QuadObj {
  // TODO: specify which normal?
  v00:  Vector,
  v01:  Vector,
  v10:  Vector,
  v11:  Vector,
  boundary_mat: Rc<dyn SurfaceMat>,
  interior_mat: Rc<VacuumVolumeMatDef>,
}

impl QuadObj {
  pub fn new(vs: Vec<Vector>, boundary_mat: Rc<dyn SurfaceMat>) -> Self {
    QuadObj{
      v00:  vs[0],
      v01:  vs[1],
      v10:  vs[2],
      v11:  vs[3],
      boundary_mat: boundary_mat,
      interior_mat: Rc::new(VacuumVolumeMatDef::default()),
    }
  }
}

impl VtraceObj for QuadObj {
  fn intersect_bwd(&self, out_ray: Ray, epsilon: f32) -> Option<(Vector, f32)> {
    // TODO
    //println!("DEBUG: QuadObj: intersect_bwd: {:?}", epsilon);
    let d = -out_ray.dir;
    let e01 = self.v10 - self.v00;
    let e03 = self.v01 - self.v00;
    let p = d.cross(e03);
    let det = e01.dot(p);
    if det.abs() < epsilon {
      return None;
    }
    let t = out_ray.origin - self.v00;
    let a = t.dot(p) / det;
    if a < 0.0 || a > 1.0 {
      return None;
    }
    let q = t.cross(e01);
    let b = d.dot(q) / det;
    if b < 0.0 || b > 1.0 {
      return None;
    }

    if (a + b) > 1.0 {
      let e23 = self.v01 - self.v11;
      let e21 = self.v10 - self.v11;
      let pp = d.cross(e21);
      let detp = e23.dot(pp);
      if detp.abs() < epsilon {
        return None;
      }
      let tp = out_ray.origin - self.v11;
      let ap = tp.dot(pp) / detp;
      if ap < 0.0 {
        return None;
      }
      let qp = tp.cross(e23);
      let bp = d.dot(qp) / detp;
      if bp < 0.0 {
        return None;
      }
    }

    let t = e03.dot(q) / det;
    if t < 0.0 {
      return None;
    }

    /*let (a11, b11) = {
      // TODO: calculate barycentric coordinates.
      let e02 = self.v11 - self.v00;
      let n = e01.cross(e03);
      if n.x.abs() >= n.y.abs() && n.x.abs() >= n.z.abs() {
        let a11 = (e02.y * e03.z - e02.z * e03.y) / n.x;
        let b11 = (e01.y * e02.z - e01.z * e02.y) / n.x;
        (a11, b11)
      } else if n.y.abs() >= n.x.abs() && n.y.abs() >= n.z.abs() {
        let a11 = (e02.z * e03.x - e02.x * e03.z) / n.y;
        let b11 = (e01.z * e02.x - e01.x * e02.z) / n.y;
        (a11, b11)
      } else {
        let a11 = (e02.x * e03.y - e02.y * e03.x) / n.z;
        let b11 = (e01.x * e02.y - e01.y * e02.x) / n.z;
        (a11, b11)
      }
    };*/

    /*let (u, v) = {
      // TODO: calculate bilinear coordinates.
      if (a11 - 1.0).abs() < epsilon {
        let u = a;
        let v = if (b11 - 1.0).abs() < epsilon {
          b
        } else {
          b / (u * (b11 - 1.0) + 1.0)
        };
        (u, v)
      } else if (b11 - 1.0).abs() < epsilon {
        let v = b;
        let u = a / (v * (a11 - 1.0) + 1.0);
        (u, v)
      } else {
        // TODO
        unimplemented!();
      }
    };*/

    let ixn_pt = self.v00 + a * e01 + b * e03;
    Some((ixn_pt, t))
  }

  fn incident_normal(&self, inc_ray: Ray) -> Option<Vector> {
    // TODO: check incident ray origin.
    let e01 = self.v10 - self.v00;
    let e03 = self.v01 - self.v00;
    let n = e01.cross(e03).normalize();
    if n.dot(inc_ray.dir) >= 0.0 {
      Some(n)
    } else {
      Some(-n)
    }
  }

  fn project(&self, x: Vector) -> Vector {
    // TODO
    unimplemented!();
  }

  fn boundary_surf_mat(&self) -> Rc<dyn SurfaceMat> {
    self.boundary_mat.clone()
  }

  fn interior_vol_mat(&self) -> Rc<dyn VolumeMat> {
    self.interior_mat.clone()
  }
}

#[derive(Clone, Copy)]
pub enum TraceEvent {
  NonTerm,
  Surface(Vector, f32, Option<usize>),
}

#[derive(Clone, Copy)]
pub enum Fov {
  Degrees(f32),
  Radians(f32),
  Tangent(f32),
}

#[derive(Clone, Copy)]
pub struct QueryOpts {
  pub trace_epsilon:    f32,
  pub importance_clip:  Option<f32>,
  pub roulette_term_p:  Option<f32>,
  pub verbose:          bool,
}

#[derive(Clone, Copy)]
pub struct RenderOpts {
  pub cam_origin:   Vector,
  pub cam_lookat:   Vector,
  pub cam_up:       Vector,
  pub cam_fov:      Fov,
  pub im_width:     usize,
  pub im_height:    usize,
  pub rays_per_pix: usize,
}

pub trait VtraceScene {
  fn trace_bwd(&self, out_ray: Ray, obj_id: Option<usize>, epsilon: f32) -> TraceEvent;
  fn query_surf_rad_bwd(&self, out_ray: Ray, inc_obj_id: Option<usize>, ext_obj_id: Option<usize>, depth: usize, default_opts: QueryOpts) -> f32;
  fn query_vol_rad_bwd(&self, out_ray: Ray, obj_id: Option<usize>, top_level: bool, depth: usize, /*top_level_opts: Option<QueryOpts>,*/ default_opts: QueryOpts) -> f32;

  fn render_depth(&self, render_opts: RenderOpts, buf: &mut MemArray3d<u8>) where Self: Sized {
    render_depth(self, render_opts, buf);
  }

  fn render_rad(&self, query_opts: QueryOpts, render_opts: RenderOpts, buf: &mut MemArray3d<u8>) where Self: Sized {
    render_vol_rad(self, query_opts, render_opts, buf);
  }
}

pub struct SimpleVtraceScene {
  objs: Vec<Rc<dyn VtraceObj>>,
}

impl SimpleVtraceScene {
  pub fn new(root_obj: Rc<dyn VtraceObj>) -> Self {
    SimpleVtraceScene{
      objs: vec![root_obj],
    }
  }

  pub fn add_object(&mut self, obj: Rc<dyn VtraceObj>) {
    self.objs.push(obj);
  }
}

impl VtraceScene for SimpleVtraceScene {
  fn trace_bwd(&self, out_ray: Ray, this_obj_id: Option<usize>, epsilon: f32) -> TraceEvent {
    let mut ixns = Vec::new();
    for (id, obj) in self.objs.iter().enumerate() {
      if let Some(this_obj_id) = this_obj_id {
        if this_obj_id == id {
          continue;
        }
      }
      if let Some((ixn_pt, ixn_dist)) = obj.intersect_bwd(out_ray, epsilon) {
        if ixn_dist > epsilon {
          ixns.push((id, ixn_pt, ixn_dist));
        }
      }
    }
    if ixns.is_empty() {
      TraceEvent::NonTerm
    } else {
      ixns.sort_unstable_by_key(|(_, _, t)| F32SupNan(*t));
      TraceEvent::Surface(ixns[0].1, ixns[0].2, Some(ixns[0].0))
    }
  }

  fn query_surf_rad_bwd(&self, out_ray: Ray, inc_obj_id: Option<usize>, ext_obj_id: Option<usize>, depth: usize, default_opts: QueryOpts) -> f32 {
    // TODO: project the intersection to the surface?
    let inc_obj = match inc_obj_id {
      None => unimplemented!("querying objects by coordinates is not supported"),
      Some(id) => self.objs[id].clone(),
    };
    let ext_obj = match ext_obj_id {
      None => unimplemented!("querying objects by coordinates is not supported"),
      Some(id) => self.objs[id].clone(),
    };
    let inc_surf = inc_obj.boundary_surf_mat();
    let inc_vol = inc_obj.interior_vol_mat();
    let ext_surf = ext_obj.boundary_surf_mat();
    let ext_vol = ext_obj.interior_vol_mat();
    /*let interface = inc_vol.interface_at(&*ext_vol, out_ray.origin);*/
    let interface = interface_at(inc_surf.clone(), inc_vol, ext_surf.clone(), ext_vol, out_ray.origin);
    let normal_dir = match (inc_obj.incident_normal(out_ray), ext_obj.incident_normal(out_ray)) {
      (None,                None)               => panic!(),
      (Some(inc_normal),    None)               => inc_normal,
      (None,                Some(ext_normal))   => ext_normal,
      (Some(inc_normal),    Some(ext_normal))   => 0.5 * (inc_normal + ext_normal),
    };
    let emit_rad = inc_surf.query_surf_emission(out_ray, -normal_dir) + ext_surf.query_surf_emission(out_ray, normal_dir);
    let mc_est_rad = {
      let (do_mc, roulette_mc_norm) = if default_opts.roulette_term_p.is_none() {
        (true, None)
      } else {
        let mc_p = 1.0 - default_opts.roulette_term_p.unwrap();
        let mc_u: f32 = thread_rng().sample(Standard);
        (mc_u < mc_p, Some(mc_p))
      };
      if do_mc {
        let (recurs_mc_est, recurs_mc_norm) = match interface.scatter_surf_bwd(out_ray.dir, normal_dir) {
          (InterfaceEvent::Absorb, scatter_norm) => {
            if default_opts.verbose {
              println!("DEBUG: query_surf_rad_bwd: depth: {} scatter surf: absorb", depth);
            }
            (0.0, scatter_norm)
          }
          (InterfaceEvent::Reflect(in_dir), scatter_norm) => {
            if default_opts.verbose {
              println!("DEBUG: query_surf_rad_bwd: depth: {} scatter surf: reflect: inc obj: {:?}", depth, inc_obj_id);
            }
            let eps = default_opts.trace_epsilon;
            let in_ray = Ray{origin: out_ray.origin - eps * in_dir, dir: in_dir};
            let next_est_rad = self.query_vol_rad_bwd(in_ray, inc_obj_id, false, depth + 1, default_opts);
            (next_est_rad, scatter_norm)
          }
          (InterfaceEvent::Transmit(in_dir), scatter_norm) => {
            if default_opts.verbose {
              println!("DEBUG: query_surf_rad_bwd: depth: {} scatter surf: transmit: ext obj: {:?}", depth, ext_obj_id);
            }
            let eps = default_opts.trace_epsilon;
            let in_ray = Ray{origin: out_ray.origin - eps * in_dir, dir: in_dir};
            let next_est_rad = self.query_vol_rad_bwd(in_ray, ext_obj_id, false, depth + 1, default_opts);
            (next_est_rad, scatter_norm)
          }
        };
        let total_mc_norm = roulette_mc_norm.unwrap_or(1.0) * recurs_mc_norm.unwrap_or(1.0);
        match default_opts.importance_clip {
          Some(c) => {
            if c * total_mc_norm < 1.0 {
              recurs_mc_est * c
            } else {
              recurs_mc_est / total_mc_norm
            }
          }
          None => {
            recurs_mc_est / total_mc_norm
          }
        }
      } else {
        if default_opts.verbose {
          println!("DEBUG: query_surf_rad_bwd: depth: {} roulette kill", depth);
        }
        0.0
      }
    };
    let this_rad = emit_rad + mc_est_rad;
    this_rad
  }

  fn query_vol_rad_bwd(&self, out_dst_ray: Ray, vol_obj_id: Option<usize>, top_level: bool, depth: usize, default_opts: QueryOpts) -> f32 {
    /*if default_opts.verbose {
      if top_level {
        println!("DEBUG: query_vol_rad_bwd:  top level");
      }
    }*/
    let (surf_cutoff_dist, surf_pt, surf_obj_id) = match self.trace_bwd(out_dst_ray, vol_obj_id, default_opts.trace_epsilon) {
      TraceEvent::NonTerm => {
        (None, None, None)
      }
      TraceEvent::Surface(p, dist, obj_id) => {
        (Some(dist), Some(p), obj_id)
      }
    };
    let this_obj = match vol_obj_id {
      None => unimplemented!("querying objects by coordinates is not supported"),
      Some(id) => self.objs[id].clone(),
    };
    let this_vol = this_obj.interior_vol_mat();
    let emit_rad = this_vol.query_vol_emission(out_dst_ray);
    let mc_est_rad = {
      let (do_mc, roulette_mc_norm) = if top_level || default_opts.roulette_term_p.is_none() {
        (true, None)
      } else {
        let mc_p = 1.0 - default_opts.roulette_term_p.unwrap();
        let mc_u: f32 = thread_rng().sample(Standard);
        (mc_u < mc_p, Some(mc_p))
      };
      if do_mc {
        let (recurs_mc_est, recurs_mc_norm)  = match this_vol.woodcock_track_bwd(out_dst_ray, surf_cutoff_dist) {
          AttenuationEvent::NonTerm => {
            if default_opts.verbose {
              println!("DEBUG: query_vol_rad_bwd:  depth: {} woodcock track: non terminal", depth);
            }
            (0.0, None)
          }
          AttenuationEvent::Cutoff(..) => {
            if default_opts.verbose {
              println!("DEBUG: query_vol_rad_bwd:  depth: {} woodcock track: cutoff: surf obj: {:?}", depth, surf_obj_id);
            }
            let out_src_ray = Ray{origin: surf_pt.unwrap(), dir: out_dst_ray.dir};
            let next_est_rad = self.query_surf_rad_bwd(out_src_ray, vol_obj_id, surf_obj_id, depth + 1, default_opts);
            (next_est_rad, None)
          }
          AttenuationEvent::Attenuate(p, dist) => {
            if default_opts.verbose {
              println!("DEBUG: query_vol_rad_bwd:  depth: {} woodcock track: attenuate", depth);
            }
            let out_src_ray = Ray{origin: p, dir: out_dst_ray.dir};
            let next_est_rad = match this_vol.scatter_vol_bwd(out_src_ray) {
              ScatterEvent::Absorb => {
                0.0
              }
              ScatterEvent::Scatter(in_ray) => {
                self.query_vol_rad_bwd(in_ray, vol_obj_id, false, depth + 1, default_opts)
              }
            };
            (next_est_rad, Some(this_vol.vol_extinction_coef_at(out_src_ray.origin)))
          }
        };
        let total_mc_norm = roulette_mc_norm.unwrap_or(1.0) * recurs_mc_norm.unwrap_or(1.0);
        match default_opts.importance_clip {
          Some(c) => {
            if c * total_mc_norm < 1.0 {
              recurs_mc_est * c
            } else {
              recurs_mc_est / total_mc_norm
            }
          }
          None => {
            recurs_mc_est / total_mc_norm
          }
        }
      } else {
        if default_opts.verbose {
          println!("DEBUG: query_vol_rad_bwd:  depth: {} roulette kill", depth);
        }
        0.0
      }
    };
    let this_rad = emit_rad + mc_est_rad;
    this_rad
  }
}

pub fn render_depth(scene: &dyn VtraceScene, render_opts: RenderOpts, buf: &mut MemArray3d<u8>) {
  let mut flat_buf = buf.flat_view_mut().unwrap();
  let flat_buf = flat_buf.as_mut_slice();

  let aspect_ratio = render_opts.im_height as f32 / render_opts.im_width as f32;
  let camera_width = 1.0;
  let camera_height = aspect_ratio;
  let camera_depth = match render_opts.cam_fov {
    Fov::Degrees(_) => unimplemented!(),
    Fov::Radians(_) => unimplemented!(),
    Fov::Tangent(t) => 0.5 * camera_width / t,
  };

  let camera_dir = (render_opts.cam_lookat - render_opts.cam_origin).normalize();
  let camera_reldir = Vector::new(0.0, 0.0, -1.0);
  let camera_relup = Vector::new(0.0, 1.0, 0.0);
  let rd2a_tfm = Quaternion::from_arc(camera_reldir, camera_dir, None);
  let a2rd_tfm = Quaternion::from_arc(camera_dir, camera_reldir, None);
  let camera_relup_rd = a2rd_tfm.rotate_vector(render_opts.cam_up).normalize();
  let r2rd_tfm = Quaternion::from_arc(camera_relup, camera_relup_rd, None);
  let rel_to_world = r2rd_tfm * rd2a_tfm;

  let camera_uc = -0.5 * camera_width;
  let camera_vc = 0.5 * camera_height;
  let camera_inc_u = camera_width / render_opts.im_width as f32;
  let camera_inc_v = -camera_height / render_opts.im_height as f32;
  let epsilon = 1.0e-6;

  for screen_v in 0 .. render_opts.im_height {
    for screen_u in 0 .. render_opts.im_width {
      let camera_u = camera_uc + (0.5 + screen_u as f32) * camera_inc_u;
      let camera_v = camera_vc + (0.5 + screen_v as f32) * camera_inc_v;
      let camera_w = -camera_depth;
      let camera_relp = Vector3{x: camera_u, y: camera_v, z: camera_w}.normalize();
      let out_dir = -rel_to_world.rotate_vector(camera_relp).normalize();
      let out_ray = Ray{origin: render_opts.cam_origin, dir: out_dir};
      let depth = match scene.trace_bwd(out_ray, Some(0), epsilon) {
        TraceEvent::NonTerm => {
          1.0 / 0.0
        }
        TraceEvent::Surface(ixn_pt, depth, _) => {
          //println!("DEBUG: render_depth: hit surface");
          depth
        }
      };
      let pix_soft_val = (depth).atan() * FRAC_2_PI;
      let pix_val: u8 = ((1.0 - pix_soft_val.max(0.0).min(1.0).powf(2.2)) * 255.0).round() as u8;
      flat_buf[0 + 3 * (screen_u + render_opts.im_width * screen_v)] = pix_val;
      flat_buf[1 + 3 * (screen_u + render_opts.im_width * screen_v)] = pix_val;
      flat_buf[2 + 3 * (screen_u + render_opts.im_width * screen_v)] = pix_val;
    }
  }
}

pub fn render_vol_rad(scene: &dyn VtraceScene, query_opts: QueryOpts, render_opts: RenderOpts, buf: &mut MemArray3d<u8>) {
  let mut flat_buf = buf.flat_view_mut().unwrap();
  let flat_buf = flat_buf.as_mut_slice();

  let aspect_ratio = render_opts.im_height as f32 / render_opts.im_width as f32;
  let camera_width = 1.0;
  let camera_height = aspect_ratio;
  let camera_depth = match render_opts.cam_fov {
    Fov::Degrees(_) => unimplemented!(),
    Fov::Radians(_) => unimplemented!(),
    Fov::Tangent(t) => 0.5 * camera_width / t,
  };

  let camera_dir = (render_opts.cam_lookat - render_opts.cam_origin).normalize();
  let camera_reldir = Vector::new(0.0, 0.0, -1.0);
  let camera_relup = Vector::new(0.0, 1.0, 0.0);
  let rd2a_tfm = Quaternion::from_arc(camera_reldir, camera_dir, None);
  let a2rd_tfm = Quaternion::from_arc(camera_dir, camera_reldir, None);
  let camera_relup_rd = a2rd_tfm.rotate_vector(render_opts.cam_up).normalize();
  let r2rd_tfm = Quaternion::from_arc(camera_relup, camera_relup_rd, None);
  let rel_to_world = r2rd_tfm * rd2a_tfm;

  let camera_uc = -0.5 * camera_width;
  let camera_vc = 0.5 * camera_height;
  let camera_inc_u = camera_width / render_opts.im_width as f32;
  let camera_inc_v = -camera_height / render_opts.im_height as f32;

  for screen_v in 0 .. render_opts.im_height {
    for screen_u in 0 .. render_opts.im_width {
      let mut do_debug = false;
      /*if screen_u >= 400 && screen_u <= 428 && screen_v >= 512 && screen_v <= 532 {
        do_debug = true;
      }*/
      if do_debug {
        println!("DEBUG: render_vol_rad: u: {} v: {}", screen_u, screen_v);
      }
      let mut pix_rad_accumulator: f32 = 0.0;
      for i in 0 .. render_opts.rays_per_pix {
        let jitter_u: f32 = thread_rng().sample(Open01);
        let jitter_v: f32 = thread_rng().sample(Open01);
        let camera_u = camera_uc + (jitter_u + screen_u as f32) * camera_inc_u;
        let camera_v = camera_vc + (jitter_v + screen_v as f32) * camera_inc_v;
        let camera_w = -camera_depth;
        let camera_relp = Vector3{x: camera_u, y: camera_v, z: camera_w}.normalize();
        let out_dir = -rel_to_world.rotate_vector(camera_relp).normalize();
        let out_ray = Ray{origin: render_opts.cam_origin, dir: out_dir};
        let query_opts = if do_debug {
          let mut debug_query_opts = query_opts;
          debug_query_opts.verbose = true;
          debug_query_opts
        } else {
          query_opts
        };
        let pix_rad = scene.query_vol_rad_bwd(out_ray, Some(0), true, 0, query_opts);
        pix_rad_accumulator += 1.0 / (i as f32 + 1.0) * (pix_rad - pix_rad_accumulator);
      }
      if pix_rad_accumulator > 0.0 {
        //println!("DEBUG: pix rad est: {:?}", pix_rad_accumulator);
      }
      /*let pix_soft_val = (pix_rad_accumulator).atan() * FRAC_2_PI;
      let pix_val: u8 = ((pix_soft_val.max(0.0).min(1.0).powf(2.2)) * 255.0).round() as u8;*/
      let pix_linear_val = pix_rad_accumulator.max(0.0).min(1.0);
      let pix_srgb_val = linear2srgb(pix_linear_val);
      let pix_val: u8 = (pix_srgb_val * 255.0).round() as u8;
      if do_debug {
        flat_buf[0 + 3 * (screen_u + render_opts.im_width * screen_v)] = 0;
        flat_buf[1 + 3 * (screen_u + render_opts.im_width * screen_v)] = pix_val;
        flat_buf[2 + 3 * (screen_u + render_opts.im_width * screen_v)] = 0;
      } else {
        flat_buf[0 + 3 * (screen_u + render_opts.im_width * screen_v)] = pix_val;
        flat_buf[1 + 3 * (screen_u + render_opts.im_width * screen_v)] = pix_val;
        flat_buf[2 + 3 * (screen_u + render_opts.im_width * screen_v)] = pix_val;
      }
    }
  }
}
