use ::geometry::*;

use cgmath::*;
use float::ord::{F32SupNan};
use rand::prelude::*;
use rand::distributions::{Uniform, StandardNormal};

use std::f32::consts::{PI};
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

#[derive(Clone)]
pub struct RayTraceState {
  pub mag:  f32,
  pub ray:  Ray,
}

#[derive(Clone)]
pub enum RayTraceNode {
  Source(SourceMatDef),
  Air(Vector3<f32>, ContinuumMatDef),
  AirInterface(Vector3<f32>, AirInterfaceMatDef),
  NoTerm,
  EarlyTerm,
}

pub trait SourceDist {
  fn sample_event(&self) -> (f32, Ray);
  //fn weight(&self) -> f32;
}

#[derive(Clone, Copy)]
pub struct HemiPointSource {
  pub mag:      f32,
  pub origin:   Vector3<f32>,
  pub normal:   Vector3<f32>,
}

impl SourceDist for HemiPointSource {
  fn sample_event(&self) -> (f32, Ray) {
    let stdnrm_dist = StandardNormal;
    let mut rng = thread_rng();
    let mut v = Vector3::new(
        rng.sample(stdnrm_dist) as f32,
        rng.sample(stdnrm_dist) as f32,
        rng.sample(stdnrm_dist) as f32,
    ).normalize();
    v.z = v.z.abs();
    let z_axis = Vector3::new(0.0, 0.0, 1.0);
    let q = Quaternion::from_arc(z_axis, self.normal, None);
    let dir = q.rotate_vector(v).normalize();
    (self.mag, Ray{orig: self.origin, dir})
  }
}

#[derive(Clone, Copy)]
pub struct QuadrilateralSource {
  // TODO
  pub origin:   Vector3<f32>,
  pub normal:   Vector3<f32>,
}

/*impl SourceDist for QuadrilateralSource {
  fn sample_event(&self) -> (f32, Ray) {
    // TODO
    unimplemented!();
  }
}*/

#[derive(Clone)]
pub struct UnionSource {
  // TODO
}

#[derive(Clone)]
pub struct SourceMatDef {
  pub source_dist:  Rc<SourceDist>,
}

#[derive(Clone, Copy)]
pub enum AirInterfaceScatterEvent {
  Absorbed,
  Reflected(Vector3<f32>), // HemiSolidAngle<f32>),
  Transmitted(Vector3<f32>), // HemiSolidAngle<f32>),
}

pub trait AirInterfaceScatterDist {
  // TODO
  fn sample_event(&self, in_mat: &AirInterfaceMatDef, in_dir: HemiSolidAngle<f32>, normal: Vector3<f32>) -> AirInterfaceScatterEvent;
}

#[derive(Clone, Copy)]
pub struct AirDiffuseInterface {
  pub absorb_prob:  f32,
  //pub reflect_prob: f32,
}

impl AirInterfaceScatterDist for AirDiffuseInterface {
  fn sample_event(&self, in_mat: &AirInterfaceMatDef, in_dir: HemiSolidAngle<f32>, normal: Vector3<f32>) -> AirInterfaceScatterEvent {
    let uniform_dist = Uniform::new(0.0, 1.0);
    let stdnrm_dist = StandardNormal;
    let mut rng = thread_rng();
    let u = rng.sample(uniform_dist);
    if u < self.absorb_prob {
      return AirInterfaceScatterEvent::Absorbed;
    }
    let mut v = Vector3::new(
        rng.sample(stdnrm_dist) as f32,
        rng.sample(stdnrm_dist) as f32,
        rng.sample(stdnrm_dist) as f32,
    ).normalize();
    v.z = v.z.abs();
    let z_axis = Vector3::new(0.0, 0.0, 1.0);
    let q = Quaternion::from_arc(z_axis, normal, None);
    let dir = q.rotate_vector(v).normalize();
    AirInterfaceScatterEvent::Reflected(dir)
  }
}

#[derive(Clone, Copy)]
pub struct AirSpecularInterface {
  pub absorb_prob:  f32,
  //pub reflect_prob: f32,
}

impl AirInterfaceScatterDist for AirSpecularInterface {
  fn sample_event(&self, in_mat: &AirInterfaceMatDef, in_dir: HemiSolidAngle<f32>, normal: Vector3<f32>) -> AirInterfaceScatterEvent {
    let uniform_dist = Uniform::new(0.0, 1.0);
    let mut rng = thread_rng();
    let u = rng.sample(uniform_dist);
    if u < self.absorb_prob {
      return AirInterfaceScatterEvent::Absorbed;
    }
    let out_radial = clip_radian(in_dir.radial + PI);
    let v_x = out_radial.cos() * in_dir.incident.sin();
    let v_y = out_radial.sin() * in_dir.incident.sin();
    let v_z = in_dir.incident.cos();
    let v = Vector3::new(v_x, v_y, v_z).normalize();
    let z_axis = Vector3::new(0.0, 0.0, 1.0);
    let q = Quaternion::from_arc(z_axis, normal, None);
    let dir = q.rotate_vector(v);
    AirInterfaceScatterEvent::Reflected(dir)
  }
}

#[derive(Clone, Copy)]
pub struct AirDielectricInterface {
  // TODO: Fresnel equations for absorb/reflect prob.
  pub absorb_prob:  f32,
  pub reflect_prob: f32,
  pub inside_idx:   f32,
}

impl AirInterfaceScatterDist for AirDielectricInterface {
  fn sample_event(&self, in_mat: &AirInterfaceMatDef, in_dir: HemiSolidAngle<f32>, normal: Vector3<f32>) -> AirInterfaceScatterEvent {
    // TODO: Snell's law; allow for possibility of total internal reflection.
    /*//let out_incident = ;
    let out_radial = clip_radian(in_dir.radial + PI);
    HemiSolidAngle{incident: out_incident, radial: out_radial}*/
    unimplemented!();
  }
}

#[derive(Clone)]
pub struct AirInterfaceMatDef {
  pub refractive_index:     f32,
  pub surf_scatter_dist:    Rc<AirInterfaceScatterDist>,
}

impl AirInterfaceMatDef {
  /*pub fn _test_types(&self) {
    let in_dir = HemiSolidAngle{incident: 0.0, radial: 0.0};
    let event = self.surf_scatter_dist.sample_event(in_dir);
  }*/
}

// TODO
pub trait ContinuumMat {
  //fn sample_out_ray(&self, in_mat: &Material, in_ray: &Ray) -> Ray;
}

#[derive(Clone)]
pub struct ContinuumMatDef {
  pub refractive_index: f32,
  pub absorb_coef:      f32,
  pub scatter_coef:     f32,
  // TODO: specify scattering phase function here.
  //pub scatter_dist:     Rc<ScatterDist>,
}

impl ContinuumMatDef {
  pub fn default_vacuum() -> ContinuumMatDef {
    ContinuumMatDef{
      refractive_index: 1.0,
      absorb_coef:      0.0,
      scatter_coef:     0.0,
    }
  }

  pub fn default_air() -> ContinuumMatDef {
    ContinuumMatDef{
      // TODO
      refractive_index: 1.003,
      absorb_coef:      0.0,
      scatter_coef:     0.0,
    }
  }
}

#[derive(Clone, Copy)]
pub enum AirInterfaceDir {
  In,
  Out,
}

#[derive(Clone)]
pub struct ScatterHit {
}

#[derive(Clone)]
pub struct AirInterfaceHit {
  pub surf_normal:  Vector3<f32>,
  pub surf_mat:     AirInterfaceMatDef,
  pub inside_mat:   Option<ContinuumMatDef>,
  pub direction:    AirInterfaceDir,
}

pub trait TraceRay: IntersectsRay {
  // TODO
  fn trace_ray(&self, ray: &Ray, threshold: f32) -> Option<(RayIntersection, AirInterfaceHit)>;
}

#[derive(Clone)]
pub struct TraceScene {
  pub air_mat:  ContinuumMatDef,
  pub objs:     Vec<Rc<TraceRay>>,
}

impl Default for TraceScene {
  fn default() -> TraceScene {
    TraceScene{
      air_mat:  ContinuumMatDef::default_vacuum(),
      objs:     vec![],
    }
  }
}

impl TraceScene {
  pub fn new(air_mat: ContinuumMatDef) -> TraceScene {
    TraceScene{
      air_mat,
      objs:         vec![],
    }
  }

  pub fn add_object<Obj: TraceRay + 'static>(&mut self, obj: Obj) {
    self.objs.push(Rc::new(obj));
  }
}

impl IntersectsRay for TraceScene {
  fn intersects_ray(&self, ray: &Ray, threshold: f32) -> Option<RayIntersection> {
    let mut obj_ixns = vec![];
    for (obj_idx, obj) in self.objs.iter().enumerate() {
      if let Some(ixn) = obj.intersects_ray(&ray, threshold) {
        obj_ixns.push((obj_idx, ixn));
      }
    }
    if obj_ixns.is_empty() {
      None
    } else {
      obj_ixns.sort_unstable_by_key(|(_, ixn)| {
        let t = ixn.ray_coord.t;
        F32SupNan(if t < threshold { 1.0 / 0.0 } else { t })
      });
      Some(obj_ixns[0].1)
    }
  }
}

impl TraceRay for TraceScene {
  fn trace_ray(&self, ray: &Ray, threshold: f32) -> Option<(RayIntersection, AirInterfaceHit)> {
    // TODO
    unimplemented!();
  }
}
