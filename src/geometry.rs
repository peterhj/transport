use cgmath::*;
use float::ord::*;
use rand::prelude::*;
use rand::distributions::{Uniform};

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

#[derive(Clone, Copy)]
pub struct Parametric<T> {
  pub t:    T,
}

#[derive(Clone, Copy)]
pub struct Barycentric2<T> {
  pub u:    T,
  pub v:    T,
}

#[derive(Clone, Copy)]
pub struct HemiSolidAngle<T> {
  pub theta:    T,
  pub phi:      T,
}

#[derive(Clone, Copy)]
pub struct Ray {
  pub orig: Vector3<f32>,
  pub dir:  Vector3<f32>,
}

#[derive(Clone, Copy)]
pub struct RayIntersection {
  pub ray_coord:    Parametric<f32>,
  //pub world_coord:  Vector3<f32>,
}

pub trait IntersectsRay {
  fn intersects_ray(&self, ray: &Ray, threshold: f32) -> Option<RayIntersection>;
}

#[derive(Clone, Copy)]
pub enum SurfaceScatterEvent {
  Reflected(HemiSolidAngle<f32>),
  Transmitted(HemiSolidAngle<f32>),
}

pub trait SurfaceScatterDist {
  // TODO
  fn sample(&self, in_dir: HemiSolidAngle<f32>) -> HemiSolidAngle<f32>;
  //fn sample(&self, in_dir: HemiSolidAngle<f32>) -> SurfaceScatterEvent;
}

pub struct DiffuseReflect {
  pub reflect_prob: f32,
}

impl SurfaceScatterDist for DiffuseReflect {
  fn sample(&self, _in_dir: HemiSolidAngle<f32>) -> HemiSolidAngle<f32> {
    // TODO: normalizing radians.
    let theta_dist = Uniform::new_inclusive(0.0, 0.5 * PI);
    let phi_dist = Uniform::new(-PI, PI);
    let mut rng = thread_rng();
    let out_theta = rng.sample(theta_dist);
    let out_phi = rng.sample(phi_dist);
    HemiSolidAngle{theta: out_theta, phi: out_phi}
  }
}

pub struct SpecularReflect {
  pub reflect_prob: f32,
}

impl SurfaceScatterDist for SpecularReflect {
  fn sample(&self, in_dir: HemiSolidAngle<f32>) -> HemiSolidAngle<f32> {
    // TODO: normalizing radians.
    let out_phi = clip_radian(in_dir.phi + PI);
    HemiSolidAngle{theta: in_dir.theta, phi: out_phi}
  }
}

pub struct DielectricSurface {
  pub reflect_prob: f32,
  pub inside_idx:   f32,
  // TODO: not a good idea to store the "previous" index here.
  //pub outside_idx:  f32,
}

impl SurfaceScatterDist for DielectricSurface {
  fn sample(&self, in_dir: HemiSolidAngle<f32>) -> HemiSolidAngle<f32> {
    // TODO: Snell's law; allow for possibility of total internal reflection.
    // TODO: need "previous" material refractive index.
    /*//let out_theta = ;
    let out_phi = clip_radian(in_dir.phi + PI);
    HemiSolidAngle{theta: out_theta, phi: out_phi}*/
    unimplemented!();
  }
}

#[derive(Clone)]
pub struct SurfaceMatDef {
  pub refract_index:        f32,
  pub surf_absorb_prob:     f32,
  pub surf_scatter_prob:    f32,
  pub surf_scatter_dist:    Rc<SurfaceScatterDist>,
}

impl SurfaceMatDef {
  pub fn _test_types(&self) {
    let in_dir = HemiSolidAngle{theta: 0.0, phi: 0.0};
    let out_dir = self.surf_scatter_dist.sample(in_dir);
  }
}

// TODO
pub trait ContinuumMat {
  //fn sample_out_ray(&self, in_mat: &Material, in_ray: &Ray) -> Ray;
}

#[derive(Clone)]
pub struct ContinuumMatDef {
  pub refract_index:    f32,
  pub absorb_coef:      f32,
  pub scatter_coef:     f32,
  //pub scatter_dist:     Rc<ScatterDist>,
}

impl ContinuumMatDef {
  pub fn default_vacuum() -> ContinuumMatDef {
    ContinuumMatDef{
      refract_index:    1.0,
      absorb_coef:      0.0,
      scatter_coef:     0.0,
    }
  }
}

#[derive(Clone)]
pub struct SurfaceHit {
  pub surf_normal:  Vector3<f32>,
  pub surf_mat:     SurfaceMatDef,
  pub inside_mat:   Option<ContinuumMatDef>,
}

pub trait TraceRay {
  // TODO
  fn trace_ray(&self, ray: &Ray, threshold: f32) -> Option<(RayIntersection, SurfaceHit)>;
}

#[derive(Clone, Copy)]
pub struct Sphere {
  pub center:   Vector3<f32>,
  pub radius:   f32,
}

impl Sphere {
  pub fn new(center: Vector3<f32>, radius: f32) -> Sphere {
    Sphere{
      center, radius,
    }
  }
}

#[derive(Clone, Copy)]
pub struct SphereRayIntersection {
  pub ray_coord:    Parametric<f32>,
}

impl IntersectsRay for Sphere {
  //type Intersection = SphereRayIntersection;

  fn intersects_ray(&self, ray: &Ray, threshold: f32) -> Option<RayIntersection> {
    let delta = ray.orig - self.center;
    let b = ray.dir.dot(delta);
    let determinant = b * b + self.radius * self.radius - delta.dot(delta);
    if determinant < -threshold {
      None
    } else if determinant.abs() <= threshold {
      let t = -b;
      Some(RayIntersection{ray_coord: Parametric{t}})
    } else if determinant > threshold {
      let t1 = -b - determinant.sqrt();
      let t2 = -b + determinant.sqrt();
      let t = match (t1 < 0.0, t2 < 0.0) {
        (false, false)  => t1.min(t2),
        (false, true)   => t1,
        (true,  false)  => t2,
        (true,  true)   => return None,
      };
      Some(RayIntersection{ray_coord: Parametric{t}})
    } else {
      unreachable!();
    }
  }
}

#[derive(Clone, Copy)]
pub struct Triangle {
  pub v0:   Vector3<f32>,
  pub v1:   Vector3<f32>,
  pub v2:   Vector3<f32>,
}

#[derive(Clone, Copy)]
pub struct RayTriangleIntersection {
  pub ray_coord:    Parametric<f32>,
  pub tri_coords:   Barycentric2<f32>,
}

impl IntersectsRay for Triangle {
  //type Intersection = RayTriangleIntersection;

  fn intersects_ray(&self, ray: &Ray, threshold: f32) -> Option<RayIntersection> {
    let e1 = self.v1 - self.v0;
    let e2 = self.v2 - self.v0;
    let pvec = ray.dir.cross(e2);
    let det = e1.dot(pvec);
    if det.abs() <= threshold {
      return None;
    }
    let inv_det = 1.0 / det;
    let tvec = ray.orig - self.v0;
    let u = tvec.dot(pvec) * inv_det;
    if u < 0.0 || u > 1.0 {
      return None;
    }
    let qvec = tvec.cross(e1);
    let v = ray.dir.dot(qvec) * inv_det;
    if v < 0.0 || u + v > 1.0 {
      return None;
    }
    let t = e2.dot(qvec) * inv_det;
    Some(RayIntersection{
      ray_coord:  Parametric{t},
      //tri_coords: Barycentric2{u, v},
    })
  }
}

#[derive(Clone)]
pub struct TriMesh {
  pub vertexs:      Vec<Vector3<f32>>,
  pub normals:      Vec<Vector3<f32>>,
  pub triangles:    Vec<Triangle>,
}

impl IntersectsRay for TriMesh {
  //type Intersection = RayTriangleIntersection;

  fn intersects_ray(&self, ray: &Ray, threshold: f32) -> Option<RayIntersection> {
    let mut ray_ixns = vec![];
    for (tri_idx, tri) in self.triangles.iter().enumerate() {
      if let Some(ixn) = tri.intersects_ray(ray, threshold) {
        ray_ixns.push((tri_idx, ixn));
      }
    }
    if ray_ixns.is_empty() {
      None
    } else {
      ray_ixns.sort_unstable_by_key(|(_, ixn)| {
        let t = ixn.ray_coord.t;
        F32SupNan(if t < 0.0 { 1.0 / 0.0 } else { t })
      });
      if ray_ixns[0].1.ray_coord.t < 0.0 {
        None
      } else {
        Some(ray_ixns[0].1)
      }
    }
  }
}

pub struct IndexedTriMesh {
  // TODO: index data structure.
  mesh: TriMesh,
}

#[derive(Clone, Default)]
pub struct SimpleMeshScene {
  pub objs: Vec<TriMesh>,
}

impl IntersectsRay for SimpleMeshScene {
  fn intersects_ray(&self, ray: &Ray, threshold: f32) -> Option<RayIntersection> {
    let mut obj_ixns = vec![];
    for (obj_idx, obj) in self.objs.iter().enumerate() {
      if let Some(ixn) = obj.intersects_ray(&ray, 1.0e-6) {
        obj_ixns.push((obj_idx, ixn));
      }
    }
    if obj_ixns.is_empty() {
      None
    } else {
      obj_ixns.sort_unstable_by_key(|(_, ixn)| {
        let t = ixn.ray_coord.t;
        F32SupNan(if t < 0.0 { 1.0 / 0.0 } else { t })
      });
      Some(obj_ixns[0].1)
    }
  }
}

#[derive(Clone, Default)]
pub struct SimpleScene {
  pub objs: Vec<Rc<IntersectsRay>>,
}

impl SimpleScene {
  pub fn new() -> SimpleScene {
    Self::default()
  }

  pub fn add_object<Obj: IntersectsRay + 'static>(&mut self, obj: Obj) {
    self.objs.push(Rc::new(obj));
  }
}

impl IntersectsRay for SimpleScene {
  fn intersects_ray(&self, ray: &Ray, threshold: f32) -> Option<RayIntersection> {
    let mut obj_ixns = vec![];
    for (obj_idx, obj) in self.objs.iter().enumerate() {
      if let Some(ixn) = obj.intersects_ray(&ray, 1.0e-6) {
        obj_ixns.push((obj_idx, ixn));
      }
    }
    if obj_ixns.is_empty() {
      None
    } else {
      obj_ixns.sort_unstable_by_key(|(_, ixn)| {
        let t = ixn.ray_coord.t;
        F32SupNan(if t < 0.0 { 1.0 / 0.0 } else { t })
      });
      Some(obj_ixns[0].1)
    }
  }
}

#[derive(Clone)]
pub struct TraceScene {
  pub default_mat:  ContinuumMatDef,
  pub objs:         Vec<Rc<TraceRay>>,
}

impl Default for TraceScene {
  fn default() -> TraceScene {
    TraceScene{
      default_mat:  ContinuumMatDef::default_vacuum(),
      objs:         vec![],
    }
  }
}

impl TraceScene {
  pub fn new(default_mat: ContinuumMatDef) -> TraceScene {
    TraceScene{
      default_mat,
      objs:         vec![],
    }
  }

  pub fn add_object<Obj: TraceRay + 'static>(&mut self, obj: Obj) {
    self.objs.push(Rc::new(obj));
  }
}
