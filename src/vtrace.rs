use ::geometry::*;

use cgmath::*;
use float::ord::*;
use memarray::*;
use rand::prelude::*;
use rand::distributions::{OpenClosed01, Standard, StandardNormal, Uniform};

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

#[derive(Clone, Copy)]
pub enum InterfaceEvent {
  Absorb,
  Reflect(Vector),
  Transmit(Vector),
}

pub trait InterfaceMat {
  fn scatter_surf_bwd(&self, inc_out_dir: Vector, inc_normal: Vector) -> InterfaceEvent;
}

pub struct DielectricDielectricInterfaceMatDef {
  inc_refractive_index: f32,
  ext_refractive_index: f32,
}

impl InterfaceMat for DielectricDielectricInterfaceMatDef {
  fn scatter_surf_bwd(&self, inc_out_dir: Vector, normal: Vector, /*epsilon: f32*/) -> InterfaceEvent {
    // TODO
    // TODO: Fresnel's laws.
    // TODO: Snell's law.
    let i_idx = self.inc_refractive_index;
    let e_idx = self.ext_refractive_index;
    let ratio = i_idx / e_idx;
    let i_cos = inc_out_dir.dot(normal).min(1.0);
    assert!(i_cos >= 0.0);
    let i_sin = (1.0 - i_cos * i_cos).sqrt().min(1.0);
    let e_sin = ratio * i_sin;
    if e_sin < 0.0 {
      unreachable!();
    } else if e_sin > 1.0 {
      let inc_in_dir = (inc_out_dir - 2.0 * i_cos * normal).normalize();
      InterfaceEvent::Reflect(inc_in_dir)
    } else {
      let e_cos = (1.0 - e_sin * e_sin).sqrt().min(1.0);
      let ext_in_dir = (ratio * inc_out_dir - (ratio * i_cos - e_cos) * normal).normalize();
      InterfaceEvent::Transmit(ext_in_dir)
    }
  }
}

pub trait SurfaceMat {
  fn query_surf_emission(&self, out_ray: Ray, normal: Vector) -> f32;
}

#[derive(Default)]
pub struct InvisibleSurfaceMatDef {
}

impl SurfaceMat for InvisibleSurfaceMatDef {
  fn query_surf_emission(&self, out_ray: Ray, normal: Vector) -> f32 {
    0.0
  }
}

pub struct HemisphericLightSurfaceMatDef {
  pub emit_rad: f32,
}

impl SurfaceMat for HemisphericLightSurfaceMatDef {
  fn query_surf_emission(&self, out_ray: Ray, normal: Vector) -> f32 {
    if out_ray.dir.dot(normal) > 0.0 {
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
  Dielectric,
}

pub trait VolumeMat {
  fn mat_kind(&self) -> VolumeMatKind;
  fn vol_absorbing_coef_at(&self, x: Vector) -> f32;
  fn vol_scattering_coef_at(&self, x: Vector) -> f32;
  fn scatter_vol_bwd(&self, out_ray: Ray) -> ScatterEvent;
  fn query_vol_emission(&self, out_ray: Ray) -> f32;

  fn real_refractive_index_at(&self, x: Vector) -> Option<f32> {
    None
  }

  fn interface_at(&self, other: &VolumeMat, x: Vector) -> Rc<dyn InterfaceMat> {
    match (self.mat_kind(), other.mat_kind()) {
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
    }
  }

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
    if max_coef == 0.0 {
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
  fn mat_kind(&self) -> VolumeMatKind {
    VolumeMatKind::Dielectric
  }

  fn vol_absorbing_coef_at(&self, x: Vector) -> f32 {
    0.0
  }

  fn vol_scattering_coef_at(&self, x: Vector) -> f32 {
    0.0
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

impl VolumeMat for HomogeneousDielectricVolumeMatDef {
  fn mat_kind(&self) -> VolumeMatKind {
    VolumeMatKind::Dielectric
  }

  fn vol_absorbing_coef_at(&self, x: Vector) -> f32 {
    self.absorb_coef
  }

  fn vol_scattering_coef_at(&self, x: Vector) -> f32 {
    self.scatter_coef
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
    unimplemented!();
  }
}

pub trait VtraceObj {
  fn intersect_bwd(&self, out_ray: Ray, epsilon: f32) -> Option<(Vector, f32)>;
  fn normal_at(&self, x: Vector) -> Option<Vector>;
  fn project(&self, x: Vector) -> Vector;
  fn boundary_surf_mat(&self) -> Rc<dyn SurfaceMat>;
  fn interior_vol_mat(&self) -> Rc<dyn VolumeMat>;
}

pub struct SpaceObj {
  boundary_mat: Rc<InvisibleSurfaceMatDef>,
  interior_mat: Rc<dyn VolumeMat>,
}

impl SpaceObj {
  pub fn new(interior_mat: Rc<dyn VolumeMat>) -> Self {
    SpaceObj{
      boundary_mat: Rc::new(InvisibleSurfaceMatDef{}),
      interior_mat,
    }
  }
}

impl VtraceObj for SpaceObj {
  fn intersect_bwd(&self, out_ray: Ray, epsilon: f32) -> Option<(Vector, f32)> {
    None
  }

  fn normal_at(&self, x: Vector) -> Option<Vector> {
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
  interior_mat: Rc<dyn VolumeMat>,
}

impl VtraceObj for SphereObj {
  fn intersect_bwd(&self, out_ray: Ray, epsilon: f32) -> Option<(Vector, f32)> {
    // TODO
    unimplemented!();
  }

  fn normal_at(&self, x: Vector) -> Option<Vector> {
    // TODO
    unimplemented!();
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

pub struct QuadObj {
  // TODO: specify normal.
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

  fn normal_at(&self, x: Vector) -> Option<Vector> {
    // TODO
    unimplemented!();
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
  fn query_surf_rad_bwd(&self, out_ray: Ray, inc_obj_id: Option<usize>, ext_obj_id: Option<usize>, default_opts: QueryOpts) -> f32;
  fn query_vol_rad_bwd(&self, out_ray: Ray, obj_id: Option<usize>, top_level: bool, /*top_level_opts: Option<QueryOpts>,*/ default_opts: QueryOpts) -> f32;
  fn render_depth(&self, render_opts: RenderOpts, buf: &mut MemArray3d<u8>);
  fn render_rad(&self, render_opts: RenderOpts, buf: &mut MemArray3d<u8>);
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
    // TODO
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

  fn query_surf_rad_bwd(&self, out_ray: Ray, inc_obj_id: Option<usize>, ext_obj_id: Option<usize>, default_opts: QueryOpts) -> f32 {
    // TODO
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
    let interface = inc_vol.interface_at(&*ext_vol, out_ray.origin);
    let normal_dir = match (inc_obj.normal_at(out_ray.origin), ext_obj.normal_at(out_ray.origin)) {
      (None,                None)               => panic!(),
      (Some(inc_normal),    None)               => -inc_normal,
      (None,                Some(ext_normal))   => ext_normal,
      (Some(inc_normal),    Some(ext_normal))   => 0.5 * (-inc_normal + ext_normal),
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
        let recurs_mc_est = match interface.scatter_surf_bwd(out_ray.dir, normal_dir) {
          InterfaceEvent::Absorb => {
            0.0
          }
          InterfaceEvent::Reflect(in_dir) => {
            let in_ray = Ray{origin: out_ray.origin, dir: in_dir};
            let next_est_rad = self.query_vol_rad_bwd(in_ray, inc_obj_id, false, default_opts);
            next_est_rad
          }
          InterfaceEvent::Transmit(in_dir) => {
            let in_ray = Ray{origin: out_ray.origin, dir: in_dir};
            let next_est_rad = self.query_vol_rad_bwd(in_ray, ext_obj_id, false, default_opts);
            next_est_rad
          }
        };
        let total_mc_norm = roulette_mc_norm.unwrap_or(1.0);
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
        0.0
      }
    };
    let this_rad = emit_rad + mc_est_rad;
    this_rad
  }

  fn query_vol_rad_bwd(&self, out_dst_ray: Ray, vol_obj_id: Option<usize>, top_level: bool, default_opts: QueryOpts) -> f32 {
    // TODO
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
            (0.0, None)
          }
          AttenuationEvent::Cutoff(..) => {
            let out_src_ray = Ray{origin: surf_pt.unwrap(), dir: out_dst_ray.dir};
            let next_est_rad = self.query_surf_rad_bwd(out_src_ray, vol_obj_id, surf_obj_id, default_opts);
            (next_est_rad, None)
          }
          AttenuationEvent::Attenuate(p, dist) => {
            let out_src_ray = Ray{origin: p, dir: out_dst_ray.dir};
            let next_est_rad = match this_vol.scatter_vol_bwd(out_src_ray) {
              ScatterEvent::Absorb => {
                0.0
              }
              ScatterEvent::Scatter(in_ray) => {
                self.query_vol_rad_bwd(in_ray, vol_obj_id, false, default_opts)
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
        0.0
      }
    };
    let this_rad = emit_rad + mc_est_rad;
    this_rad
  }

  fn render_depth(&self, render_opts: RenderOpts, buf: &mut MemArray3d<u8>) {
    // TODO
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

    // The camera relative coords are specified so that "z" is towards the
    // viewer, and "x" and "y" follow the geometric convention on screen.
    let camera_dir = (render_opts.cam_lookat - render_opts.cam_origin).normalize();
    let camera_reldir = Vector::new(0.0, 0.0, -1.0);
    let camera_relup = Vector::new(0.0, 1.0, 0.0);
    let dir_rd2a_tfm = Quaternion::from_arc(camera_reldir, camera_dir, None);
    let dir_a2rd_tfm = Quaternion::from_arc(camera_dir, camera_reldir, None);
    let camera_relup_rd = dir_a2rd_tfm.rotate_vector(render_opts.cam_up).normalize();
    let up_r2rd_tfm = Quaternion::from_arc(camera_relup, camera_relup_rd, None);
    let rel_to_world = up_r2rd_tfm * dir_rd2a_tfm;

    let camera_inc_u = camera_width / render_opts.im_width as f32;
    let camera_inc_v = camera_height / render_opts.im_height as f32;
    let epsilon = 1.0e-6;

    for screen_v in 0 .. render_opts.im_height {
      for screen_u in 0 .. render_opts.im_width {
        // The ray is defined to point toward the camera.
        let camera_u = -0.5 * camera_width + (0.5 + screen_u as f32) * camera_inc_u;
        let camera_v = -0.5 * camera_height + (0.5 + screen_v as f32) * camera_inc_v;
        let camera_w = camera_depth;
        let camera_relp = Vector3{x: camera_u, y: camera_v, z: camera_w}.normalize();
        let camera_p = rel_to_world.rotate_vector(camera_relp).normalize();
        let out_ray = Ray{origin: render_opts.cam_origin, dir: camera_p};
        let depth = match self.trace_bwd(out_ray, Some(0), epsilon) {
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

  fn render_rad(&self, render_opts: RenderOpts, buf: &mut MemArray3d<u8>) {
    // TODO
    unimplemented!();
  }
}
