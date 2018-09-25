use ::geometry::*;

use cgmath::*;
use rand::prelude::*;
use rand::distributions::{OpenClosed01, Standard, StandardNormal, Uniform};

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
pub enum InterfaceEvent {
  Absorb,
  Reflect(Ray),
  Transmit(Ray),
}

pub trait InterfaceMat {
  fn scatter_surf_bwd(&self, out_ray: Ray) -> InterfaceEvent;
}

pub struct DielectricDielectricInterfaceMatDef {
  // TODO
}

impl InterfaceMat for DielectricDielectricInterfaceMatDef {
  fn scatter_surf_bwd(&self, out_ray: Ray) -> InterfaceEvent {
    // TODO
    unimplemented!();
  }
}

pub trait SurfaceMat {
  fn query_surf_emission(&self, out_ray: Ray) -> f32;
}

#[derive(Default)]
pub struct InvisibleSurfaceMatDef {
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

  fn interface_with(&self, other: &VolumeMat) -> Rc<dyn InterfaceMat> {
    match (self.mat_kind(), other.mat_kind()) {
      (VolumeMatKind::Dielectric, VolumeMatKind::Dielectric) => {
        Rc::new(DielectricDielectricInterfaceMatDef{
          // TODO
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
    let mut xp = out_ray.orig;
    let mut s = 0.0;
    loop {
      let u1: f32 = thread_rng().sample(OpenClosed01);
      let u2: f32 = thread_rng().sample(Standard);
      s -= u1.ln();
      xp = out_ray.orig - s * out_ray.dir / max_coef;
      if let Some(cutoff_s) = cutoff_s {
        if s >= cutoff_s {
          return AttenuationEvent::Cutoff(xp, s / max_coef);
        }
      }
      if u2 * max_coef < self.vol_extinction_coef_at(xp) {
        break;
      }
    }
    AttenuationEvent::Attenuate(xp, s / max_coef)
  }
}

#[derive(Clone)]
pub struct HomogeneousDielectricVolumeMatDef {
  //pub mat_kind:     VolumeMatKind,
  pub refractive_index: f32,
  pub absorb_coef:      f32,
  pub scatter_coef:     f32,
  pub scatter_dist:     Option<Rc<dyn HomogeneousScatterDist>>,
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

#[derive(Clone, Copy)]
pub enum TraceEvent {
  NonTerm,
  Surface(Vector, f32, Option<usize>),
}

#[derive(Clone, Copy)]
pub struct QueryOpts {
  pub importance_clip:  Option<f32>,
  pub roulette_term_p:  Option<f32>,
}

pub trait VtraceObj {
  fn boundary_surf_mat(&self) -> Rc<dyn SurfaceMat>;
  fn interior_vol_mat(&self) -> Rc<dyn VolumeMat>;
}

pub trait VtraceScene {
  fn trace_bwd(&self, out_ray: Ray, obj_id: Option<usize>) -> TraceEvent;
  fn query_surf_rad_bwd(&self, out_ray: Ray, inc_obj_id: Option<usize>, ext_obj_id: Option<usize>, default_opts: QueryOpts) -> f32;
  fn query_vol_rad_bwd(&self, out_ray: Ray, obj_id: Option<usize>, top_level: bool, /*top_level_opts: Option<QueryOpts>,*/ default_opts: QueryOpts) -> f32;
}

pub struct SimpleVtraceScene {
  objs: Vec<Rc<dyn VtraceObj>>,
}

impl VtraceScene for SimpleVtraceScene {
  fn trace_bwd(&self, out_ray: Ray, obj_id: Option<usize>) -> TraceEvent {
    // TODO
    unimplemented!();
  }

  fn query_surf_rad_bwd(&self, out_ray: Ray, inc_obj_id: Option<usize>, ext_obj_id: Option<usize>, default_opts: QueryOpts) -> f32 {
    // TODO
    // TODO: project the intersection to the surface;
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
    let interface = inc_vol.interface_with(&*ext_vol);
    let emit_rad = inc_surf.query_surf_emission(out_ray) + ext_surf.query_surf_emission(out_ray);
    let mc_est_rad = {
      let (do_mc, roulette_mc_norm) = if default_opts.roulette_term_p.is_none() {
        (true, None)
      } else {
        let mc_p = 1.0 - default_opts.roulette_term_p.unwrap();
        let mc_u: f32 = thread_rng().sample(Standard);
        (mc_u < mc_p, Some(mc_p))
      };
      if do_mc {
        let recurs_mc_est = match interface.scatter_surf_bwd(out_ray) {
          InterfaceEvent::Absorb => {
            0.0
          }
          InterfaceEvent::Reflect(in_ray) => {
            let next_est_rad = self.query_vol_rad_bwd(in_ray, inc_obj_id, false, default_opts);
            next_est_rad
          }
          InterfaceEvent::Transmit(in_ray) => {
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
    let (surf_cutoff_dist, surf_pt, surf_obj_id) = match self.trace_bwd(out_dst_ray, vol_obj_id) {
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
            let out_src_ray = Ray{orig: surf_pt.unwrap(), dir: out_dst_ray.dir};
            let next_est_rad = self.query_surf_rad_bwd(out_src_ray, vol_obj_id, surf_obj_id, default_opts);
            (next_est_rad, None)
          }
          AttenuationEvent::Attenuate(p, dist) => {
            let out_src_ray = Ray{orig: p, dir: out_dst_ray.dir};
            let next_est_rad = match this_vol.scatter_vol_bwd(out_src_ray) {
              ScatterEvent::Absorb => {
                0.0
              }
              ScatterEvent::Scatter(in_ray) => {
                self.query_vol_rad_bwd(in_ray, vol_obj_id, false, default_opts)
              }
            };
            (next_est_rad, Some(this_vol.vol_extinction_coef_at(out_src_ray.orig)))
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
}
