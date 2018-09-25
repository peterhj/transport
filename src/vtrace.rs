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

pub trait Interface {
  fn scatter_bwd(&self, out_ray: Ray) -> InterfaceEvent;
}

pub trait SurfaceMat {
  fn emit_surf(&self, out_ray: Ray) -> f32;
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

pub trait VolumeMat {
  fn absorbing_coef_at(&self, x: Vector) -> f32;
  fn scattering_coef_at(&self, x: Vector) -> f32;
  fn scatter_bwd(&self, out_ray: Ray) -> ScatterEvent;

  fn extinction_coef_at(&self, x: Vector) -> f32 {
    self.absorbing_coef_at(x) + self.scattering_coef_at(x)
  }

  fn max_extinction_coef(&self) -> Option<f32> {
    None
  }

  fn woodcock_track_bwd(&self, out_ray: Ray, cutoff_dist: Option<f32>) -> AttenuationEvent {
    let max_coef = match self.max_extinction_coef() {
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
      if u2 * max_coef < self.extinction_coef_at(xp) {
        break;
      }
    }
    AttenuationEvent::Attenuate(xp, s / max_coef)
  }
}

#[derive(Clone)]
pub struct HomogeneousVolumeMatDef {
  pub absorb_coef:  f32,
  pub scatter_coef: f32,
  pub scatter_dist: Option<Rc<dyn HomogeneousScatterDist>>,
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
  pub roulette_term_p:  Option<f32>,
}

pub trait VtraceScene {
  fn trace_bwd(&self, out_ray: Ray, obj_id: Option<usize>) -> TraceEvent;
  fn query_surf_rad_bwd(&self, out_ray: Ray, inc_obj_id: Option<usize>, ext_obj_id: Option<usize>, default_opts: QueryOpts) -> f32;
  fn query_vol_rad_bwd(&self, out_ray: Ray, obj_id: Option<usize>, top_level: bool, default_opts: QueryOpts) -> f32;
}

pub trait VtraceObj {
  fn interior_mat(&self) -> Rc<dyn VolumeMat>;
}

pub struct SimpleVtraceScene {
  default_mat:  Rc<dyn VolumeMat>,
  //lights:       Vec<Rc<dyn VtraceLight>>,
  //objs:         Vec<Rc<dyn VtraceObj>>,
}

impl VtraceScene for SimpleVtraceScene {
  fn trace_bwd(&self, out_ray: Ray, obj_id: Option<usize>) -> TraceEvent {
    // TODO
    unimplemented!();
  }

  fn query_surf_rad_bwd(&self, out_ray: Ray, inc_obj_id: Option<usize>, ext_obj_id: Option<usize>, default_opts: QueryOpts) -> f32 {
    // TODO
    // FIXME: need to project the intersection to the surface;
    // FIXME: determine the current objects.
    let inc_obj = unimplemented!();
    let ext_obj = unimplemented!();
    // FIXME: interface depends on _two_ object mats.
    let interface = unimplemented!();
    let emit_rad = ext_obj.emit_surf(out_ray);
    // TODO: roulette importance sampling.
    let mc_est_rad = match interface.scatter_bwd(out_ray) {
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
    // TODO: roulette importance sampling.
    // FIXME: determine which volume is currently "inside";
    // if `vol_obj_id` is given, use that one, otherwise calculate from scratch.
    let this_obj = self.default_mat.clone();
    // TODO: volume emission.
    let emit_rad = 0.0;
    // TODO: roulette importance sampling.
    let mc_est_rad = match this_obj.woodcock_track_bwd(out_dst_ray, surf_cutoff_dist) {
      AttenuationEvent::NonTerm => {
        0.0
      }
      AttenuationEvent::Cutoff(..) => {
        let out_src_ray = Ray{orig: surf_pt.unwrap(), dir: out_dst_ray.dir};
        let next_est_rad = self.query_surf_rad_bwd(out_src_ray, vol_obj_id, surf_obj_id, default_opts);
        next_est_rad
      }
      AttenuationEvent::Attenuate(p, dist) => {
        let out_src_ray = Ray{orig: p, dir: out_dst_ray.dir};
        let next_est_rad = match this_obj.scatter_bwd(out_src_ray) {
          ScatterEvent::Absorb => {
            0.0
          }
          ScatterEvent::Scatter(in_ray) => {
            self.query_vol_rad_bwd(in_ray, vol_obj_id, false, default_opts)
          }
        };
        next_est_rad
      }
    };
    let this_rad = emit_rad + mc_est_rad;
    this_rad
  }
}
