use ::geometry::*;
use ::trace::*;

use cgmath::*;
use float::ord::*;
use memarray::*;

use rand::prelude::*;
use rand::distributions::{Uniform};
use std::f32;

#[derive(Clone, Copy)]
pub enum Fov {
  Degrees(f32),
  HalfTangent(f32),
}

pub trait Render<Cfg> {
  fn render(&self, cfg: &Cfg, buf: &mut MemArray3d<u8>);
}

pub struct NoFrillsDepthRender {
  pub cam_orig:     Vector3<f32>,
  pub cam_lookat:   Vector3<f32>,
  pub cam_width:    f32,
  pub cam_depth:    f32,
  //pub cam_fov:      Fov,
  pub width:        usize,
  pub height:       usize,
}

fn simple_depth_render<Scene: IntersectsRay>(scene: &Scene, cfg: &NoFrillsDepthRender, buf: &mut MemArray3d<u8>) {
  let mut flat_buf = buf.flat_view_mut().unwrap();
  let flat_buf = flat_buf.as_mut_slice();
  // FIXME: this is really a pinhole camera model, the camera dims are a
  // fiction.
  let aspect_ratio = cfg.height as f32 / cfg.width as f32;
  let camera_width = cfg.cam_width;
  let camera_height = aspect_ratio * camera_width;
  let camera_dir = (cfg.cam_lookat - cfg.cam_orig).normalize();
  let camera_reldir = Vector3::new(0.0, 0.0, 1.0);
  let rel_to_world = Quaternion::from_arc(camera_reldir, camera_dir, None);
  let camera_orig = cfg.cam_orig;
  //let camera_fwd = camera_orig + camera_depth * camera_dir;
  let camera_inc_u = camera_width / cfg.width as f32;
  let camera_inc_v = camera_height / cfg.height as f32;
  for screen_v in 0 .. cfg.height {
    for screen_u in 0 .. cfg.width {
      let camera_u = -0.5 * camera_width + (0.5 + screen_u as f32) * camera_inc_u;
      let camera_v = -0.5 * camera_height + (0.5 + screen_v as f32) * camera_inc_v;
      let camera_w = cfg.cam_depth;
      let camera_relp = Vector3{x: camera_u, y: camera_v, z: camera_w}.normalize();
      let camera_p = rel_to_world.rotate_vector(camera_relp).normalize();
      let ray = Ray{orig: camera_orig, dir: camera_p};
      let depth = match scene.intersects_ray(&ray, 1.0e-6) {
        None => 1.0 / 0.0,
        Some(ixn) => ixn.ray_coord.t,
      };
      let pix_soft_val = (depth).atan() * f32::consts::FRAC_2_PI;
      let pix_val: u8 = ((1.0 - pix_soft_val.max(0.0).min(1.0).powf(2.2)) * 255.0).round() as u8;
      flat_buf[0 + 3 * (screen_u + cfg.width * screen_v)] = pix_val;
      flat_buf[1 + 3 * (screen_u + cfg.width * screen_v)] = pix_val;
      flat_buf[2 + 3 * (screen_u + cfg.width * screen_v)] = pix_val;
    }
  }
}

impl Render<NoFrillsDepthRender> for SimpleMeshScene {
  fn render(&self, cfg: &NoFrillsDepthRender, buf: &mut MemArray3d<u8>) {
    simple_depth_render(self, cfg, buf);
  }
}

impl Render<NoFrillsDepthRender> for SimpleScene {
  fn render(&self, cfg: &NoFrillsDepthRender, buf: &mut MemArray3d<u8>) {
    simple_depth_render(self, cfg, buf);
  }
}

pub struct RouletteRender {
  pub cam_orig:     Vector3<f32>,
  pub cam_lookat:   Vector3<f32>,
  pub cam_fov:      Fov,
  pub source_mat:   SourceMatDef,
  pub width:        usize,
  pub height:       usize,
  pub rays_per_pix: usize,
  pub early_term_p: f32,
}

fn backward_roulette_render<Scene: TraceRay>(scene: &Scene, cfg: &RouletteRender, buf: &mut MemArray3d<u8>) {
  // TODO: this is the standard path tracing algo.
  let mut flat_buf = buf.flat_view_mut().unwrap();
  let flat_buf = flat_buf.as_mut_slice();

  let aspect_ratio = cfg.height as f32 / cfg.width as f32;
  let camera_width = 1.0;
  let camera_height = aspect_ratio;
  let camera_depth = match cfg.cam_fov {
    Fov::Degrees(_) => unimplemented!(),
    Fov::HalfTangent(t) => 0.5 * camera_width / t,
  };
  let camera_dir = (cfg.cam_lookat - cfg.cam_orig).normalize();
  let camera_reldir = Vector3::new(0.0, 0.0, 1.0);
  let rel_to_world = Quaternion::from_arc(camera_reldir, camera_dir, None);
  let camera_orig = cfg.cam_orig;
  //let camera_fwd = camera_orig + camera_depth * camera_dir;

  let camera_inc_u = camera_width / cfg.width as f32;
  let camera_inc_v = camera_height / cfg.height as f32;
  // TODO: multi channel radiance estimates.
  let half_uniform = Uniform::new_inclusive(-0.5, 0.5);
  let epsilon = 1.0e-6;

  let mut rng = thread_rng();
  let mut rad_ests = Vec::with_capacity(cfg.rays_per_pix);
  for screen_vc in 0 .. cfg.height {
    for screen_uc in 0 .. cfg.width {
      rad_ests.clear();
      for _ in 0 .. cfg.rays_per_pix {
        let mut ray_rad_est = None;
        loop {
          // TODO: choose noisy ray direction.
          let screen_u = screen_uc as f32 + rng.sample(half_uniform);
          let screen_v = screen_vc as f32 + rng.sample(half_uniform);
          let camera_u = -0.5 * camera_width + (0.5 + screen_u as f32) * camera_inc_u;
          let camera_v = -0.5 * camera_height + (0.5 + screen_v as f32) * camera_inc_v;
          let camera_w = camera_depth;
          let camera_relp = Vector3{x: camera_u, y: camera_v, z: camera_w}.normalize();
          let camera_p = rel_to_world.rotate_vector(camera_relp).normalize();
          let ray = Ray{orig: camera_orig, dir: camera_p};
          match scene.trace_ray(&ray, epsilon) {
            None => {
              ray_rad_est = Some(0.0);
              break;
            }
            Some((ixn, hit)) => {
              // TODO
              unimplemented!();
            }
          }
        }
        // TODO: update radiance estimates.
        rad_ests.push(ray_rad_est.unwrap());
      }
    }
  }

  // TODO: write pixels.
  // TODO: need to do sRGB conversion?
  unimplemented!();
}

fn forward_roulette_render<Scene: TraceRay>(scene: &Scene, cfg: &RouletteRender, buf: &mut MemArray3d<u8>) {
  // TODO: this is "photon tracing".
  let mut flat_buf = buf.flat_view_mut().unwrap();
  let flat_buf = flat_buf.as_mut_slice();

  let aspect_ratio = cfg.height as f32 / cfg.width as f32;
  let camera_width = 1.0;
  let camera_height = aspect_ratio;
  let camera_depth = match cfg.cam_fov {
    Fov::Degrees(_) => unimplemented!(),
    Fov::HalfTangent(t) => 0.5 * camera_width / t,
  };
  let camera_dir = (cfg.cam_lookat - cfg.cam_orig).normalize();
  let camera_reldir = Vector3::new(0.0, 0.0, 1.0);
  let rel_to_world = Quaternion::from_arc(camera_reldir, camera_dir, None);
  let camera_orig = cfg.cam_orig;
  //let camera_fwd = camera_orig + camera_depth * camera_dir;

  // Define the camera quadrilateral.
  let camera_u0 = -0.5 * camera_width;
  let camera_u1 = 0.5 * camera_width;
  let camera_v0 = -0.5 * camera_height;
  let camera_v1 = 0.5 * camera_height;
  let camera_w = camera_depth;
  let camera_relp0 = Vector3::new(camera_u0, camera_v0, camera_w);
  let camera_relp1 = Vector3::new(camera_u0, camera_v1, camera_w);
  let camera_relp2 = Vector3::new(camera_u1, camera_v0, camera_w);
  let camera_relp3 = Vector3::new(camera_u1, camera_v1, camera_w);
  let camera_p0 = camera_orig + rel_to_world.rotate_vector(camera_relp0);
  let camera_p1 = camera_orig + rel_to_world.rotate_vector(camera_relp1);
  let camera_p2 = camera_orig + rel_to_world.rotate_vector(camera_relp2);
  let camera_p3 = camera_orig + rel_to_world.rotate_vector(camera_relp3);
  let camera_quad = Quadrilateral{
    v0: camera_p0,
    v1: camera_p1,
    v2: camera_p2,
    v3: camera_p3,
  };

  let mut rng = thread_rng();
  let uniform = Uniform::new(0.0, 1.0);
  loop {
    // FIXME: initialize the "ray state" (consisting of magnitude, ray).
    let mut state = None;
    //let mut node = RayTraceNode::Air(init_origin, ContinuumMatDef::default_vacuum());
    let mut node = RayTraceNode::Source(cfg.source_mat.clone());
    let mut prev_node = None;
    loop {
      /* TODO:
          - loop:
          - seed a ray from light, hit something in scene?
          - depending on material of the hit, reflect, or transmit
            (possibly account for refraction)
            - save "previous" material for interface conditions
          - note: ray may terminate before hit?
      */

      match (prev_node, node) {
        (None, RayTraceNode::Source(ref source_mat)) => {
          // TODO: sample a light point to seed the ray.
          let (src_mag, src_ray) = source_mat.source_dist.sample_event();
          state = Some(RayTraceState{
            mag: src_mag, ray: src_ray,
          });
          unimplemented!();
        }
        (Some(RayTraceNode::Source(..)), RayTraceNode::Air(..)) => {
          // TODO
          unimplemented!();
        }
        (Some(RayTraceNode::Air(..)), RayTraceNode::AirInterface(..)) => {
          // TODO
          unimplemented!();
        }
        _ => unimplemented!(),
      }

      // TODO: intersection test based on the current ray state.
      match scene.intersects_ray(&state.unwrap().ray, 1.0e-6) {
      //match scene.trace_ray(&state.unwrap().ray, 1.0e-6) {
        None => {
          // TODO: this ray shoots off into space, contributes nothing to image;
          // terminate the path.
        }
        Some(_) => {
          // TODO
        }
      }

      // TODO
      /*let depth = match scene.intersects_ray(&ray, 1.0e-6) {
        None => 1.0 / 0.0,
        Some(ixn) => ixn.ray_coord.t,
      };*/
      let early_term_u = rng.sample(uniform);
      if early_term_u < cfg.early_term_p {
        // TODO
      }

      // TODO: update nodes.
      //let prev_node = node;
      //let node = next_node;

      unimplemented!();
    }
  }
}
