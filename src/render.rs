use ::geometry::*;

use cgmath::*;
use float::ord::*;
use memarray::*;

use std::f32;

pub trait Render<Cfg> {
  fn render(&self, cfg: &Cfg, buf: &mut MemArray3d<u8>);
}

pub struct NoFrillsDepthRender {
  pub cam_orig:     Vector3<f32>,
  pub cam_lookat:   Vector3<f32>,
  pub cam_width:    f32,
  pub cam_depth:    f32,
  pub width:        usize,
  pub height:       usize,
}

fn simple_depth_render<Scene: IntersectsRay>(scene: &Scene, cfg: &NoFrillsDepthRender, buf: &mut MemArray3d<u8>) {
  let mut flat_buf = buf.flat_view_mut().unwrap();
  let flat_buf = flat_buf.as_mut_slice();
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
  pub cam_width:    f32,
  pub cam_depth:    f32,
  pub width:        usize,
  pub height:       usize,
  pub rays_per_pix: usize,
  pub terminal_p:   f32,
}

fn roulette_render<Scene: TraceRay>(scene: &Scene, cfg: &RouletteRender, buf: &mut MemArray3d<u8>) {
  let mut flat_buf = buf.flat_view_mut().unwrap();
  let flat_buf = flat_buf.as_mut_slice();
  let aspect_ratio = cfg.height as f32 / cfg.width as f32;
  let camera_width = cfg.cam_width;
  let camera_height = aspect_ratio * camera_width;
  let camera_dir = (cfg.cam_lookat - cfg.cam_orig).normalize();
  let camera_reldir = Vector3::new(0.0, 0.0, 1.0);
  let rel_to_world = Quaternion::from_arc(camera_reldir, camera_dir, None);
  let camera_orig = cfg.cam_orig;
  //let camera_fwd = camera_orig + camera_depth * camera_dir;
  // TODO: define camera plane.
  let camera_inc_u = camera_width / cfg.width as f32;
  let camera_inc_v = camera_height / cfg.height as f32;
  /*let camera_u = -0.5 * camera_width + (0.5 + screen_u as f32) * camera_inc_u;
  let camera_v = -0.5 * camera_height + (0.5 + screen_v as f32) * camera_inc_v;
  let camera_w = cfg.cam_depth;
  let camera_relp = Vector3{x: camera_u, y: camera_v, z: camera_w}.normalize();
  let camera_p = rel_to_world.rotate_vector(camera_relp).normalize();
  let ray = Ray{orig: camera_orig, dir: camera_p};*/
  loop {
    /* TODO:
        - loop:
        - seed a ray from light, hit something in scene?
        - depending on material of the hit, reflect, or transmit
          (possibly account for refraction)
          - save "previous" material for interface conditions
        - note: ray may terminate before hit?
    */
    /*let depth = match scene.intersects_ray(&ray, 1.0e-6) {
      None => 1.0 / 0.0,
      Some(ixn) => ixn.ray_coord.t,
    };*/
    unimplemented!();
  }
}
