use ::geometry::*;

use cgmath::*;
use float::ord::*;
use memarray::*;

pub struct SimpleMeshScene {
  pub objs: Vec<TriMesh>,
}

pub trait Renderable<Cfg> {
  fn render(&self, cfg: &Cfg, buf: &mut MemArray3d<u8>);
}

pub struct NoFrillsDepthRender {
  pub width: usize,
  pub height: usize,
}

impl Renderable<NoFrillsDepthRender> for SimpleMeshScene {
  fn render(&self, cfg: &NoFrillsDepthRender, buf: &mut MemArray3d<u8>) {
    let mut flat_buf = buf.flat_view_mut().unwrap();
    let mut flat_buf = flat_buf.as_mut_slice();
    let aspect_ratio = cfg.height as f32 / cfg.width as f32;
    let camera_width: f32 = 0.3;
    let camera_height = aspect_ratio * camera_width;
    let camera_depth: f32 = 0.3;
    let mut camera_dir: Vector3<f32> = Vector3::zero();
    camera_dir.y = 1.0;
    let camera_dir = camera_dir;
    let mut camera_reldir: Vector3<f32> = Vector3::zero();
    camera_reldir.z = 1.0;
    let camera_reldir = camera_reldir;
    let rel_to_world = Quaternion::from_arc(camera_reldir, camera_dir, None);
    let camera_orig: Vector3<f32> = Vector3::zero();
    let camera_fwd = camera_orig + camera_depth * camera_dir;
    let camera_inc_u = camera_width / cfg.width as f32;
    let camera_inc_v = camera_height / cfg.height as f32;
    for screen_v in 0 .. cfg.height {
      for screen_u in 0 .. cfg.width {
        let camera_u = -0.5 * camera_width + (0.5 + screen_u as f32) * camera_inc_u;
        let camera_v = -0.5 * camera_height + (0.5 + screen_v as f32) * camera_inc_v;
        let camera_w = camera_depth;
        let camera_relp = Vector3{x: camera_u, y: camera_v, z: camera_w};
        let camera_p = rel_to_world.rotate_vector(camera_relp);
        let ray = Ray{orig: camera_orig, dir: camera_p - camera_orig};
        let mut ray_ixns = vec![];
        for (obj_idx, obj) in self.objs.iter().enumerate() {
          for (tri_idx, tri) in obj.triangles.iter().enumerate() {
            if let Some(ixn) = intersection(&ray, tri, 1.0e-6) {
              ray_ixns.push((obj_idx, tri_idx, ixn));
            }
          }
        }
        let depth: f32 = if ray_ixns.is_empty() {
          1.0 / 0.0
        } else {
          ray_ixns.sort_unstable_by_key(|(oidx, tidx, ixn)| {
            let t = ixn.ray_coord.t;
            F32SupNan(if t < 0.0 { 1.0 / 0.0 } else { t })
          });
          ray_ixns[0].2.ray_coord.t
        };
        let pix_soft_val = depth.atan();
        let pix_val: u8 = (pix_soft_val.max(0.0).min(1.0) * 255.0).round() as u8;
        flat_buf[0 + 3 * (screen_u + cfg.width * screen_v)] = pix_val;
        flat_buf[1 + 3 * (screen_u + cfg.width * screen_v)] = pix_val;
        flat_buf[2 + 3 * (screen_u + cfg.width * screen_v)] = pix_val;
      }
    }
  }
}
