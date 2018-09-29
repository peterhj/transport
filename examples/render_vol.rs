extern crate cgmath;
extern crate memarray;
extern crate stb_image;
extern crate transport;

use cgmath::*;
use memarray::*;
use stb_image::image::{Image};
//use transport::geometry::*;
use transport::vtrace::*;
//use transport::utils::*;

use std::fs::{File};
use std::io::{Write};
use std::path::{PathBuf};
use std::rc::{Rc};

fn main() {
  let root_obj = Rc::new(SpaceObj::new(Rc::new(VacuumVolumeMatDef::default())));
  //let root_obj = Rc::new(SpaceObj::new(Rc::new(HomogeneousDielectricVolumeMatDef::default_hg(0.001, 0.1))));
  let light_obj = Rc::new(QuadObj::new(vec![
      Vector3::new(-1.0, 1.58, -1.0),
      Vector3::new(-1.0, 1.58, 1.0),
      Vector3::new(1.0, 1.58, -1.0),
      Vector3::new(1.0, 1.58, 1.0),
      /*Vector3::new(-0.24, 1.58, -0.22),
      Vector3::new(-0.24, 1.58, 0.16),
      Vector3::new(0.23, 1.58, -0.22),
      Vector3::new(0.23, 1.58, 0.16),*/
  ], Rc::new(SphericalLightSurfaceMatDef{emit_rad: 1.0})));
  //], Rc::new(HemisphericalLightSurfaceMatDef{emit_rad: 1.0})));
  let floor_obj = Rc::new(QuadObj::new(vec![
      /*Vector3::new(-2.0, 0.0, -2.0),
      Vector3::new(-2.0, 0.0, 2.0),
      Vector3::new(2.0, 0.0, -2.0),
      Vector3::new(2.0, 0.0, 2.0),*/
      Vector3::new(-0.99, 0.0, -1.04),
      Vector3::new(-1.01, 0.0, 0.99),
      Vector3::new(1.0, 0.0, -1.04),
      Vector3::new(1.0, 0.0, 0.99),
  ], Rc::new(LambertianSurfaceMatDef{absorb_prob: 0.0})));
  //], Rc::new(MirrorSurfaceMatDef{absorb_prob: 0.0})));
  let leftwall_obj = Rc::new(QuadObj::new(vec![
      Vector3::new(-1.02, 1.59, -1.04),
      Vector3::new(-1.02, 1.59, 0.99),
      Vector3::new(-1.01, 0.00, -1.04),
      Vector3::new(-1.01, 0.00, 0.99),
  ], Rc::new(InvisibleSurfaceMatDef::default())));
  let rightwall_obj = Rc::new(QuadObj::new(vec![
      Vector3::new(1.0, 1.59, -1.04),
      Vector3::new(1.0, 1.59, 0.99),
      Vector3::new(1.0, 0.00, -1.04),
      Vector3::new(1.0, 0.00, 0.99),
  ], Rc::new(InvisibleSurfaceMatDef::default())));
  /*let ceiling_obj = Rc::new(QuadObj::new(vec![
  ], Rc::new()));*/
  let sphere1_obj = Rc::new(SphereObj::new(
      Vector3::new(-0.42, 0.33, -0.28), 0.33,
      Rc::new(MirrorSurfaceMatDef{absorb_prob: 0.0})));
  let sphere2_obj = Rc::new(SphereObj::new(
      Vector3::new(0.45, 0.33, 0.38), 0.33,
      Rc::new(MirrorSurfaceMatDef{absorb_prob: 0.0})));

  let mut scene = SimpleVtraceScene::new(root_obj);
  scene.add_object(light_obj);
  scene.add_object(floor_obj);
  //scene.add_object(leftwall_obj);
  //scene.add_object(rightwall_obj);
  //scene.add_object(ceiling_obj);
  scene.add_object(sphere1_obj);
  scene.add_object(sphere2_obj);

  let query_opts = QueryOpts{
    trace_epsilon:      1.0e-6,
    importance_clip:    None,
    //roulette_term_p:    None,
    //roulette_term_p:    Some(0.01),
    roulette_term_p:    Some(0.1),
  };

  //let im_dim = 64;
  //let im_dim = 320;
  let im_dim = 640;

  let render_opts = RenderOpts{
    cam_origin: Vector3::new(0.0, 0.92, 5.4),
    cam_lookat: Vector3::new(0.0, 0.89, 4.4),
    cam_up:     Vector3::new(0.0, 1.0, 0.0),
    //cam_fov:    Fov::Tangent(4.0),
    cam_fov:    Fov::Tangent(0.25),
    im_width:   im_dim,
    im_height:  im_dim,
    //rays_per_pix:   1,
    //rays_per_pix:   2,
    //rays_per_pix:   10,
    //rays_per_pix:   100,
    rays_per_pix:   1000,
    //rays_per_pix:   10000,
  };

  let mut buf = MemArray3d::<u8>::zeros([3, im_dim, im_dim]);

  //scene.render_depth(render_opts, &mut buf);
  scene.render_rad(query_opts, render_opts, &mut buf);

  let img_buf: Vec<_> = buf.flat_view().unwrap().as_slice().to_owned();
  let img = Image::new(im_dim, im_dim, 3, img_buf);
  let png_data = img.write_png().unwrap();
  let mut png_file = File::create(&PathBuf::from("out.png")).unwrap();
  png_file.write_all(&png_data).unwrap();
}
