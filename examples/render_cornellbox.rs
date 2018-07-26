extern crate cgmath;
extern crate memarray;
extern crate stb_image;
extern crate transport;

use cgmath::*;
use memarray::*;
use stb_image::image::{Image};
use transport::render::*;
use transport::utils::*;

use std::fs::{File};
use std::io::{Write};
use std::path::{PathBuf};

fn main() {
  let sphere1 = WavefrontObj::open(PathBuf::from("data/cbox/meshes/sphere1.obj"));
  let sphere2 = WavefrontObj::open(PathBuf::from("data/cbox/meshes/sphere2.obj"));
  let walls = WavefrontObj::open(PathBuf::from("data/cbox/meshes/walls.obj"));
  let leftwall = WavefrontObj::open(PathBuf::from("data/cbox/meshes/leftwall.obj"));
  let rightwall = WavefrontObj::open(PathBuf::from("data/cbox/meshes/rightwall.obj"));
  let scene = SimpleMeshScene{objs: vec![
      sphere1.to_mesh(),
      sphere2.to_mesh(),
      walls.to_mesh(),
      leftwall.to_mesh(),
      rightwall.to_mesh(),
  ]};
  let render_cfg = NoFrillsDepthRender{
    cam_orig: Vector3::new(0.0, 0.919769, 5.41159),
    cam_lookat: Vector3::new(0.0, 0.893051, 4.41198),
    cam_width: 0.003, cam_depth: 0.0063,
    width: 640, height: 640,
  };
  let mut buf = MemArray3d::<u8>::zeros([3, 640, 640]);
  scene.render(&render_cfg, &mut buf);
  let img_buf: Vec<_> = buf.flat_view().unwrap().as_slice().to_owned();
  let img = Image::new(640, 640, 3, img_buf);
  let png_data = img.write_png().unwrap();
  let mut png_file = File::create(&PathBuf::from("out.png")).unwrap();
  png_file.write_all(&png_data).unwrap();
}
