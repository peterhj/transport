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
  let light_obj = Rc::new(QuadObj::new(vec![
      Vector3::new(-0.24, 1.58, -0.22),
      Vector3::new(-0.24, 1.58, 0.16),
      Vector3::new(0.23, 1.58, -0.22),
      Vector3::new(0.23, 1.58, 0.16),
  ], Rc::new(HemisphericLightSurfaceMatDef{emit_rad: 1.0})));

  let mut scene = SimpleVtraceScene::new(root_obj);
  scene.add_object(light_obj);

  let render_opts = RenderOpts{
    cam_origin: Vector3::new(0.0, 1.0, 5.0),
    cam_lookat: Vector3::new(0.0, 1.0, 4.0),
    cam_up:     Vector3::new(0.0, 1.0, 0.0),
    cam_fov:    Fov::Tangent(0.25),
    im_width:   640,
    im_height:  640,
  };
  let mut buf = MemArray3d::<u8>::zeros([3, 640, 640]);
  scene.render_depth(render_opts, &mut buf);
  let img_buf: Vec<_> = buf.flat_view().unwrap().as_slice().to_owned();
  let img = Image::new(640, 640, 3, img_buf);
  let png_data = img.write_png().unwrap();
  let mut png_file = File::create(&PathBuf::from("out.png")).unwrap();
  png_file.write_all(&png_data).unwrap();
}
