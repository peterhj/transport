use ::geometry::*;

use cgmath::*;

use std::fs::{File};
use std::io::{BufRead, BufReader};
use std::path::{PathBuf};

pub struct WavefrontObj {
  //faces:        Vec<Vec<(usize, Option<usize>)>>,
  faces:        Vec<Vec<usize>>,
  vertexs:      Vec<Vector3<f64>>,
  texcoords:    Vec<Vector2<f64>>,
  normals:      Vec<Vector3<f64>>,
}

impl WavefrontObj {
  pub fn open(path: PathBuf) -> Self {
    let file = File::open(&path).unwrap();
    let reader = BufReader::new(file);
    //let mut vertex_ctr = 0;
    let mut faces = vec![];
    let mut vertexs = vec![];
    let mut normals = vec![];
    let mut texcoords = vec![];
    for line in reader.lines() {
      let line = line.unwrap();
      let toks: Vec<_> = line.split_whitespace().collect();
      match toks[0] {
        "f" => {
          let mut face = vec![];
          for tok in toks[1 .. ].iter() {
            let f_toks: Vec<_> = tok.split('/').collect();
            match f_toks.len() {
              1 => {
                let v_rank: usize = f_toks[0].parse().unwrap();
                assert!(v_rank >= 1);
                face.push(v_rank - 1);
              }
              2 => {
                // TODO
                let v_rank: usize = f_toks[0].parse().unwrap();
                assert!(v_rank >= 1);
                face.push(v_rank - 1);
                /*let vt_rank: usize = f_toks[1].parse().unwrap();
                assert!(vt_rank >= 1);
                face.push((v_rank - 1, Some(vt_rank - 1)));*/
              }
              3 => {
                // TODO
                let v_rank: usize = f_toks[0].parse().unwrap();
                assert!(v_rank >= 1);
                face.push(v_rank - 1);
                /*let vt_rank: usize = f_toks[2].parse().unwrap();
                assert!(vt_rank >= 1);
                face.push((v_rank - 1, Some(vt_rank - 1)));*/
              }
              _ => unimplemented!(),
            }
          }
          faces.push(face);
        }
        "v" => {
          let vx: f64 = toks[1].parse().unwrap();
          let vy: f64 = toks[2].parse().unwrap();
          let vz: f64 = toks[3].parse().unwrap();
          vertexs.push(Vector3::new(vx, vy, vz));
        }
        "vn" => {
          let nx: f64 = toks[1].parse().unwrap();
          let ny: f64 = toks[2].parse().unwrap();
          let nz: f64 = toks[3].parse().unwrap();
          normals.push(Vector3::new(nx, ny, nz));
        }
        "vt" => {
          let tx: f64 = toks[1].parse().unwrap();
          let ty: f64 = toks[2].parse().unwrap();
          texcoords.push(Vector2::new(tx, ty));
        }
        _ => {}
      }
    }
    WavefrontObj{
      faces,
      vertexs,
      normals,
      texcoords,
    }
  }

  pub fn polys(&self) -> Vec<Vec<Vector3<f64>>> {
    let &WavefrontObj{ref faces, ref vertexs, ..} = self;
    let mut polys = vec![];
    for f in faces.iter() {
      let mut fvs = vec![];
      for &fv in f.iter() {
        fvs.push(vertexs[fv]);
      }
      polys.push(fvs);
    }
    polys
  }

  pub fn to_mesh(&self) -> TriMesh {
    let vertexs: Vec<_> = self.vertexs.iter().map(|v| v.cast().unwrap()).collect();
    let normals: Vec<_> = self.normals.iter().map(|v| v.cast().unwrap()).collect();
    let polys = self.polys();
    let mut triangles = Vec::with_capacity(polys.len());
    for poly in polys.iter() {
      assert_eq!(3, poly.len());
      triangles.push(Triangle{
        v0: poly[0].cast().unwrap(),
        v1: poly[1].cast().unwrap(),
        v2: poly[2].cast().unwrap(),
      });
    }
    println!("DEBUG: mesh: num vertexs: {}", vertexs.len());
    println!("DEBUG: mesh: num triangles: {}", triangles.len());
    TriMesh{
      vertexs, normals, triangles,
    }
  }
}
