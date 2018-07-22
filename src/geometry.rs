use cgmath::*;

pub struct Ray {
  pub orig: Vector3<f32>,
  pub dir:  Vector3<f32>,
}

pub struct Triangle {
  pub v0:   Vector3<f32>,
  pub v1:   Vector3<f32>,
  pub v2:   Vector3<f32>,
  pub idxs: [usize; 3],
}

pub struct RayTriangleIntersect {
  pub t:    f32,
  pub u:    f32,
  pub v:    f32,
}

pub fn intersection(ray: &Ray, tri: &Triangle, threshold: f32) -> Option<RayTriangleIntersect> {
  let e1 = tri.v1 - tri.v0;
  let e2 = tri.v2 - tri.v0;
  let pvec = ray.dir.cross(e2);
  let det = e1.dot(pvec);
  if det.abs() <= threshold {
    return None;
  }
  let inv_det = 1.0 / det;
  let tvec = ray.orig - tri.v0;
  let u = tvec.dot(pvec) * inv_det;
  if u < 0.0 || u > 1.0 {
    return None;
  }
  let qvec = tvec.cross(e1);
  let v = ray.dir.dot(qvec) * inv_det;
  if v < 0.0 || u + v > 1.0 {
    return None;
  }
  let t = e2.dot(qvec) * inv_det;
  Some(RayTriangleIntersect{t, u, v})
}

pub fn culling_intersection(ray: &Ray, tri: &Triangle, normal: &Vector3<f32>, threshold: f32) -> Option<RayTriangleIntersect> {
  // TODO
  unimplemented!();
}

pub struct TriMesh {
  pub vertexs:      Vec<Vector3<f32>>,
  pub normals:      Vec<Vector3<f32>>,
  pub triangles:    Vec<Triangle>,
}
