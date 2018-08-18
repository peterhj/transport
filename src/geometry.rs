use cgmath::*;

pub struct Parametric<T> {
  pub t:    T,
}

pub struct Barycentric2<T> {
  pub u:    T,
  pub v:    T,
}

pub struct Ray {
  pub orig: Vector3<f32>,
  pub dir:  Vector3<f32>,
}

pub trait IntersectsRay {
  type Intersection;

  fn intersects_ray(&self, ray: &Ray, threshold: f32) -> Option<Self::Intersection>;
}

pub struct Sphere {
  pub center:   Vector3<f32>,
  pub radius:   f32,
}

pub struct SphereRayIntersection {
  pub ray_coord:    Parametric<f32>,
}

impl IntersectsRay for Sphere {
  type Intersection = SphereRayIntersection;

  fn intersects_ray(&self, ray: &Ray, threshold: f32) -> Option<Self::Intersection> {
    let delta = ray.orig - self.center;
    let b = ray.dir.dot(delta);
    let determinant = b * b + self.radius * self.radius - delta.dot(delta);
    if determinant < -threshold {
      None
    } else if determinant.abs() <= threshold {
      let t = -b;
      Some(SphereRayIntersection{ray_coord: Parametric{t}})
    } else if determinant > threshold {
      let t1 = -b - determinant.sqrt();
      let t2 = -b + determinant.sqrt();
      let t = match (t1 < 0.0, t2 < 0.0) {
        (false, false)  => t1.min(t2),
        (false, true)   => t1,
        (true,  false)  => t2,
        (true,  true)   => return None,
      };
      Some(SphereRayIntersection{ray_coord: Parametric{t}})
    } else {
      unreachable!();
    }
  }
}

pub struct Triangle {
  pub v0:   Vector3<f32>,
  pub v1:   Vector3<f32>,
  pub v2:   Vector3<f32>,
  //pub idxs: [usize; 3],
}

pub struct RayTriangleIntersect {
  pub ray_coord:    Parametric<f32>,
  pub tri_coords:   Barycentric2<f32>,
}

impl IntersectsRay for Triangle {
  type Intersection = RayTriangleIntersect;

  fn intersects_ray(&self, ray: &Ray, threshold: f32) -> Option<Self::Intersection> {
    // TODO
    intersection(ray, self, threshold)
  }
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
  Some(RayTriangleIntersect{
    ray_coord:  Parametric{t},
    tri_coords: Barycentric2{u, v},
  })
}

pub struct TriMesh {
  pub vertexs:      Vec<Vector3<f32>>,
  pub normals:      Vec<Vector3<f32>>,
  pub triangles:    Vec<Triangle>,
}
