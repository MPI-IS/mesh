// A variant of CGAL/internal/AABB_tree/nearest_point_triangle_3.h
// it differs in that it records which part of the triangle (interior, edge, or vertex)
// the query point is closest to

#ifndef NEAREST_POINT_TRIANGLE_3_INTERIOR_EDGE_VERTEX_H_
#define NEAREST_POINT_TRIANGLE_3_INTERIOR_EDGE_VERTEX_H_

#include <CGAL/kernel_basic.h>
#include <CGAL/enum.h>


namespace CGAL {
namespace iev {
// tests whether a query point q (assumed to be in the plane of the triangle)
// is inside or outside a triangle edge (p1,p2)
// returns true iff q is outside
// q is inside (p1,p2) if: 
//  q is on the correct side of the line through (p1,p2)
//  q's projection on this line lies between p1 and p2
// if q is outside the edge (and therefore the triangle) but projects between p1 and p2
// set outside=true and result=the projection of q onto the line through p1 and p2
template <class K>
inline
bool
is_inside_triangle_3_aux(const typename K::Vector_3& w, // scaled triangle normal (b-a) x (c-b)
                         const typename K::Point_3& p1, 
                         const typename K::Point_3& p2,
                         const typename K::Point_3& q, // query point
                         typename K::Point_3& result,
                         bool& outside,
                         const K& k)
{
  typedef typename K::Vector_3 Vector_3;
  typedef typename K::FT FT;

  typename K::Construct_vector_3 vector =
    k.construct_vector_3_object();
  typename K::Construct_projected_point_3 projection =
    k.construct_projected_point_3_object();
  typename K::Construct_line_3 line =
    k.construct_line_3_object();
  typename K::Compute_scalar_product_3 scalar_product =
    k.compute_scalar_product_3_object();
  typename K::Construct_cross_product_vector_3 cross_product =
    k.construct_cross_product_vector_3_object();

  const Vector_3 v = cross_product(vector(p1,p2), vector(p1,q));
  if ( scalar_product(v,w) < FT(0))
  {
    if (   scalar_product(vector(p1,q), vector(p1,p2)) >= FT(0)
        && scalar_product(vector(p2,q), vector(p2,p1)) >= FT(0) )
    {
      result = projection(line(p1, p2), q);
      return true;
    }
    outside = true;
  }

  return false;
}


/**
 * Returns the nearest point of p1,p2,p3 from origin
 * @param origin the origin point
 * @param p1 the first point
 * @param p2 the second point
 * @param p3 the third point
 * @param k the kernel
 * @return the nearest point from origin
 */
template <class K>
inline
typename K::Point_3
nearest_point_3(const typename K::Point_3& origin,
                const typename K::Point_3& p1,
                const typename K::Point_3& p2,
                const typename K::Point_3& p3,
                const K& k)
{
  typedef typename K::FT FT;

  typename K::Compute_squared_distance_3 sq_distance =
    k.compute_squared_distance_3_object();

  const FT dist_origin_p1 = sq_distance(origin,p1);
  const FT dist_origin_p2 = sq_distance(origin,p2);
  const FT dist_origin_p3 = sq_distance(origin,p3);

  if (   dist_origin_p2 >= dist_origin_p1
      && dist_origin_p3 >= dist_origin_p1 )
  {
    return p1;
  }
  if ( dist_origin_p3 >= dist_origin_p2 )
  {
    return p2;
  }

  return p3;
}

/**
 * @brief returns true if p is inside triangle t. If p is not inside t,
 * result is the nearest point of t from p. WARNING: it is assumed that
 * t and p are on the same plane.
 * @param p the reference point
 * @param t the triangle
 * @param result if p is not inside t, the nearest point of t from p
 * @param k the kernel
 * @return true if p is inside t
 */
template <class K>
inline
int
nearest_primitive(const typename K::Point_3& origin,
                     const typename K::Triangle_3& t,
                     typename K::Point_3& result,
                     const K& k)
{
  typedef typename K::Point_3 Point_3;
  typedef typename K::Vector_3 Vector_3;

  typename K::Construct_vector_3 vector =
    k.construct_vector_3_object();
  typename K::Construct_vertex_3 vertex_on =
    k.construct_vertex_3_object();
  typename K::Construct_cross_product_vector_3 cross_product =
    k.construct_cross_product_vector_3_object();
  typename K::Construct_supporting_plane_3 supporting_plane =
    k.construct_supporting_plane_3_object();
  typename K::Construct_projected_point_3 projection =
    k.construct_projected_point_3_object();

  const Point_3 p = projection(supporting_plane(t), origin);
  const Point_3& t0 = vertex_on(t,0);
  const Point_3& t1 = vertex_on(t,1);
  const Point_3& t2 = vertex_on(t,2);

  Vector_3 w = cross_product(vector(t0,t1), vector(t1,t2));

  bool outside = false;
  if(is_inside_triangle_3_aux(w, t0, t1, p, result, outside, k)) { return 1; }
  if(is_inside_triangle_3_aux(w, t1, t2, p, result, outside, k)) { return 2; }
  if(is_inside_triangle_3_aux(w, t2, t0, p, result, outside, k)) { return 3; }
  if(outside) {
    result = nearest_point_3(p,t0,t1,t2,k);
    if(result == t0) { return 4; }
    if(result == t1) { return 5; }
    if(result == t2) { return 6; }
  }

  return 0;
}

} // end namespace iev
} // end namespace CGAL


#endif // NEAREST_POINT_TRIANGLE_3_H_
