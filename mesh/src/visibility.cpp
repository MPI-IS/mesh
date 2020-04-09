#define CGAL_CFG_NO_CPP0X_VARIADIC_TEMPLATES 1
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>
#include <CGAL/centroid.h>
#include <CGAL/Simple_cartesian.h>

#include <CGAL/intersections.h>
#include <CGAL/Bbox_3.h>

#include <boost/cstdint.hpp>
#include <boost/array.hpp>

#if HAVE_TBB
#include "tbb/parallel_for.h"
#include "tbb/parallel_for_each.h"
#include "tbb/blocked_range.h"
#include "tbb/task_scheduler_init.h"
#include <boost/iterator/counting_iterator.hpp>
#endif

#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#include "nearest_triangle.hpp"

using boost::uint32_t;
using boost::array;
using std::vector;

typedef CGAL::Simple_cartesian<double> K;
typedef K::Point_3 Point;
typedef K::Segment_3 Segment;
typedef K::Vector_3 Vector;
typedef K::Triangle_3 Triangle;
typedef K::Ray_3 Ray;

typedef std::vector<Triangle>::iterator Iterator;
typedef CGAL::AABB_triangle_primitive<K,Iterator> Primitive;

typedef CGAL::AABB_traits<K, Primitive> AABB_triangle_traits;
typedef AABB_triangle_traits::Point_and_primitive_id Point_and_Primitive_id;
typedef CGAL::AABB_tree<AABB_triangle_traits> Tree;
typedef Tree::Object_and_primitive_id Object_and_Primitive_id;

// if we are on MSCV, we need to instantiate the extern variable ORIGIN otherwise there will be a linking error
#ifdef _MSC_VER
namespace CGAL {
    const CGAL::Origin ORIGIN;
};
#endif

#include "visibility.h"

struct VisibilityTask{
    const array<double, 9>* sensors_arr;
    const array<double, 3>* cams_arr;
    const std::vector<K::Point_3>& verts_v;
    const std::vector<K::Vector_3>& normals_v;
    const bool use_sensors;
    const Tree& tree;
    const double min_dist;
    uint32_t* visibility_mat;
    double* normal_dot_cam_mat;

    VisibilityTask(const array<double, 9>* sensors_arr,
                   const array<double, 3>* cams_arr,
                   const std::vector<K::Point_3>& verts_v,
                   const std::vector<K::Vector_3>& normals_v,
                   const bool use_sensors,
                   const Tree& tree,
                   const double& min_dist,
                   uint32_t* visibility_mat,
                   double* normal_dot_cam_mat):
                   sensors_arr(sensors_arr), cams_arr(cams_arr), verts_v(verts_v),
                   normals_v(normals_v), use_sensors(use_sensors),
                   tree(tree),
                   min_dist(min_dist), visibility_mat(visibility_mat),
                   normal_dot_cam_mat(normal_dot_cam_mat){;}

    void operator() (const int icam) const{
        Point cam(cams_arr[icam][0], cams_arr[icam][1], cams_arr[icam][2]);
        K::Vector_3 xoff,yoff,zoff;
        double planeoff;
        if (use_sensors){
            xoff = K::Vector_3(sensors_arr[icam][0],sensors_arr[icam][1],sensors_arr[icam][2]);
            yoff = K::Vector_3(sensors_arr[icam][3],sensors_arr[icam][4],sensors_arr[icam][5]);
            zoff = K::Vector_3(-sensors_arr[icam][6],-sensors_arr[icam][7],-sensors_arr[icam][8]);
            // compute D; dot product between plane normal zoff and a point (cam+zoff) on the plane
            planeoff = zoff*((cam+zoff)-CGAL::ORIGIN);
        }
        for(unsigned ivert=0; ivert<verts_v.size(); ++ivert)
        {
            Vector dir = cam - verts_v[ivert];
            dir = dir/sqrt(dir.squared_length());
            // if a ray in the normal half-volume does not intersect
            // there's nothing between the "cam" and the vert
            uint32_t reach_lens = 0;
            reach_lens = !(tree.do_intersect(Ray(verts_v[ivert] + min_dist*dir,dir)));
            if(!normals_v.empty())
                normal_dot_cam_mat[ivert + icam*verts_v.size()] = normals_v[ivert]*dir;;
            if (use_sensors){
                if (reach_lens){
                    // compute intersection between ray and sensor plane
                    // specifically, its deviation from the sensor center
                    double t = -(zoff*(verts_v[ivert]-CGAL::ORIGIN) - planeoff)/(zoff*dir);
                    K::Vector_3 p_i = (verts_v[ivert] + t*dir) - (cam+zoff);
                    // check that the projection of the deviation from the sensor into xy sensor axis
                    // is smaller than the axis themselves
                    // x*(bar(y)) < ||y|| -> x*y < ||y||^2
                    uint32_t reach_sensor = ((fabs(p_i*xoff) < xoff.squared_length()) &&
                            (fabs(p_i*yoff) < yoff.squared_length()));
                    visibility_mat[ivert + icam*verts_v.size()] = reach_sensor;
                }
                else
                    visibility_mat[ivert + icam*verts_v.size()] = 0;
            }
            else
                visibility_mat[ivert + icam*verts_v.size()] = reach_lens;
        }
    }

#if HAVE_TBB
    void operator()( const tbb::blocked_range<int>& range) const{
        for(int icam=range.begin(); icam!=range.end(); ++icam)
            this->operator()(icam);
    }
#elif HAVE_OMP
    #pragma omp parallel for
    void operator()( const std::vector<int>& range) const{
        for(std::vector<int>::const_iterator itcam=range.begin(); itcam!=range.end(); ++itcam)
            this->operator()(*itcam);
    }
#else
    void operator()( const std::vector<int>& range) const{
        for(std::vector<int>::const_iterator itcam=range.begin(); itcam!=range.end(); ++itcam)
            this->operator()(*itcam);
    }
#endif
};

void _internal_compute(const TreeAndTri* search, const double* normals,
                       const double* cams, const size_t n_cams,
                       const bool use_sensors, const double* sensors,
                       const double& min_dist, uint32_t *visibility_mat,
                       double *normal_dot_cam_mat){


    std::vector<K::Vector_3> normals_v;
    if(normals != NULL){
        const array<double, 3>* normals_arr=reinterpret_cast<const array<double,3>*>(normals);
        normals_v.reserve(search->points.size());
        for(size_t pp=0; pp<search->points.size(); ++pp){
            normals_v.push_back(K::Vector_3(normals_arr[pp][0],
                                            normals_arr[pp][1],
                                            normals_arr[pp][2]));
        }
    }

    const array<double, 3>* cams_arr=reinterpret_cast<const array<double,3>*>(cams);
    const array<double, 9>* sensors_arr=reinterpret_cast<const array<double,9>*>(sensors);

    VisibilityTask vtask(sensors_arr, cams_arr, search->points, normals_v,
                         use_sensors, search->tree,
                         min_dist, visibility_mat, normal_dot_cam_mat);
#if HAVE_TBB
    tbb::task_scheduler_init init;
    tbb::parallel_for( tbb::blocked_range<int>(0,n_cams), vtask);
#else
    std::vector<int> range(n_cams);
    for(unsigned i=0;i<n_cams;++i)
        range[i] = i;
    vtask(range);
#endif
    //tbb::parallel_for_each(boost::counting_iterator<int>(0),
    //                       boost::counting_iterator<int>(n_cams),
    //                       vtask);
    //vtask(tbb::blocked_range<int>(0,n_cams));
}

