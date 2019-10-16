#ifndef __NEAREST_TRIANGLE_HPP__
#define __NEAREST_TRIANGLE_HPP__

#include <vector>

#define CGAL_CFG_NO_CPP0X_VARIADIC_TEMPLATES 1
#include <CGAL/AABB_tree.h> // must be inserted before kernel
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>
#include <CGAL/Simple_cartesian.h>

#include <boost/cstdint.hpp>
#include <boost/array.hpp>


typedef CGAL::Simple_cartesian<double> K;
using boost::uint32_t;
using boost::uint64_t;
using boost::array;
using std::vector;

typedef CGAL::AABB_triangle_primitive<K, vector<K::Triangle_3>::iterator > Primitive;
typedef CGAL::AABB_traits<K, Primitive> AABB_triangle_traits;
typedef CGAL::AABB_tree<AABB_triangle_traits> Tree;

typedef CGAL::Vector_3<K> Normal;

struct TreeAndTri {
    vector<K::Triangle_3> triangles;
    Tree tree;
};

template<typename T>
boost::uint64_t wrapPointer(T *ptr) {
    return reinterpret_cast<uint64_t>(ptr);
}
template<typename T>
T* unwrapPointer(uint64_t ptr) {
    return reinterpret_cast<T*>(ptr);
}

#include <CGAL/Orthogonal_incremental_neighbor_search.h>
#include <CGAL/Search_traits_3.h>

typedef CGAL::Search_traits_3<K> IncTreeTraits;
typedef CGAL::Orthogonal_incremental_neighbor_search<IncTreeTraits> INN_search;
typedef INN_search::iterator INN_iterator;
typedef INN_search::Tree ITree;

struct ITreeAndNormals{
    vector<Normal> normals;
    vector<K::Point_3> centers;
    ITree itree;
};

#endif
