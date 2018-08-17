// Copyright (c) 2018 Max Planck Society for non-commercial scientific research
// This file is part of psbody.mesh project which is released under MPI License.
// See file LICENSE.txt for full license details.

#ifndef __NEAREST_TRIANGLE_HPP__
#define __NEAREST_TRIANGLE_HPP__

#include <vector>

#define CGAL_CFG_NO_CPP0X_VARIADIC_TEMPLATES 1
#include <CGAL/AABB_tree.h> // must be inserted before kernel
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>
#include <CGAL/Simple_cartesian.h>

#include <CGAL/intersections.h>

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

struct TreeAndTri {
    vector<K::Triangle_3> triangles;
    vector<K::Point_3> points;
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

#endif
