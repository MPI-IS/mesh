// Author(s) : Javier Romero

#ifndef AABB_N_TREE_H
#define AABB_N_TREE_H

#include <vector>

#define CGAL_CFG_NO_CPP0X_VARIADIC_TEMPLATES 1
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/internal/AABB_tree/nearest_point_triangle_3.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/AABB_triangle_primitive.h>

#include <CGAL/intersections.h>
#include <CGAL/Bbox_3.h>

#include <boost/cstdint.hpp>
#include <boost/array.hpp>


typedef CGAL::Simple_cartesian<double> K;
using boost::uint32_t;
using boost::uint64_t;
using boost::array;
using std::vector;

typedef K::Point_3 Point;
typedef K::Segment_3 Segment;
typedef K::Triangle_3 Triangle;
typedef K::Vector_3 Normal;
typedef std::pair<Point,Normal> Point_Normal;

typedef std::vector<Triangle>::iterator Iterator;
typedef CGAL::AABB_triangle_primitive<K,Iterator> Primitive;


namespace CGAL {
    template<typename FT>
    FT dist_point_normal(const Point_Normal& a, const Point_Normal& b, FT eps){
        return (sqrt(squared_distance(a.first, b.first)) + eps*(1 - a.second*b.second));
    }

    // Adaptation of nearest_point_3 to take into account normals
    // product with weight eps
    template<typename FT>
    Point_Normal nearest_pointnormal_3(const Point_Normal& origin,
                                       const Triangle& triangle,
                                       const Point_Normal& bound,
                                       const FT eps,
                                       const K& k){
        // compute normal penalties
        const FT dist_n_bound = eps*(1 - origin.second*bound.second);
        //const Normal tri_n = unit_normal(triangle[0],triangle[1],triangle[2]);
        Normal tri_n = triangle.supporting_plane().orthogonal_direction().vector();
        tri_n = tri_n / sqrt(tri_n.squared_length());
        const FT dist_n_triangle = eps*(1 - origin.second*tri_n);

        // Distance from origin to bound
        const FT dist_bound = sqrt(squared_distance(origin.first,bound.first)) + dist_n_bound;

        // since dist_n_triangle < dist_triangle
        if (dist_n_triangle > dist_bound)
            return bound;

        // Project origin on triangle supporting plane
        const Point_Normal proj = std::make_pair(triangle.supporting_plane().projection(origin.first), tri_n);

        const FT dist_proj = sqrt(squared_distance(origin.first,proj.first)) + dist_n_triangle;

        Point moved_point;
        // If point is projected outside, return bound
        if ( dist_proj > dist_bound)
            return bound;
        // If proj is inside triangle, total dist is dist_proj
        else if ( CGAL::internal::is_inside_triangle_3<K>(proj.first,triangle,moved_point,k) )
            return proj;
        // Else return the constructed point (nearest point of triangle from proj)
        // if it is closest to origin than bound
        else{
            const FT dist_moved = sqrt(squared_distance(origin.first, moved_point)) + dist_n_triangle;
            return (dist_moved > dist_bound) ? bound : std::make_pair(moved_point, tri_n);
        }
    }

    // extends AABB_Traits with classes handling points with normals
    template <typename GeomTraits, typename AABB_primitive>
        class AABB_n_traits:public AABB_traits<GeomTraits,AABB_primitive>{

    public:

        typedef Point_Normal PointNormal;
        typedef AABB_n_traits<GeomTraits, AABB_primitive> AT;

        class Do_intersect{
        public:
            template<typename Query>
                bool operator()(const Query& q, const CGAL::Bbox_3& bbox) const{
                return CGAL::do_intersect(q, bbox);
            }

            template<typename Query>
                bool operator()(const Query& q, const AABB_primitive& pr) const{
                return GeomTraits().do_intersect_3_object()(q, pr.datum());
            }

            bool operator()(const typename GeomTraits::Triangle_3& q, const AABB_primitive& pr) const{

                // if any point is the same, don't consider it'
                if(q[0] == pr.datum()[0] || q[0] == pr.datum()[1] ||q[0] == pr.datum()[2] ||
                   q[1] == pr.datum()[0] || q[1] == pr.datum()[1] ||q[1] == pr.datum()[2] ||
                   q[2] == pr.datum()[0] || q[2] == pr.datum()[1] ||q[2] == pr.datum()[2])
                    return false;
                else
                    return CGAL::do_intersect(q, pr.datum());
            }
        };

        class Closest_point {
            typedef typename AT::Point_3 Point;
            typedef typename AT::PointNormal PointNormal;
            typedef typename AT::Primitive Primitive;
            typedef typename AT::FT FT;

        public:

            // for intersection: return the closest point on
            // triangle pr or bound to pn
            PointNormal operator()(const PointNormal& pn, const Primitive& pr,
                                   const PointNormal& bound, FT eps) const {
                return nearest_pointnormal_3(pn, pr.datum(), bound, eps, K());
            }

        };

        class Compare_distance {
            typedef typename AT::Point_3 Point;
            typedef typename AT::PointNormal PointNormal;
            typedef typename AT::FT FT;
            typedef typename AT::Primitive Primitive;
        public:

            // create a sphere that contains all possible results
            // (all points closer than current result) and
            // check if pr intersects
            template <class Solid>
                CGAL::Comparison_result operator()(const PointNormal& p, const Solid& pr, const PointNormal& bound, FT eps) const
                {
                    // d_q = ||q - p|| + eps(1 - n_q*n_p) > ||q-p||
                    // d_q < d_b -> ||q-p|| < d_b
                    // a sphere containing all q such that ||q-p|| < d_b
                    // contains all q such that d_q < d_b
                    FT safe_dist = dist_point_normal(p,bound,eps);
                    return GeomTraits().do_intersect_3_object()
                        (GeomTraits().construct_sphere_3_object()
                         (p.first, safe_dist*safe_dist), pr)?
                        CGAL::SMALLER : CGAL::LARGER;
                }
        };

        Closest_point closest_point_object() {return Closest_point();}
        Compare_distance compare_distance_object() {return Compare_distance();}
        Do_intersect do_intersect_object() {return Do_intersect();}

    };// end of AABB_n_traits


    /**
     * @class Do_intersect_noself_traits
     */
    template<typename AABBTraits, typename Query>
        class Do_intersect_noself_traits
    {
        typedef typename AABBTraits::FT FT;
        typedef typename AABBTraits::Point_3 Point;
        typedef typename AABBTraits::Primitive Primitive;
        typedef typename AABBTraits::Bounding_box Bounding_box;
        typedef typename AABBTraits::Primitive::Id Primitive_id;
        typedef typename AABBTraits::Point_and_primitive_id Point_and_primitive_id;
        typedef typename AABBTraits::Object_and_primitive_id Object_and_primitive_id;
        typedef ::CGAL::AABB_node<AABBTraits> Node;
        typedef typename ::CGAL::AABB_tree<AABBTraits>::size_type size_type;

    public:
    Do_intersect_noself_traits()
        : m_is_found(false)
            {}

        bool go_further() const { return !m_is_found; }

        void intersection(const Query& query, const Primitive& primitive)
        {
            if( AABBTraits().do_intersect_object()(query, primitive) )
                m_is_found = true;
        }

        bool do_intersect(const Query& query, const Node& node) const
        {
            return AABBTraits().do_intersect_object()(query, node.bbox());
        }

        bool is_intersection_found() const { return m_is_found; }

    private:
        bool m_is_found;
    };

    /**
     * @class Projection_n_traits
     */
    template <typename AABBTraits>
        class Projection_n_traits
        {
            typedef typename AABBTraits::FT FT;
            typedef typename AABBTraits::Point_3 Point;
            typedef typename AABBTraits::PointNormal PointNormal;
            typedef typename AABBTraits::Primitive Primitive;
            typedef typename AABBTraits::Bounding_box Bounding_box;
            typedef typename AABBTraits::Primitive::Id Primitive_id;
            typedef typename AABBTraits::Point_and_primitive_id Point_and_primitive_id;
            typedef typename AABBTraits::Object_and_primitive_id Object_and_primitive_id;
            typedef ::CGAL::AABB_node<AABBTraits> Node;

        public:
        Projection_n_traits(const PointNormal& hint,
                              const typename Primitive::Id& hint_primitive,
                              const FT eps)
            : m_closest_point(hint), m_closest_primitive(hint_primitive), eps(eps) {}

            bool go_further() const { return true; }

            void intersection(const PointNormal& query, const Primitive& primitive)
            {
                PointNormal new_closest_point = AABBTraits().closest_point_object()
                    (query, primitive, m_closest_point, eps);
                if(new_closest_point.first != m_closest_point.first)
                    {
                        m_closest_primitive = primitive.id();
                        m_closest_point = new_closest_point; // this effectively shrinks the sphere
                    }
            }

            bool do_intersect(const PointNormal& query, const Node& node) const
            {
                return AABBTraits().compare_distance_object()
                    (query, node.bbox(), m_closest_point, eps) == CGAL::SMALLER;
            }

            Point closest_point() const { return m_closest_point.first; }
            Point_and_primitive_id closest_point_and_primitive() const
            {
                return Point_and_primitive_id(m_closest_point.first, m_closest_primitive);
            }

        private:
            PointNormal m_closest_point;
            typename Primitive::Id m_closest_primitive;
            const FT eps;
        };

    // Class that extends AABB tree with PointNormal Search
    template <typename AABBTraits>
        class AABB_n_tree:public AABB_tree<AABBTraits>{
    public:
        typedef typename AABBTraits::Point_3 Point;
        typedef typename AABBTraits::PointNormal PointNormal;
        typedef typename AABBTraits::Point_and_primitive_id Point_and_primitive_id;

        AABB_n_tree():AABB_tree<AABBTraits>(){}

        template<typename ConstPrimitiveIterator>
            AABB_n_tree(ConstPrimitiveIterator first, ConstPrimitiveIterator beyond,
                        typename AABBTraits::FT eps):
        AABB_tree<AABBTraits>(first, beyond), eps(eps){}

        // XXX The hint is random; that is slow and could be closer (euclideanly) than the best point
        Point_and_primitive_id closest_point_and_primitive(const PointNormal& query) const{
            // return closest_point_and_primitive(query,best_hint(query.first));
            return closest_point_and_primitive(query,this->any_reference_point_and_id());
        }

        Point_and_primitive_id closest_point_and_primitive(const PointNormal& query,
                                                           const Point_and_primitive_id& hint) const{

            Normal hint_n = (*hint.second).supporting_plane().orthogonal_direction().vector();
            PointNormal hint_pn = std::make_pair(hint.first, hint_n/sqrt(hint_n.squared_length()));
            //            hint_pn = std::make_pair(Point(10000,10000,10000),Normal(0,0,1));
            Projection_n_traits<AABBTraits> projection_traits(hint_pn, hint.second, eps);
            this->traversal(query, projection_traits);
            return projection_traits.closest_point_and_primitive();
        }

        template<typename Query>
        bool do_intersect(const Query& query) const
        {
            //using namespace CGAL::internal::AABB_tree;
            Do_intersect_noself_traits<AABBTraits, Query> traversal_traits;
            this->traversal(query, traversal_traits);
            return traversal_traits.is_intersection_found();
        }

        typename AABBTraits::FT eps;
   };
}

typedef CGAL::AABB_n_traits<K, Primitive> AABB_n_triangle_traits;
typedef AABB_n_triangle_traits::Point_and_primitive_id Point_and_Primitive_id;
typedef CGAL::AABB_n_tree<AABB_n_triangle_traits> Tree;

struct TreeAndTri {
    TreeAndTri(const array<uint32_t, 3>* p_mesh_tri,
               const array<double, 3>* p_mesh_points,
               const double eps,
               const size_t T,
               const size_t P)
    {
        std::vector<K::Point_3> mesh_points;
        mesh_points.reserve(P);
        for(size_t pp=0; pp<P; ++pp){
            mesh_points.push_back(K::Point_3(p_mesh_points[pp][0],
                                             p_mesh_points[pp][1],
                                             p_mesh_points[pp][2]));
        }

        triangles.reserve(T);
        for(size_t tt=0; tt<T; ++tt) {
            triangles.push_back(K::Triangle_3(mesh_points[p_mesh_tri[tt][0]],
                                              mesh_points[p_mesh_tri[tt][1]],
                                              mesh_points[p_mesh_tri[tt][2]]));
        }

        tree.eps = eps;
        tree.rebuild(triangles.begin(), triangles.end());
        //        tree.accelerate_distance_queries();
    }

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

#endif // AABB_N_TREE_H

/***EMACS SETTINGS***/
/* Local Variables: */
/* tab-width: 2     */
/* End:             */
