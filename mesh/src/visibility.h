#ifndef VISIBILITY_H
#define VISIBILITY_H
#include <string>
#define CGAL_CFG_NO_CPP0X_VARIADIC_TEMPLATES 1
#include <boost/cstdint.hpp>
#include <vector>
#include <CGAL/Simple_cartesian.h>

typedef CGAL::Simple_cartesian<double>::Point_3 Point;
void _internal_compute(const TreeAndTri* search, const double* normals,
                       const double* cams, const size_t n_cams,
                       const bool use_sensors, const double* sensors,
                       const double& min_dist, uint32_t *visibility_mat,
                       double *normal_dot_cam_mat);

class VisibilityException: public std::exception {
public:
    VisibilityException(std::string m="VisibilityException!"):msg(m) {}
    ~VisibilityException() throw() {}
    const char* what() const throw() { return msg.c_str(); }
private:
    std::string msg;
};
#endif // VISIBILITY_H
