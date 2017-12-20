#ifndef q_carley_h
#define q_carley_h

#include<deal.II/base/quadrature_lib.h>
#include <boost/math/special_functions/legendre.hpp>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>

DEAL_II_NAMESPACE_OPEN

// template <int dim>
// class QCarley: public Quadrature<dim>
// {
// public:
//   /**
//    * The constructor  takes as parameters the order @p n,
//    * location of a singularity, and the order of legendre polynomial the function will integrate exactly using
//    * logrithm(r), 1/r, 1/r^2.
//    * A Gauss Legendre quadrature of order n
//    * will be used as base quadrature.
//    */
//   QCarley (const unsigned int n, const unsigned int m, const Point<dim> singularity);
//
// private:
//
//   void compute_weights();
//
//   void assemble_matrix_rhs(FullMatrix<double> &matrix_final, Vector<double> &rhs_final);
//
//   double logarithm_primitive(const double t);
//
//   double compute_alone_integral(const unsigned int legendre_order);
//
//   double compute_log_integral(const unsigned int legendre_order);
//
//   double compute_1_r_integral(const unsigned int legendre_order);
//
//   double compute_1_r2_integral(const unsigned int legendre_order);
//
//   double legendre_polynomial_derivative(const unsigned int legendre_order, const double point_x);
//
//
//   unsigned int N;
//   unsigned int M;
//   Point<1> x;
//
//
//
// };

template <int dim>
class QTellesGen: public Quadrature<dim>
{
public:

  QTellesGen(const Quadrature<1> &base_quad, const Point<dim> &singularity, const unsigned int order=3);

  QTellesGen(const unsigned int n, const Point<dim> &singularity, const unsigned int order=3);

// protected:
//   std::vector<Point<dim, long double> > quadrature_points;

};

template <int dim>
class QTellesOnBoundary: public Quadrature<dim>
{
public:

  // QTellesOnBoundary(const Quadrature<1> &base_quad, const Point<dim> &singularity, const unsigned int order=3);

  QTellesOnBoundary(const unsigned int n, const Point<dim> &singularity, const unsigned int order=3);

private:

  unsigned int quad_size(const Point<2> singularity, const unsigned int n);
};


template <int dim>
class MyQGaussOneOverR : public Quadrature<dim>
{
public:
  /**
   * This constructor takes three arguments: the order of the Gauss formula,
   * the point of the reference element in which the singularity is located,
   * and whether we include the weighting singular function inside the
   * quadrature, or we leave it in the user function to be integrated.
   *
   * Traditionally, quadrature formulas include their weighting function, and
   * the last argument is set to false by default. There are cases, however,
   * where this is undesirable (for example when you only know that your
   * singularity has the same order of 1/R, but cannot be written exactly in
   * this way).
   *
   * In other words, you can use this function in either of the following way,
   * obtaining the same result:
   *
   * @code
   * MyQGaussOneOverR singular_quad(order, q_point, false);
   * // This will produce the integral of f(x)/R
   * for(unsigned int i=0; i<singular_quad.size(); ++i)
   *   integral += f(singular_quad.point(i))*singular_quad.weight(i);
   *
   * // And the same here
   * MyQGaussOneOverR singular_quad_noR(order, q_point, true);
   *
   * // This also will produce the integral of f(x)/R, but 1/R has to
   * // be specified.
   * for(unsigned int i=0; i<singular_quad.size(); ++i) {
   *   double R = (singular_quad_noR.point(i)-cell->vertex(vertex_id)).norm();
   *   integral += f(singular_quad_noR.point(i))*singular_quad_noR.weight(i)/R;
   * }
   * @endcode
   */
  MyQGaussOneOverR(const unsigned int n,
                 const Point<dim> singularity,
                 const bool factor_out_singular_weight=false);
  /**
   * The constructor takes three arguments: the order of the Gauss formula,
   * the index of the vertex where the singularity is located, and whether we
   * include the weighting singular function inside the quadrature, or we
   * leave it in the user function to be integrated. Notice that this is a
   * specialized version of the previous constructor which works only for the
   * vertices of the quadrilateral.
   *
   * Traditionally, quadrature formulas include their weighting function, and
   * the last argument is set to false by default. There are cases, however,
   * where this is undesirable (for example when you only know that your
   * singularity has the same order of 1/R, but cannot be written exactly in
   * this way).
   *
   * In other words, you can use this function in either of the following way,
   * obtaining the same result:
   *
   * @code
   * MyQGaussOneOverR singular_quad(order, vertex_id, false);
   * // This will produce the integral of f(x)/R
   * for(unsigned int i=0; i<singular_quad.size(); ++i)
   *   integral += f(singular_quad.point(i))*singular_quad.weight(i);
   *
   * // And the same here
   * MyQGaussOneOverR singular_quad_noR(order, vertex_id, true);
   *
   * // This also will produce the integral of f(x)/R, but 1/R has to
   * // be specified.
   * for(unsigned int i=0; i<singular_quad.size(); ++i) {
   *   double R = (singular_quad_noR.point(i)-cell->vertex(vertex_id)).norm();
   *   integral += f(singular_quad_noR.point(i))*singular_quad_noR.weight(i)/R;
   * }
   * @endcode
   */
  MyQGaussOneOverR(const unsigned int n,
                 const unsigned int vertex_index,
                 const bool factor_out_singular_weight=false);
private:
  /**
   * Given a quadrature point and a degree n, this function returns the size
   * of the singular quadrature rule, considering whether the point is inside
   * the cell, on an edge of the cell, or on a corner of the cell.
   */
  static unsigned int quad_size(const Point<dim> singularity,
                                const unsigned int n);
};
DEAL_II_NAMESPACE_CLOSE
#endif
