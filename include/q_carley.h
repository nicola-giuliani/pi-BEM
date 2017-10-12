#ifndef q_carley_h
#define q_carley_h

#include<deal.II/base/quadrature_lib.h>
#include <boost/math/special_functions/legendre.hpp>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>

DEAL_II_NAMESPACE_OPEN

template <int dim>
class QCarley: public Quadrature<dim>
{
public:
  /**
   * The constructor  takes as parameters the order @p n,
   * location of a singularity, and the order of legendre polynomial the function will integrate exactly using
   * logrithm(r), 1/r, 1/r^2.
   * A Gauss Legendre quadrature of order n
   * will be used as base quadrature.
   */
  QCarley (const unsigned int n, const unsigned int m, const Point<dim> &singularity);

private:

  void compute_weights();

  void assemble_matrix_rhs(FullMatrix<double> &matrix_final, Vector<double> &rhs_final);

  double logarithm_primitive(const double t);

  double compute_alone_integral(const unsigned int legendre_order);

  double compute_log_integral(const unsigned int legendre_order);

  double compute_1_r_integral(const unsigned int legendre_order);

  double compute_1_r2_integral(const unsigned int legendre_order);

  double legendre_polynomial_derivative(const unsigned int legendre_order, const double point_x);


  unsigned int N;
  unsigned int M;
  Point<1> x;



};

DEAL_II_NAMESPACE_CLOSE
#endif
