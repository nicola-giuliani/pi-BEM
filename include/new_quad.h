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


/**
 * Given an arbitrary quadrature formula, return one that chops the quadrature
 * points above the hyper-plane defined by $\sum_i x_i = 1$. In other words,
 * it extracts those quadrature points from the base formula that satisfy
 * $\sum_i (\mathbf x_q)_i \le 1$."
 *
 * In general the resulting quadrature is not very useful, unless the
 * quadrature you started from has been constructed specifically to integrate
 * over triangles or tetrahedra. This class ensures that the resulting
 * quadrature formula only has quadrature points in the reference simplex or on
 * its boundary.
 *
 * No transformation is applied to the weights. The weights referring to points
 * that live outside the reference simplex are simply discarded. If you chop a
 * quadrature formula that is good for both the lower and upper triangles
 * separately, then the weights should add up to 1/2.
 *
 * @author Luca Heltai, 2017.
 */
template <int dim>
class QSimplex : public Quadrature<dim>
{
public:
  /**
   * Construct a quadrature that only contains the points that are in the lower
   * left reference simplex.
   *
   * @param[in] quad The input quadrature.
   */
  QSimplex(const Quadrature<dim> &quad);

  /**
   * Return an affine transformation of this quadrature, that can be used to integrate
   * on the simplex identified by `vertices`.
   *
   * Both the quadrature point locations and the weights are transformed, so that you
   * can effectively use the resulting quadrature to integrate on the simplex.
   *
   * The transformation is defined as
   * \f[
   * x = v_0 + B \hat x
   * \f]
   * where the matrix $B$ is given by $B_{ij} = v[j][i]-v[0][i]$.
   *
   * The weights are scaled with the absolute value of the determinant of $B$,
   * that is $J := |\text{det}(B)|$. If $J$ is zero, an empty quadrature is
   * returned. This may happen, in two dimensions, if the three vertices are
   * aligned, or in three dimensions if the four vertices are on the same
   * plane.
   *
   * @param[in] vertices The vertices of the simplex you wish to integrate on
   * @return A quadrature object that can be used to integrate on the simplex
   */
  Quadrature<dim>
  compute_affine_transformation(const std::array<Point<dim>, dim+1> &vertices) const;

};

/**
 * A two dimensional simplex quadrature with zero Jacobian on vertex zero. The
 * quadrature is obtained through the following polar transformation:
 *
 * \f[
 *  \begin{pmatrix}
 *  x \\
 *  y
 *  \end{pmatrix}
 *  =
 * \begin{pmatrix}
 *  \frac{\hat x}{\sin(\theta)+\cos(\theta)} cos(\theta) \\
 *  \frac{\hat x}{\sin(\theta)+\cos(\theta)} sin(\theta)
 *  \end{pmatrix}
 *  \qquad \theta := \frac\pi 2 \hat y
 * \f]
 *
 * @author Luca Heltai, 2017
 */
class  QTrianglePolar: public QSimplex<2>
{
public:
  /**
   * Construct a QTrianglePolar quadrature, with different formulas in the
   * radial and angular directions.
   *
   * @param radial_quadrature Radial quadrature
   * @param angular_quadrature Angular quadrature
   */
  QTrianglePolar(const Quadrature<1> &radial_quadrature,
                 const Quadrature<1> &angular_quadrature);

  /**
   * Call the other constructor, with QGauss<1>(n) for both radial and
   * angular quadrature.
   *
   * @param n Order of QGauss quadrature
   */
  QTrianglePolar(const unsigned int &n);
};



class  QDuffy: public QSimplex<2>
{
public:

  QDuffy(const unsigned int &n, const unsigned int &beta=1.);
};

/**
 * A quadrature that implements the Lachat-Watson transformation from a
 * square to a triangle to integrate singularities in the origin of the
 * reference simplex.
 *
 * The Lachat-Watson transformation is defined as
 * \f[
 * \begin{pmatrix}
 * x\\
 * y
 * \end{pmatrix}
 * =
 * \begin{pmatrix}
 * \hat x(1-\hat y)\\
 * \hat x \hat y
 * \end{pmatrix}
 * \f]
 * with determinant of the Jacobian equal to $J = \hat x$. Such
 * transformation maps the reference square $[0,1]\times[0,1]$ to the
 * reference simplex, by collapsing the left side of the square, and
 * shearing the resulting triangle to the reference one. This
 * transformation allows one integrate singularities of order $1/R$
 * in the origin.
 *
 * @author Luca Heltai, 2017.
 */
class  QLachatWatson: public QSimplex<2>
{
public:
  /**
   * Constructor that allows the specificatino of different quadrature rules
   * along the "radial" and "angular" directions.
   *
   * Since this quadrature is not based on a Polar change of coordinates, it
   * is not fully proper to talk about radial and angular directions. However,
   * the effect of the Lachat-Watson transformation is similar to a polar change
   * of coordinates, since the resulting quadrature points are aligned radially
   * with respect to the singularity.
   *
   * This quadrature formula is cheaper to compute than the QTrianglePolar and
   * it is slightly less efficient on $1/R$ type singularities, however it
   * behaves better on non-singular arguments, making it somewhat more
   * flexible, if you want to integrate both singular and non-singular
   * functions.
   *
   * @param radial_quadrature Base quadrature to use in the radial direction
   * @param angular_quadrature Base quadrature to use in the angular direction
   */
  QLachatWatson(const Quadrature<1> &radial_quadrature,
                const Quadrature<1> &angular_quadrature);

  /**
   * Calls the above constructor with QGauss<1>(n) quadrature formulas for
   * both the radial and angular quadratures.
   *
   * @param n
   */
  QLachatWatson(const unsigned int &n);
};

/**
 * A quadrature to use when the cell should be split in subregions to integrate
 * using one or more base quadratures.
 *
 * @author Luca Heltai, 2017.
 */
template<int dim>
class QSplit : public Quadrature<dim>
{
public:
  /**
   * Construct a quadrature formula by splitting the reference hyper cube into
   * the minimum number of simplices that have vertex zero coinciding with
   * `split_point`, and patch together affine transformations of the `base`
   * quadrature.
   *
   * The resulting quadrature can be used, for example, to integrate functions
   * with integrable singularities at the split point, provided that you select
   * as base quadrature one that can integrate singular points on vertex zero
   * of the reference simplex.
   *
   * An example usage in dimension two is given by:
   * @code
   * const unsigned int order = 5;
   * QSplit<2> quad(QTrianglePolar(order), Point<2>(.3,.4));
   * @endcode
   *
   * The resulting quadrature will look like the following:
   * @image html split_quadrature.png ""
   *
   * @param base Base QSimplex quadrature to use
   * @param split_point Where to split the hyper cube
   */
  QSplit(const QSimplex<dim> &base,
         const Point<dim> &split_point);
};

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
