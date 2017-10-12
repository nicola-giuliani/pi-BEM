#include "../include/q_carley.h"
#include <math.h>
DEAL_II_NAMESPACE_OPEN


int sgn(double val) {
    return (0. < val) - (val < 0.);
}


template <int dim>
QCarley<dim>::QCarley (
  const unsigned int n, const unsigned int m, const Point<dim> singularity)
  :
/**
* We need the explicit implementation if dim == 1. If dim > 1 we use the
* former implementation and apply a tensorial product to obtain the higher
* dimensions.
**/
  Quadrature<dim>(
    dim == 2 ?
    QAnisotropic<dim>(
      QCarley<1>(n, m, Point<1>(singularity[0])),
      QCarley<1>(n, m, Point<1>(singularity[1]))) :
    dim == 3 ?
    QAnisotropic<dim>(
      QCarley<1>(n, m, Point<1>(singularity[0])),
      QCarley<1>(n, m, Point<1>(singularity[1])),
      QCarley<1>(n, m, Point<1>(singularity[2]))) :
    Quadrature<dim>())
{
}


template <int dim>
double QCarley<dim>::logarithm_primitive(const double t)
{
  double tol = 1e-10;
  if(std::fabs(t) < tol)
    return 0.;
  else
    return t*(std::log(std::fabs(t)) - 1.);
  return 0.;
}

template <int dim>
double QCarley<dim>::compute_alone_integral(const unsigned int legendre_order)
{
  if(legendre_order == 0)
    return 2.;
  else
    return 0.;
}
template <int dim>
double QCarley<dim>::legendre_polynomial_derivative(const unsigned int legendre_order, const double point_x)
{
  double tol = 1e-10;
  if(legendre_order == 0)
    return 0.;
  else if(std::fabs(point_x - 1.) < tol)
    return legendre_order * (legendre_order + 1.) * 0.5;
  else if(std::fabs(point_x + 1.) < tol)
    return -1. * std::pow(-1., legendre_order) * legendre_order * (legendre_order + 1.) * 0.5;
  else
  {
    auto Pn    = boost::math::legendre_p(legendre_order  , point_x);
    auto Pn_m1 = boost::math::legendre_p(legendre_order-1, point_x);
    return legendre_order * (point_x * Pn - Pn_m1) / (point_x*point_x - 1.);
  }
}

template <int dim>
double QCarley<dim>::compute_log_integral(const unsigned int legendre_order)
{
  double result = 0.;
  double tol = 1e-10;
  if(legendre_order == 0)
  {
    result = logarithm_primitive(0.)-logarithm_primitive(x[0]-1.);
    result += (logarithm_primitive(x[0]+1.)-logarithm_primitive(0.));
  }
  else
  {
    if(std::fabs(std::fabs(x[0])-1.) > tol)
    {
      auto Qn_p1 = boost::math::legendre_q(legendre_order+1,x[0]);
      auto Qn_m1 = boost::math::legendre_q(legendre_order-1,x[0]);
      result = 2. * (Qn_p1 - Qn_m1) / (2. * legendre_order);
    }
    else
    {
      if(legendre_order % 2 == 0)
      {
        unsigned int m = legendre_order / 2;
        double helper_1 = 0.;
        double helper_2 = 0.;
        for(unsigned int k=0; k<=m; ++k)
        {
          helper_2 = 0.;
          for(unsigned int q=0; q<=m-k; ++q)
            helper_2 += 1. / (2.*q+1);
          helper_1 += (std::pow(-1., k) * boost::math::factorial<double>(4*m - 2*k))/
                       (boost::math::factorial<double>(k) * boost::math::factorial<double>(2*m - k) * boost::math::factorial<double>(2*m - 2*k + 1.)) *
                      (std::log(2.)-helper_2);
        }
        result = helper_1 * 2. / std::pow(4., m);
      }
      else
      {
        unsigned int m = (legendre_order - 1) / 2;
        double helper_1 = 0.;
        double helper_2 = 0.;
        for(unsigned int k=0; k<=m; ++k)
        {
          helper_2 = 0.;
          for(unsigned int q=0; q<=m-k; ++q)
            helper_2 += 1. / (2.*q+1);
          helper_1 += (std::pow(-1., k)* boost::math::factorial<double>(4*m - 2*k + 2.))/
                       (boost::math::factorial<double>(k) * boost::math::factorial<double>(2*m - k + 1.) * boost::math::factorial<double>(2*m - 2*k + 2.)) *
                      (helper_2);
        }
        result = helper_1 * 1. / std::pow(4., m) * sgn(x[0]);
      }
    }
  }
  return result;
}

template <int dim>
double QCarley<dim>::compute_1_r_integral(const unsigned int legendre_order)
{
  double tol = 1e-10;
  double result = 0.;
  auto Pn_x = boost::math::legendre_p(legendre_order, x[0]);
  double helper = 0.;
  unsigned int remainder_order = 6;
  QGauss<1> quad_rem(remainder_order);
  for(unsigned int j=0; j<quad_rem.size(); ++j)
  {
    double tj, wj;
    tj = 2.*(quad_rem.point(j)[0]-0.5);
    wj = quad_rem.weight(j)*2.;
    auto Pn_tj = boost::math::legendre_p(legendre_order, tj);
    helper += wj * (Pn_tj - Pn_x) / (x[0] - tj);
  }
  if(std::fabs(std::fabs(x[0])-1.) < tol)
    result = Pn_x * std::log(2.) * sgn(x[0]);
  else
    result = Pn_x * std::log(std::fabs((x[0] + 1.) / (1. - x[0])));
  return result + helper;
}

template <int dim>
double QCarley<dim>::compute_1_r2_integral(const unsigned int legendre_order)
{
  double tol = 1e-10;
  double result = 0.;
  auto Pn_x = boost::math::legendre_p(legendre_order, x[0]);
  auto Pnprime_x = legendre_polynomial_derivative(legendre_order, x[0]);
  double helper = 0.;
  unsigned int remainder_order = 6;
  QGauss<1> quad_rem(remainder_order);

  for(unsigned int j=0; j<quad_rem.size(); ++j)
  {
    double tj, wj;
    tj = 2.*(quad_rem.point(j)[0]-0.5);
    wj = quad_rem.weight(j)*2.;
    auto Pn_tj = boost::math::legendre_p(legendre_order, tj);
    helper += wj * (Pn_tj - Pn_x + Pnprime_x * (x[0] - tj)) / (x[0] - tj) / (x[0] - tj);
  }
  if(std::fabs(std::fabs(x[0])-1.) < tol)
  {
    result = Pn_x * (-0.5);
    result += -Pnprime_x * std::log(2.) * sgn(x[0]);
  }
  else
  {
    result = Pn_x * 2. / (x[0] * x[0] - 1.);
    result += -Pnprime_x * std::log(std::fabs((x[0] + 1.) / (1. - x[0])));
  }
  std::cout<<"r2 "<<result<<" "<<helper<<" "<<Pnprime_x<<" "<<x[0]<<std::endl;
  return result + helper;
}


template <int dim>
void QCarley<dim>::assemble_matrix_rhs(FullMatrix<double> &matrix_final, Vector<double> &rhs_final)
{
  AssertThrow(true, ExcMessage("This part must not be used unless in one dimension"));
}

template <int dim>
void QCarley<dim>::compute_weights()
{
  AssertThrow(true, ExcMessage("This part must not be used unless in one dimension"));
}

template <>
void QCarley<1>::assemble_matrix_rhs(FullMatrix<double> &matrix_final, Vector<double> &rhs_final)
{
  FullMatrix<double> matrix_ls(4*M,N);
  Vector<double> rhs_ls(4*M);

  for(unsigned int i=0; i<M; ++i)
  {
    rhs_ls[i]=compute_alone_integral(i);
    rhs_ls[M+i]=compute_log_integral(i);
    rhs_ls[2*M+i]=compute_1_r_integral(i);
    rhs_ls[3*M+i]=compute_1_r2_integral(i);
    for(unsigned int j=0; j<N; ++j)
    {
      auto t = quadrature_points[j];
      auto Pit = boost::math::legendre_p(i,t[0]);
      std::cout<<i<<" "<<t[0]<<" "<<Pit<<std::endl;
      matrix_ls.set(i    , j, Pit);
      matrix_ls.set(i+M  , j, Pit * std::log(std::fabs(x[0] - t[0])));
      matrix_ls.set(i+2*M, j, Pit / (x[0] - t[0]));
      matrix_ls.set(i+3*M, j, Pit / (x[0] - t[0]) / (x[0] - t[0]));
    }
  }
  matrix_ls.Tmmult(matrix_final, matrix_ls);
  matrix_ls.print_formatted(std::cout);
  std::cout<<std::endl;
  rhs_ls.print(std::cout);
  std::cout<<std::endl;

  matrix_ls.Tvmult(rhs_final,rhs_ls);

}


template <>
void QCarley<1>::compute_weights()
{
  FullMatrix<double> matrix_ls_final(N,N);
  Vector<double> rhs_ls_final(N);
  assemble_matrix_rhs(matrix_ls_final, rhs_ls_final);
  matrix_ls_final.print_formatted(std::cout);
  std::cout<<std::endl;
  rhs_ls_final.print(std::cout);
  matrix_ls_final.gauss_jordan();
  Vector<double> w(weights.size());
  matrix_ls_final.vmult(w, rhs_ls_final);
  for(unsigned int i=0; i<weights.size(); ++i)
    weights[i] = w[i];

}

template <>
QCarley<1>::QCarley (
  const unsigned int n, const unsigned int m, const Point<1> singularity)
  :
/**
* We explicitly implement the Carley rule if dim == 1.
**/
  N(n),
  M(m),
  x(singularity)
{
  // We generate the base GaussLegendre quadrature ans we initialize the quadrature points and weights
  x[0] = (x[0] - 0.5) * 2.;
  QGauss<1> base_quad(N);
  weights.resize(base_quad.size());
  quadrature_points.resize(base_quad.size());
  for(unsigned int i=0; i<base_quad.size(); ++i)
  {
    weights[i] = 0.;
    quadrature_points[i][0] = 2. * (base_quad.point(i)[0] - 0.5);
  }
  compute_weights();
  for(unsigned int i=0; i<base_quad.size(); ++i)
  {
    weights[i] *= 0.5;
    quadrature_points[i][0] = 0.5 * (base_quad.point(i)[0] + 1.);
  }
}


template class QCarley<1> ;
template class QCarley<2> ;
template class QCarley<3> ;

DEAL_II_NAMESPACE_CLOSE
