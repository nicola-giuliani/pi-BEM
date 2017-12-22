#include "../include/new_quad.h"
#include <deal.II/base/geometry_info.h>
#include <math.h>
// #include <Eigen/Dense>
// #include <Eigen/SVD>
// #include <Eigen/LU>
// #include <Eigen/src/SVD/BDCSVD.h>
// #include <Accelerate/Accelerate.h>

// using namespace Eigen;

DEAL_II_NAMESPACE_OPEN


int sgn(double val) {
    return (0. < val) - (val < 0.);
}


// template <int dim>
// QCarley<dim>::QCarley (
//   const unsigned int n, const unsigned int m, const Point<dim> singularity)
//   :
// /**
// * We need the explicit implementation if dim == 1. If dim > 1 we use the
// * former implementation and apply a tensorial product to obtain the higher
// * dimensions.
// **/
//   Quadrature<dim>(
//     dim == 2 ?
//     QAnisotropic<dim>(
//       QCarley<1>(n, m, Point<1>(singularity[0])),
//       QCarley<1>(n, m, Point<1>(singularity[1]))) :
//     dim == 3 ?
//     QAnisotropic<dim>(
//       QCarley<1>(n, m, Point<1>(singularity[0])),
//       QCarley<1>(n, m, Point<1>(singularity[1])),
//       QCarley<1>(n, m, Point<1>(singularity[2]))) :
//     Quadrature<dim>())
// {
// }
//
//
// template <int dim>
// double QCarley<dim>::logarithm_primitive(const double t)
// {
//   double tol = 1e-10;
//   if(std::fabs(t) < tol)
//     return 0.;
//   else
//     return t*(std::log(std::fabs(t)) - 1.);
//   return 0.;
// }
//
// template <int dim>
// double QCarley<dim>::compute_alone_integral(const unsigned int legendre_order)
// {
//   if(legendre_order == 0)
//     return 2.;
//   else
//     return 0.;
// }
// template <int dim>
// double QCarley<dim>::legendre_polynomial_derivative(const unsigned int legendre_order, const double point_x)
// {
//   double tol = 1e-10;
//   if(legendre_order == 0)
//     return 0.;
//   else if(std::fabs(point_x - 1.) < tol)
//     return legendre_order * (legendre_order + 1.) * 0.5;
//   else if(std::fabs(point_x + 1.) < tol)
//     return -1. * std::pow(-1., legendre_order) * legendre_order * (legendre_order + 1.) * 0.5;
//   else
//   {
//     auto Pn    = boost::math::legendre_p(legendre_order  , point_x);
//     auto Pn_m1 = boost::math::legendre_p(legendre_order-1, point_x);
//     return legendre_order * (point_x * Pn - Pn_m1) / (point_x*point_x - 1.);
//   }
// }
//
// template <int dim>
// double QCarley<dim>::compute_log_integral(const unsigned int legendre_order)
// {
//   double result = 0.;
//   double tol = 1e-10;
//   if(legendre_order == 0)
//   {
//     result = logarithm_primitive(0.)-logarithm_primitive(x[0]-1.);
//     result += (logarithm_primitive(x[0]+1.)-logarithm_primitive(0.));
//   }
//   else
//   {
//     if(std::fabs(std::fabs(x[0])-1.) > tol)
//     {
//       auto Qn_p1 = boost::math::legendre_q(legendre_order+1,x[0]);
//       auto Qn_m1 = boost::math::legendre_q(legendre_order-1,x[0]);
//       result = 2. * (Qn_p1 - Qn_m1) / (2. * legendre_order+1.);
//       // std::cout<<"HERE I AM "<<Qn_p1<<" "<<Qn_m1<<std::endl;
//
//     }
//     else
//     {
//       if(legendre_order % 2 == 0)
//       {
//         unsigned int m = legendre_order / 2;
//         double helper_1 = 0.;
//         double helper_2 = 0.;
//         for(unsigned int k=0; k<=m; ++k)
//         {
//           helper_2 = 0.;
//           for(unsigned int q=0; q<=m-k; ++q)
//             helper_2 += 1. / (2.*q+1);
//           helper_1 += (std::pow(-1., k) * boost::math::factorial<double>(4*m - 2*k))/
//                        (boost::math::factorial<double>(k) * boost::math::factorial<double>(2*m - k) * boost::math::factorial<double>(2*m - 2*k + 1.)) *
//                       (std::log(2.)-helper_2);
//         }
//         result = helper_1 * 2. / std::pow(4., m);
//       }
//       else
//       {
//         unsigned int m = (legendre_order - 1) / 2;
//         double helper_1 = 0.;
//         double helper_2 = 0.;
//         for(unsigned int k=0; k<=m; ++k)
//         {
//           helper_2 = 0.;
//           for(unsigned int q=0; q<=m-k; ++q)
//             helper_2 += 1. / (2.*q+1);
//           helper_1 += (std::pow(-1., k)* boost::math::factorial<double>(4*m - 2*k + 2.))/
//                        (boost::math::factorial<double>(k) * boost::math::factorial<double>(2*m - k + 1.) * boost::math::factorial<double>(2*m - 2*k + 2.)) *
//                       (helper_2);
//
//         }
//         result = helper_1 * 1. / std::pow(4., m) * sgn(x[0]);
//       }
//     }
//   }
//   // std::cout<<"LOOK AT ME!!! "<<result<<" "<<legendre_order<<std::endl;
//   return result;
// }
//
// template <int dim>
// double QCarley<dim>::compute_1_r_integral(const unsigned int legendre_order)
// {
//   double tol = 1e-10;
//   double result = 0.;
//   auto Pn_x = boost::math::legendre_p(legendre_order, x[0]);
//   double helper = 0.;
//   unsigned int remainder_order = 6;
//   QGauss<1> quad_rem(remainder_order);
//   for(unsigned int j=0; j<quad_rem.size(); ++j)
//   {
//     double tj, wj;
//     tj = 2.*(quad_rem.point(j)[0]-0.5);
//     wj = quad_rem.weight(j)*2.;
//     auto Pn_tj = boost::math::legendre_p(legendre_order, tj);
//     helper += wj * (Pn_tj - Pn_x) / (x[0] - tj);
//   }
//   if(std::fabs(std::fabs(x[0])-1.) < tol)
//     result = Pn_x * std::log(2.) * sgn(x[0]);
//   else
//     result = Pn_x * std::log(std::fabs((x[0] + 1.) / (1. - x[0])));
//   return result + helper;
// }
//
// template <int dim>
// double QCarley<dim>::compute_1_r2_integral(const unsigned int legendre_order)
// {
//   double tol = 1e-10;
//   double result = 0.;
//   auto Pn_x = boost::math::legendre_p(legendre_order, x[0]);
//   auto Pnprime_x = legendre_polynomial_derivative(legendre_order, x[0]);
//   double helper = 0.;
//   unsigned int remainder_order = 6;
//   QGauss<1> quad_rem(remainder_order);
//
//   for(unsigned int j=0; j<quad_rem.size(); ++j)
//   {
//     double tj, wj;
//     tj = 2.*(quad_rem.point(j)[0]-0.5);
//     wj = quad_rem.weight(j)*2.;
//     auto Pn_tj = boost::math::legendre_p(legendre_order, tj);
//     helper += wj * (Pn_tj - Pn_x + Pnprime_x * (x[0] - tj)) / (x[0] - tj) / (x[0] - tj);
//   }
//   if(std::fabs(std::fabs(x[0])-1.) < tol)
//   {
//     result = Pn_x * (-0.5);
//     result += -Pnprime_x * std::log(2.) * sgn(x[0]);
//   }
//   else
//   {
//     result = Pn_x * 2. / (x[0] * x[0] - 1.);
//     result += -Pnprime_x * std::log(std::fabs((x[0] + 1.) / (1. - x[0])));
//   }
//   // std::cout<<"r2 "<<result<<" "<<helper<<" "<<Pnprime_x<<" "<<x[0]<<std::endl;
//   return result + helper;
// }
//
//
// template <int dim>
// void QCarley<dim>::assemble_matrix_rhs(FullMatrix<double> &matrix_final, Vector<double> &rhs_final)
// {
//   AssertThrow(true, ExcMessage("This part must not be used unless in one dimension"));
// }
//
// template <int dim>
// void QCarley<dim>::compute_weights()
// {
//   AssertThrow(true, ExcMessage("This part must not be used unless in one dimension"));
// }
//
// template <>
// void QCarley<1>::assemble_matrix_rhs(FullMatrix<double> &matrix_final, Vector<double> &rhs_final)
// {
//   MatrixXf matrix_ls(4*M,N);
//   VectorXf rhs_ls(4*M);
//
//   for(unsigned int i=0; i<M; ++i)
//   {
//     rhs_ls(i)     = compute_alone_integral(i);
//     rhs_ls(M+i)   = compute_log_integral(i);
//     rhs_ls(2*M+i) = compute_1_r_integral(i);
//     rhs_ls(3*M+i) = compute_1_r2_integral(i);
//     for(unsigned int j=0; j<N; ++j)
//     {
//       auto t = quadrature_points[j];
//       auto Pit = boost::math::legendre_p(i,t[0]);
//       // std::cout<<i<<" "<<t[0]<<" "<<Pit<<std::endl;
//       matrix_ls(i    , j) = Pit;
//       matrix_ls(i+M  , j) = Pit * std::log(std::fabs(x[0] - t[0]));
//       matrix_ls(i+2*M, j) = Pit / (x[0] - t[0]);
//       matrix_ls(i+3*M, j) = Pit / (x[0] - t[0]) / (x[0] - t[0]);
//       // matrix_ls.set(i    , j, Pit);
//       // matrix_ls.set(i+M  , j, Pit * std::log(std::fabs(x[0] - t[0])));
//       // matrix_ls.set(i+2*M, j, Pit / (x[0] - t[0]));
//       // matrix_ls.set(i+3*M, j, Pit / (x[0] - t[0]) / (x[0] - t[0]));
//     }
//   }
//   // matrix_ls.Tmmult(matrix_final, matrix_ls);
//   // matrix_ls.print_formatted(std::cout);
//   // std::cout<<std::endl;
//   // rhs_ls.print(std::cout);
//   // std::cout<<std::endl;
//   //
//   // matrix_ls.Tvmult(rhs_final,rhs_ls);
//   VectorXf solution(N);
//   // VectorXf checkk(N);
//   // VectorXf checkkk(N);
//   // std::cout << matrix_ls << std::endl;
//   // std::cout << rhs_ls << std::endl;
//   // checkkk(0)=-2.41455756;
//   // checkkk(1)=3.42263836;
//   // checkkk(2)=-0.36642816;
//   // checkkk(3)=-0.2180764;
//   // checkkk(4)=2.85357908;
//   // checkkk(5)=-1.73885892;
//   // checkkk(6)=-0.74846057;
//   // checkkk(7)=1.21016417;
//   // solution = matrix_ls.bdcSvd().solve(rhs_ls);
//   // solution = (matrix_ls.transpose() * matrix_ls).ldlt().solve(matrix_ls.transpose() * rhs_ls);
//   solution = matrix_ls.fullPivHouseholderQr().solve(rhs_ls);
//   // checkk = matrix_ls*solution - rhs_ls;
//   // solution = matrix_ls.jacobiSvd(ComputeThinU | ComputeThinV).solve(rhs_ls);
//   // std::cout << solution << std::endl;
//   // std::cout <<"try1"<<std::endl;
//   // std::cout << checkk << std::endl;
//   // std::cout <<"try2"<<std::endl;
//   // std::cout << matrix_ls*checkkk - rhs_ls << std::endl;
//   // std::cout << "rhs (3) "<<std::endl<<rhs_ls <<std::endl;
//   for(unsigned int i = 0; i<rhs_final.size(); ++i)
//     rhs_final[i] = solution(i);
//
// }
//
//
// template <>
// void QCarley<1>::compute_weights()
// {
//   FullMatrix<double> matrix_ls_final(N,N);
//   Vector<double> rhs_ls_final(N);
//   assemble_matrix_rhs(matrix_ls_final, rhs_ls_final);
//   // matrix_ls_final.print_formatted(std::cout);
//   // std::cout<<std::endl;
//   // rhs_ls_final.print(std::cout);
//   // matrix_ls_final.gauss_jordan();
//   // Vector<double> w(weights.size());
//   // matrix_ls_final.vmult(w, rhs_ls_final);
//   for(unsigned int i=0; i<weights.size(); ++i)
//     weights[i] = rhs_ls_final[i];
//
// }
//
// template <>
// QCarley<1>::QCarley (
//   const unsigned int n, const unsigned int m, const Point<1> singularity)
//   :
// /**
// * We explicitly implement the Carley rule if dim == 1.
// **/
//   N(n),
//   M(m),
//   x(singularity)
// {
//   // We generate the base GaussLegendre quadrature ans we initialize the quadrature points and weights
//   x[0] = (x[0] - 0.5) * 2.;
//   QGauss<1> base_quad(N);
//   weights.resize(base_quad.size());
//   quadrature_points.resize(base_quad.size());
//   for(unsigned int i=0; i<base_quad.size(); ++i)
//   {
//     weights[i] = 0.;
//     quadrature_points[i][0] = 2. * (base_quad.point(i)[0] - 0.5);
//   }
//   compute_weights();
//   for(unsigned int i=0; i<base_quad.size(); ++i)
//   {
//     weights[i] *= 0.5;
//     quadrature_points[i][0] = 0.5 * (base_quad.point(i)[0] + 1.);
//   }
// }


template<int dim>
QSimplex<dim>::QSimplex(const Quadrature<dim> &quad)
{
  std::vector<Point<dim> > qpoints;
  std::vector<double > weights;

  for (unsigned int i=0; i<quad.size(); ++i)
    {
      double r=0;
      for (unsigned int d=0; d<dim; ++d)
        r += quad.point(i)[d];
      if (r <= 1)
        {
          this->quadrature_points.push_back(quad.point(i));
          this->weights.push_back(quad.weight(i));
        }
    }
}



template<int dim>
Quadrature<dim>
QSimplex<dim>::compute_affine_transformation(const std::array<Point<dim>, dim+1>& vertices) const
{
  unsigned int i=0;
  Tensor<2,dim> B;
  for (unsigned int d=0; d<dim; ++d)
    B[d] = vertices[d+1]-vertices[0];

  B = transpose(B);
  auto J = std::abs(determinant(B));

  // if the determinant is zero, we return an empty quadrature
  if (J < 1e-12)
    return Quadrature<dim>();

  std::vector<Point<dim> > qp(this->size());
  std::vector<double> w(this->size());

  for (unsigned int i=0; i<this->size(); ++i)
    {
      qp[i] = Point<dim>(vertices[0]+B*this->point(i));
      w[i] = J*this->weight(i);
    }

  return Quadrature<dim>(qp, w);
}



QTrianglePolar::QTrianglePolar(const Quadrature<1> &radial_quadrature,
                               const Quadrature<1> &angular_quadrature) :
  QSimplex<2>(Quadrature<2>())
{
  QAnisotropic<2> base(radial_quadrature, angular_quadrature);
  this->quadrature_points.resize(base.size());
  this->weights.resize(base.size());
  for (unsigned int i=0; i<base.size(); ++i)
    {
      const auto &q = base.point(i);
      const auto &w = base.weight(i);

      const auto &xhat = q[0];
      const auto &yhat = q[1];

      const double t = numbers::PI_2*yhat;
      const double &pi = numbers::PI;
      const double st = std::sin(t);
      const double ct = std::cos(t);
      const double r = xhat/(st+ct);

      const double J = pi*xhat/(2*(std::sin(pi*yhat) + 1));

      this->quadrature_points[i] = Point<2>(r*ct, r*st);
      this->weights[i] = w*J;
    }
}


QDuffy::QDuffy(const unsigned int &n, const unsigned int &beta) :
  QSimplex<2>(Quadrature<2>())
{
  QGauss<2> q_helper(n);

  this->quadrature_points.resize(q_helper.size());
  this->weights.resize(q_helper.size());

  for(unsigned int i = 0; i<q_helper.size(); ++i)
  {
    auto u = q_helper.point(i)[0];
    auto v = q_helper.point(i)[1];
    auto w = q_helper.weight(i);

    auto Zx = std::pow(u, beta) * (1-v);
    auto Zy = std::pow(u, beta) * v;
    auto Zw = w * beta * std::pow(u, 2.*beta-1.);

    this->quadrature_points[i][0] = Zx;
    this->quadrature_points[i][1] = Zy;
    this->weights[i] = Zw;
  }

}



QTrianglePolar::QTrianglePolar(const unsigned int &n)
  :QTrianglePolar(QGauss<1>(n),QGauss<1>(n))
{}



QLachatWatson::QLachatWatson(const Quadrature<1> &radial_quadrature,
                             const Quadrature<1> &angular_quadrature) :
  QSimplex<2>(Quadrature<2>())
{
  QAnisotropic<2> base(radial_quadrature, angular_quadrature);
  this->quadrature_points.resize(base.size());
  this->weights.resize(base.size());
  for (unsigned int i=0; i<base.size(); ++i)
    {
      const auto &q = base.point(i);
      const auto &w = base.weight(i);

      const auto &xhat = q[0];
      const auto &yhat = q[1];

      const double x = xhat*(1-yhat);
      const double y = xhat*yhat;

      const double J = xhat;

      this->quadrature_points[i] = Point<2>(x,y);
      this->weights[i] = w*J;
    }
}



QLachatWatson::QLachatWatson(const unsigned int &n)
  :QLachatWatson(QGauss<1>(n),QGauss<1>(n))
{}



template<int dim>
QSplit<dim>::QSplit(const QSimplex<dim> &base,
                    const Point<dim> &split_point)
{
  std::array<Point<dim>, dim+1> vertices;
  vertices[0] = split_point;

  // Make a simplex from the split_point and the first dim vertices of each
  // face. In dimension three, we need to split the face in two triangles, so
  // we use once the first dim vertices of each face, and the second time the
  // the dim vertices of each face starting from 1.
  for (unsigned int f=0; f< GeometryInfo<dim>::faces_per_cell; ++f)
    for (unsigned int start=0; start < (dim >2 ? 2 : 1); ++start)
      {
        for (unsigned int i=0; i<dim; ++i)
          vertices[i+1] =
            GeometryInfo<dim>::unit_cell_vertex(
              GeometryInfo<dim>::face_to_cell_vertices(f, i+start)
            );
        auto quad = base.compute_affine_transformation(vertices);
        if (quad.size())
          {
            this->quadrature_points.insert(
              this->quadrature_points.end(),
              quad.get_points().begin(),
              quad.get_points().end());
            this->weights.insert(
              this->weights.end(),
              quad.get_weights().begin(),
              quad.get_weights().end());
          }
      }
}


template <int dim>
QTellesGen<dim>::QTellesGen (
  const unsigned int n, const Point<dim> &singularity, const unsigned int order)
  :
/**
* In this case we map the standard Gauss Legendre formula using the given
* singularity point coordinates.
**/
  Quadrature<dim>(QTellesGen<dim>(QGauss<1>(n), singularity, order))
{}

template <>
QTellesGen<1>::QTellesGen (
  const Quadrature<1> &base_quad, const Point<1> &singularity, const unsigned int order)
  :
/**
* We explicitly implement the Telles' variable change if dim == 1.
**/
  Quadrature<1>(base_quad)
{
  AssertThrow(order > 0, ExcMessage("The order of the variable change must be positive"));
  AssertThrow(order%2 == 1, ExcMessage("The order of the variable change must be odd"));
  double q = order;
  // The original algorithm is designed for a quadrature interval [-1,1].
  for(unsigned int i = 0; i<quadrature_points.size(); ++i)
  {
    quadrature_points[i][0] = (quadrature_points[i][0]-0.5) * 2.;
    weights[i] *= 2.;
  }
  std::vector<Point<1, long double> > quadrature_points_dummy(quadrature_points.size());
  std::vector<Point<1, long double> > quadrature_points_dummy_2(quadrature_points.size());
  std::vector<double> weights_dummy(weights.size());
  long double s0 = (singularity[0] - 0.5) * 2.;
  // std::cout<<singularity[0] << " "<<s0<<std::endl;
  unsigned int cont = 0;
  // std::cout<<"orginal 1"<<std::endl;
  // for (unsigned int d = 0; d < size(); ++d)
  //   std::cout<<quadrature_points[d][0]<<" "<<weights[d]<<std::endl;
  const double tol = 1e-10;
  for (unsigned int d = 0; d < quadrature_points.size(); ++d)
    {
      if (std::abs(quadrature_points[d][0] - s0) > tol)
        {
          quadrature_points_dummy[d-cont][0] = quadrature_points[d][0];
          quadrature_points_dummy_2[d-cont][0] = quadrature_points[d][0];
          weights_dummy[d-cont] = weights[d];
        }
      else
        {
          // We need to remove the singularity point from the quadrature point
          // list. To do so we use the variable cont.
          cont = 1;
        }

    }

  if (cont == 1)
    {
      quadrature_points.resize(quadrature_points_dummy.size()-1);
      quadrature_points_dummy_2.resize(quadrature_points_dummy.size()-1);
      weights.resize(weights_dummy.size()-1);
      for (unsigned int d = 0; d < quadrature_points.size(); ++d)
        {
          // std::cout<<quadrature_points_dummy[d][0]<<" ";
          quadrature_points_dummy_2[d] = quadrature_points_dummy[d];
          weights[d] = weights_dummy[d];
          // std::cout<<quadrature_points[d][0]<<std::endl;
        }
    }
  // std::cout<<"CONT "<<cont<<std::endl;
  // We need to check if the singularity is at the boundary of the interval.
  long double delta = std::pow(2.,-q) * std::pow(std::pow((1+s0),1./q)+std::pow((1-s0),1./q),q);
  double t0 = (std::pow((1.+s0),1./q)-std::pow((1.-s0),1./q))/(std::pow((1.+s0),1./q)+std::pow((1.-s0),1./q));
  // std::cout<<s0<<" "<<t0<<std::endl;
  // std::cout<<std::pow((1.+s0),1./q)<<" "<<std::pow((1.-s0),1./q)<<" "<<(std::pow((1.+s0),1./q)-std::pow((1.-s0),1./q))<<" "<<(std::pow((1.+s0),1./q)+std::pow((1.-s0),1./q))<<std::endl;
  double t, J;
  // std::cout<<"orginal 2"<<std::endl;
  // for (unsigned int d = 0; d < size(); ++d)
  //   std::cout<<quadrature_points[d][0]<<" "<<weights[d]<<std::endl;
  long double half = 0.5;
  for (unsigned int d = 0; d < quadrature_points.size(); ++d)
  {
    t = quadrature_points_dummy_2[d][0];
    // std::cout<<d<<" ";
    quadrature_points_dummy_2[d][0] = delta * std::pow((t-t0), q) + s0;
    std::cout.precision(20);
    // std::cout<<t<<" "<<t0<<" "<<delta<<" "<<std::pow((t-t0), q)<<" "<<delta * std::pow((t-t0), q)<<" "<<quadrature_points_dummy_2[d][0]<<" "<<quadrature_points_dummy_2[d][0]*half+half<<std::endl;
    J = delta * q * std::pow((t-t0), -1.+q);
    weights[d] *= J;
  }
  // std::cout<<std::endl;
  // for (unsigned int d = 0; d < size(); ++d)
  //   std::cout<<quadrature_points[d][0]<<" "<<weights[d]<<std::endl;

  for (unsigned int d = 0; d < size(); ++d)
  {
      quadrature_points[d][0] = static_cast<double> (quadrature_points_dummy_2[d][0]*half+half);
      weights[d] *= .5;
  }

}

template<>
unsigned int QTellesOnBoundary<2>::quad_size(const Point<2> singularity,
                                          const unsigned int n)
{
  double eps=1e-8;
  bool on_edge=false;
  bool on_vertex=false;
  for (unsigned int i=0; i<2; ++i)
    if ( ( std::abs(singularity[i]  ) < eps ) ||
         ( std::abs(singularity[i]-1) < eps ) )
      on_edge = true;
  if (on_edge && (std::abs( (singularity-Point<2>(.5, .5)).norm_square()-.5)
                  < eps) )
    on_vertex = true;
  if (on_vertex) return (n*n);
  if (on_edge) return (2*n*n);
  return (4*n*n);
}

template<>
QTellesOnBoundary<1>::QTellesOnBoundary(const unsigned int n,
                                  const Point<1> &singularity,
                                  const unsigned int order) :
  Quadrature<1>(n)
{
  QTellesGen<1> telly(n, singularity, order);
  quadrature_points.resize(telly.size());
  weights.resize(telly.size());
  for(unsigned int i = 0; i<telly.size(); ++i)
  {
    quadrature_points[i] = telly.point(i);
    weights[i] = telly.weight(i);
  }
}
template<>
QTellesOnBoundary<2>::QTellesOnBoundary(const unsigned int n,
                                  const Point<2> &singularity,
                                  const unsigned int order) :
  Quadrature<2>(quad_size(singularity, n))
{
  // We treat all the cases in the
  // same way. Split the element in 4
  // pieces, measure the area, if
  // it's relevant, add the
  // quadrature connected to that
  // singularity.
  std::vector<QTellesGen<2> > quads;
  std::vector<Point<2> > origins;
  // Id of the corner with a
  // singularity
  Point<2> p3(1.,1.),p1(1.,0.),p2(0.,1.),p0(0.,0.);


  quads.push_back(QTellesGen<2> (n, p3, order));
  quads.push_back(QTellesGen<2> (n, p2, order));
  quads.push_back(QTellesGen<2> (n, p1, order));
  quads.push_back(QTellesGen<2> (n, p0, order));

  origins.push_back(Point<2>(0.,0.));
  origins.push_back(Point<2>(singularity[0],0.));
  origins.push_back(Point<2>(0.,singularity[1]));
  origins.push_back(singularity);

  // Lexicographical ordering.

  double eps = 1e-8;
  unsigned int q_id = 0; // Current quad point index.
  Tensor<1,2> dist;

  for (unsigned int box=0; box<4; ++box)
    {
      dist = (singularity-GeometryInfo<2>::unit_cell_vertex(box));
      dist = Point<2>(std::abs(dist[0]), std::abs(dist[1]));
      double area = dist[0]*dist[1];
      if (area > eps)
        for (unsigned int q=0; q<quads[box].size(); ++q, ++q_id)
          {
            const Point<2> &qp = quads[box].point(q);
            this->quadrature_points[q_id] =
              origins[box]+
              Point<2>(dist[0]*qp[0], dist[1]*qp[1]);
            this->weights[q_id] = quads[box].weight(q)*area;
          }
    }
}

template <int dim>
QTellesGen<dim>::QTellesGen (
  const Quadrature<1> &base_quad, const Point<dim> &singularity, const unsigned int order)
  :
/**
* We need the explicit implementation if dim == 1. If dim > 1 we use the
* former implementation and apply a tensorial product to obtain the higher
* dimensions.
**/
  Quadrature<dim>(
    dim == 2 ?
    QAnisotropic<dim>(
      QTellesGen<1>(base_quad, Point<1>(singularity[0]), order),
      QTellesGen<1>(base_quad, Point<1>(singularity[1]), order)) :
    dim == 3 ?
    QAnisotropic<dim>(
      QTellesGen<1>(base_quad, Point<1>(singularity[0]), order),
      QTellesGen<1>(base_quad, Point<1>(singularity[1]), order),
      QTellesGen<1>(base_quad, Point<1>(singularity[2]), order)) :
    Quadrature<dim>())
{}


  template <>
  unsigned int MyQGaussOneOverR<2>::quad_size(const Point<2> singularity,
                                            const unsigned int n)
  {
    double eps=1e-8;
    bool on_edge=false;
    bool on_vertex=false;
    for (unsigned int i=0; i<2; ++i)
      if ( ( std::abs(singularity[i]  ) < eps ) ||
           ( std::abs(singularity[i]-1) < eps ) )
        on_edge = true;
    if (on_edge && (std::abs( (singularity-Point<2>(.5, .5)).norm_square()-.5)
                    < eps) )
      on_vertex = true;
    if (on_vertex) return (2*n*n);
    if (on_edge) return (4*n*n);
    return (8*n*n);
  }

  template <>
  MyQGaussOneOverR<2>::MyQGaussOneOverR(const unsigned int n,
                                    const Point<2> singularity,
                                    const bool factor_out_singularity) :
    Quadrature<2>(quad_size(singularity, n))
  {
    // We treat all the cases in the
    // same way. Split the element in 4
    // pieces, measure the area, if
    // it's relevant, add the
    // quadrature connected to that
    // singularity.
    std::vector<MyQGaussOneOverR<2> > quads;
    std::vector<Point<2> > origins;
    // Id of the corner with a
    // singularity
    quads.emplace_back(n, 3, factor_out_singularity);
    quads.emplace_back(n, 2, factor_out_singularity);
    quads.emplace_back(n, 1, factor_out_singularity);
    quads.emplace_back(n, 0, factor_out_singularity);

    origins.emplace_back(0., 0.);
    origins.emplace_back(singularity[0], 0.);
    origins.emplace_back(0., singularity[1]);
    origins.push_back(singularity);

    // Lexicographical ordering.

    double eps = 1e-8;
    unsigned int q_id = 0; // Current quad point index.
    Tensor<1,2> dist;

    for (unsigned int box=0; box<4; ++box)
      {
        dist = (singularity-GeometryInfo<2>::unit_cell_vertex(box));
        dist = Point<2>(std::abs(dist[0]), std::abs(dist[1]));
        double area = dist[0]*dist[1];
        if (area > eps)
          for (unsigned int q=0; q<quads[box].size(); ++q, ++q_id)
            {
              const Point<2> &qp = quads[box].point(q);
              this->quadrature_points[q_id] =
                origins[box]+
                Point<2>(dist[0]*qp[0], dist[1]*qp[1]);
              this->weights[q_id] = quads[box].weight(q)*area;
            }
      }
  }


  template <>
  MyQGaussOneOverR<2>::MyQGaussOneOverR(const unsigned int n,
                                    const unsigned int vertex_index,
                                    const bool factor_out_singularity) :
    Quadrature<2>(2*n *n)
  {
    // This version of the constructor
    // works only for the 4
    // vertices. If you need a more
    // general one, you should use the
    // one with the Point<2> in the
    // constructor.
    Assert(vertex_index <4, ExcIndexRange(vertex_index, 0, 4));

    // Start with the gauss quadrature formula on the (u,v) reference
    // element.
    QGauss<2> gauss(n);

    Assert(gauss.size() == n*n, ExcInternalError());

    // For the moment we only implemented this for the vertices of a
    // quadrilateral. We are planning to do this also for the support
    // points of arbitrary FE_Q elements, to allow the use of this
    // class in boundary element programs with higher order mappings.
    Assert(vertex_index < 4, ExcIndexRange(vertex_index, 0, 4));

    // We create only the first one. All other pieces are rotation of
    // this one.
    // In this case the transformation is
    //
    // (x,y) = (u, u tan(pi/4 v))
    //
    // with Jacobian
    //
    // J = pi/4 R / cos(pi/4 v)
    //
    // And we get rid of R to take into account the singularity,
    // unless specified differently in the constructor.
    std::vector<Point<2> >      &ps = this->quadrature_points;
    std::vector<double>         &ws = this->weights;
    double pi4 = numbers::PI/4;

    for (unsigned int q=0; q<gauss.size(); ++q)
      {
        const Point<2> &gp = gauss.point(q);
        ps[q][0] = gp[0];
        ps[q][1] = gp[0] * std::tan(pi4*gp[1]);
        ws[q]    = gauss.weight(q)*pi4/std::cos(pi4 *gp[1])/std::cos(pi4 *gp[1]);
        if (factor_out_singularity)
          ws[q] *= ps[q][0];//(ps[q]-GeometryInfo<2>::unit_cell_vertex(0)).norm();
        // The other half of the quadrilateral is symmetric with
        // respect to xy plane.
        ws[gauss.size()+q]    = ws[q];
        ps[gauss.size()+q][0] = ps[q][1];
        ps[gauss.size()+q][1] = ps[q][0];
      }

    // Now we distribute these vertices in the correct manner
    double theta = 0;
    switch (vertex_index)
      {
      case 0:
        theta = 0;
        break;
      case 1:
        //
        theta = numbers::PI/2;
        break;
      case 2:
        theta = -numbers::PI/2;
        break;
      case 3:
        theta = numbers::PI;
        break;
      }

    double R00 =  std::cos(theta), R01 = -std::sin(theta);
    double R10 =  std::sin(theta), R11 =  std::cos(theta);

    if (vertex_index != 0)
      for (unsigned int q=0; q<size(); ++q)
        {
          double x = ps[q][0]-.5,  y = ps[q][1]-.5;

          ps[q][0] = R00*x + R01*y + .5;
          ps[q][1] = R10*x + R11*y + .5;
        }
  }

// template class QCarley<1> ;
// template class QCarley<2> ;
// template class QCarley<3> ;

template class QTellesGen<1> ;
template class QTellesGen<2> ;
template class QTellesGen<3> ;

template class QTellesOnBoundary<1>;
template class QTellesOnBoundary<2>;

template class QSimplex<1>;
template class QSimplex<2>;
template class QSimplex<3>;

template class QSplit<1>;
template class QSplit<2>;
template class QSplit<3>;

DEAL_II_NAMESPACE_CLOSE