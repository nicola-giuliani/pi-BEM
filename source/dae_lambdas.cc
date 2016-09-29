#include "dae_lambdas.h"
#include "dae_bem.h"

template <int dim>
Lambdas<dim>::Lambdas ()
{}


template <int dim>
Lambdas<dim>::
Lambdas (DAEBEM<dim> &dae_obj)
  :
  dae (&dae_obj)
{}


template <int dim>
Lambdas<dim>::~Lambdas ()
{}


template <int dim>
void
Lambdas<dim>::
initialize_dae (DAEBEM<dim> &dae_obj)
{
  dae = &dae_obj;
}


template <int dim>
void
Lambdas<dim>::
set_functions_to_default()
{
  create_new_vector = [this]() ->shared_ptr<TrilinosWrappers::MPI::BlockVector>
  {
    return this->dae->create_new_vector();
  };

  residual = [this](const double t,
                    const TrilinosWrappers::MPI::BlockVector &y,
                    const TrilinosWrappers::MPI::BlockVector &y_dot,
                    TrilinosWrappers::MPI::BlockVector &residual) ->int
  {
    return this->dae->residual(t,y,y_dot,residual);
  };

  setup_jacobian = [this](const double t,
                          const TrilinosWrappers::MPI::BlockVector &y,
                          const TrilinosWrappers::MPI::BlockVector &y_dot,
                          const double alpha) ->int
  {
    return this->dae->setup_jacobian(t,y,y_dot,alpha);
  };

  // ida.solve_jacobian_system = [this](const VEC &rhs,
  //                                    VEC &dst) ->int
  // {
  //   return this->solve_jacobian_system(rhs,dst);
  // };

  solve_jacobian_system = [this](const TrilinosWrappers::MPI::BlockVector &rhs,
                                 TrilinosWrappers::MPI::BlockVector &dst) ->int
  {
    return this->dae->solve_jacobian_system(rhs,dst);
  };

  output_step = [this](const double t,
                       const TrilinosWrappers::MPI::BlockVector &y,
                       const TrilinosWrappers::MPI::BlockVector &y_dot,
                       const unsigned int step_number)
  {
    this->dae->output_step(t,y,y_dot,step_number);
  };

  solver_should_restart = [this](const double t,
                                 TrilinosWrappers::MPI::BlockVector &y,
                                 TrilinosWrappers::MPI::BlockVector &y_dot) ->bool
  {
    return this->dae->solver_should_restart(t,y,y_dot);
  };

  differential_components = [this]() ->TrilinosWrappers::MPI::BlockVector &
  {
    return this->dae->differential_components();
  };

  get_local_tolerances = [this]() ->TrilinosWrappers::MPI::BlockVector &
  {
    AssertThrow(false, ExcPureFunctionCalled("Please implement get_local_tolerances function."));
    static auto lt = this->create_new_vector();
    return *lt;
  };

  get_lumped_mass_matrix = [&]() ->TrilinosWrappers::MPI::BlockVector &
  {
    return this->dae->get_lumped_mass_matrix();
  };

  jacobian_vmult = [this](const TrilinosWrappers::MPI::BlockVector &src,
                          TrilinosWrappers::MPI::BlockVector &dst) ->int
  {
    return this->dae->jacobian_vmult(src,dst);
  };

  vector_norm = [this](const TrilinosWrappers::MPI::BlockVector &vector) ->double
  {
    return this->dae->vector_norm(vector);
  };

}

template class Lambdas<3>;
