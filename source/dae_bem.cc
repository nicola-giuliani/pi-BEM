
#include <dae_bem.h>
#include "Teuchos_TimeMonitor.hpp"

using Teuchos::Time;
using Teuchos::TimeMonitor;
using Teuchos::RCP;

RCP<Time> PrepareTimeDAE = Teuchos::TimeMonitor::getNewTimer("PrepareBEMVectors");
RCP<Time> ErrorsTimerDAE = Teuchos::TimeMonitor::getNewTimer("Errors");
RCP<Time> OutputTimerDAE = Teuchos::TimeMonitor::getNewTimer("Output");
RCP<Time> ResidualTimerDAE = Teuchos::TimeMonitor::getNewTimer("Residual");



template <int dim>
void
DAEBEM<dim>::set_functions_to_default()
{
  create_new_vector = [this]() ->shared_ptr<TrilinosWrappers::MPI::BlockVector>
  {
    return this->create_new_vector();
  };

  residual = [this](const double t,
                    const TrilinosWrappers::MPI::BlockVector &y,
                    const TrilinosWrappers::MPI::BlockVector &y_dot,
                    TrilinosWrappers::MPI::BlockVector &residual) ->int
  {
    return this->residual(t,y,y_dot,residual);
  };

  setup_jacobian = [this](const double t,
                          const TrilinosWrappers::MPI::BlockVector &y,
                          const TrilinosWrappers::MPI::BlockVector &y_dot,
                          const double alpha) ->int
  {
    return this->setup_jacobian(t,y,y_dot,alpha);
  };

  solve_jacobian_system = [this](const TrilinosWrappers::MPI::BlockVector &rhs,
                                 TrilinosWrappers::MPI::BlockVector &dst) ->int
  {
    return this->solve_jacobian_system(rhs,dst);
  };

  // output_step = [this](const double t,
  //                      const TrilinosWrappers::MPI::BlockVector &y,
  //                      const TrilinosWrappers::MPI::BlockVector &y_dot,
  //                      const unsigned int step_number)
  // {
  //   this->simulator->output_step(t,y,y_dot,step_number);
  // };

  solver_should_restart = [this](const double t,
                                 TrilinosWrappers::MPI::BlockVector &y,
                                 TrilinosWrappers::MPI::BlockVector &y_dot) ->bool
  {
    return this->solver_should_restart(t,y,y_dot);
  };

  differential_components = [this]() ->TrilinosWrappers::MPI::BlockVector &
  {
    return this->differential_components();
  };

  // get_local_tolerances = [this]() ->TrilinosWrappers::MPI::BlockVector &
  // {
  //   AssertThrow(false, ExcPureFunctionCalled("Please implement get_local_tolerances function."));
  //   static auto lt = this->create_new_vector();
  //   return *lt;
  // };
  //
  // get_lumped_mass_matrix = [&]() ->TrilinosWrappers::MPI::BlockVector &
  // {
  //   static shared_ptr<TrilinosWrappers::MPI::BlockVector> lm;
  //   lm = this->create_new_vector();
  //   this->simulator->get_lumped_mass_matrix(*lm);
  //   return *lm;
  // };

  jacobian_vmult = [this](const TrilinosWrappers::MPI::BlockVector &src,
                          TrilinosWrappers::MPI::BlockVector &dst) ->int
  {
    return this->jacobian_vmult(src,dst);
  };

  // vector_norm = [this](const TrilinosWrappers::MPI::BlockVector &vector) ->double
  // {
  //   return this->simulator->vector_norm(vector);
  // };

}

template <int dim>
int
DAEBEM<dim>::setup_jacobian(const double t,
                                            const TrilinosWrappers::MPI::BlockVector &src_yy,
                                            const TrilinosWrappers::MPI::BlockVector &src_yp,
                                            const TrilinosWrappers::MPI::BlockVector &,
                                            const double alpha)
{

  BlockDynamicSparsityPattern csp(src_yp.n_blocks(), src_yy.n_blocks());

  for(unsigned int i=0; i<src_yp.n_blocks(); ++i)
    for(unsigned int j=0; j<src_yy.n_blocks(); ++j)
      DoFTools::make_sparsity_pattern (bem.dh,
                                       csp.block(i,j),
                                       bem.constraints);
  csp.compress();

  jacobian_sparsity.copy_from(csp);

  jacobian_matrix.reinit(jacobian_sparsity);
  jacobian_preconditioner_matrix.reinit(jacobian_sparsity);

  for(auto i : neumann_set)
    jacobian_matrix.block(0,0).set(i,i,1.0);

  for(auto i : dirichlet_set_bem)
    jacobian_matrix.block(0,0).set(i,i,1.0);

  for(auto i : neumann_dot_set)
    jacobian_matrix.block(0,0).set(i,i,alpha);


  for(auto i : dirichlet_set)
    jacobian_matrix.block(1,1).set(i,i,1.0);

  for(auto i : neumann_set_bem)
    jacobian_matrix.block(1,1).set(i,i,1.0);

  for(auto i : dirichlet_dot_set)
    jacobian_matrix.block(1,1).set(i,i,alpha);


  // for(types::global_dof_index i = 0; i<bem.dh.n_dofs(); ++i)
  // {
  //   jacobian_matrix.block(0,0).set(i,i,1.);
  //   jacobian_matrix.block(0,1).set(i,i,1.);
  //   jacobian_matrix.block(1,0).set(i,i,1.);
  //   jacobian_matrix.block(1,1).set(i,i,1.);
  // }

  // LO JACOBIANO E' DEL RESIDUO RISPETTO ALLE INCOGNITE DI TALE STRONZO MALEDETTO

  // for(unsigned int i=0; i<src_yp.n_blocks(); ++i)
  //   for(unsigned int j=0; j<src_yy.n_blocks(); ++j)
  //     jacobian_preconditioner_matrix.block(i,j).reinit(TrilinosWrappers::PreconditionJacobi(jacobian_matrix.block(i,j)));

  for(types::global_dof_index i = 0; i<bem.dh.n_dofs(); ++i)
  {
    jacobian_preconditioner_matrix.block(0,0).set(i,i,1.);
    // jacobian_preconditioner_matrix.block(0,1).set(i,i,1.);
    // jacobian_preconditioner_matrix.block(1,0).set(i,i,1.);
    jacobian_preconditioner_matrix.block(1,1).set(i,i,1.);
  }





  return 0;
}

template <int dim>
TrilinosWrappers::MPI::BlockVector &
DAEBEM<dim>::differential_components() const
{
  static TrilinosWrappers::MPI::BlockVector diff_comps;
  diff_comps.reinit(solution);

  for(auto i : dirichlet_dot_set)
  {
    diff_comps.block(1)=1.;
  }

  for(auto i : neumann_dot_set)
  {
    diff_comps.block(0)=1.;
  }

  // set_constrained_dofs_to_zero(diff_comps);
  return diff_comps;
}


template <int dim>
bool
DAEBEM<dim>::solver_should_restart(const double t,
                                                   const unsigned int /*step_number*/,
                                                   const double /*h*/,
                                                   TrilinosWrappers::MPI::BlockVector &solution,
                                                   TrilinosWrappers::MPI::BlockVector &solution_dot)
{
  //
  // auto _timer = computing_timer.scoped_timer ("Solver should restart");
  // if (use_space_adaptivity)
  //   {
  //     double max_kelly=0;
  //     auto _timer = computing_timer.scoped_timer ("Compute error estimator");
  //     update_functions_and_constraints(t);
  //
  //     constraints.distribute(solution);
  //     locally_relevant_solution = solution;
  //     constraints_dot.distribute(solution_dot);
  //     locally_relevant_solution_dot = solution_dot;
  //
  //     Vector<float> estimated_error_per_cell (triangulation->n_active_cells());
  //
  //     interface.estimate_error_per_cell(estimated_error_per_cell);
  //
  //     max_kelly = estimated_error_per_cell.linfty_norm();
  //     max_kelly = Utilities::MPI::max(max_kelly, comm);
  //
  //     if (max_kelly > kelly_threshold)
  //
  //       {
  //         pcout << "  ################ restart ######### \n"
  //               << "max_kelly > threshold\n"
  //               << max_kelly  << " >  " << kelly_threshold
  //               << std::endl
  //               << "######################################\n";
  //         pgr.mark_cells(estimated_error_per_cell, *triangulation);
  //
  //         refine_and_transfer_solutions(solution,
  //                                       solution_dot,
  //                                       explicit_solution,
  //                                       locally_relevant_solution,
  //                                       locally_relevant_solution_dot,
  //                                       locally_relevant_explicit_solution,
  //                                       adaptive_refinement);
  //
  //
  //         MPI::COMM_WORLD.Barrier();
  //
  //         return true;
  //       }
  //     else // if max_kelly > kelly_threshold
  //       {
  //
  //         return false;
  //       }
  //
  //   }
  // else // use space adaptivity

    return false;
}


template <int dim>
int
DAEBEM<dim>::solve_jacobian_system(//const double /*t*/,
                                    //               const TrilinosWrappers::MPI::BlockVector &/*y*/,
                                      //             const TrilinosWrappers::MPI::BlockVector &/*y_dot*/,
                                        //           const TrilinosWrappers::MPI::BlockVector &,
                                          //       const double /*alpha*/,
                                                   const TrilinosWrappers::MPI::BlockVector &src,
                                                   TrilinosWrappers::MPI::BlockVector &dst) const
{
  // auto _timer = computing_timer.scoped_timer ("Solve system");
  // set_constrained_dofs_to_zero(dst);
  //
  // typedef dealii::BlockSparseMatrix<double> sMAT;
  // typedef dealii::BlockVector<double> sVEC;
  //
  if (jacobian_direct_resolution == true)
    {
      //
      // SparseDirectUMFPACK inverse;
      // inverse.factorize((sMAT &) *matrices[0]);
      // inverse.vmult((sVEC &)dst, (sVEC &)src);

    }
  else
    {
          unsigned int tot_iteration = 0;
          SolverControl solver_control (1000,
                                        1e-10);

          PrimitiveVectorMemory<TrilinosWrappers::MPI::BlockVector> mem;

          SolverFGMRES<TrilinosWrappers::MPI::BlockVector>
          solver(solver_control, mem,
                 typename SolverFGMRES<TrilinosWrappers::MPI::BlockVector>::AdditionalData(50, true));

          auto S_inv = inverse_operator(jacobian_matrix, solver, jacobian_preconditioner_matrix);
          S_inv.vmult(dst, src);

          tot_iteration += solver_control.last_step();

      }
  return 0;
}



template <int dim>
shared_ptr<TrilinosWrappers::MPI::BlockVector>
DAEBEM<dim>::create_new_vector() const
{
  shared_ptr<TrilinosWrappers::MPI::BlockVector> ret = SP(new TrilinosWrappers::MPI::BlockVector(solution));
  *ret *= 0;
  return ret;
}

template <int dim>
int
DAEBEM<dim>::residual(const double t,
                      const TrilinosWrappers::MPI::BlockVector &solution,
                      const TrilinosWrappers::MPI::BlockVector &solution_dot,
                      TrilinosWrappers::MPI::BlockVector &dst)
{
  Teuchos::TimeMonitor LocalTimer(*ResidualTimerDAE);

  // spacchetto phi e phin tanto lui se lo gestisce;
  // prendo solution e lo copio e lo giro a bem problem;
  // bem.solve()//con phi e phin spacchettate phi_bem, phin_bem
  // solution - condizioni esatte se e' algebrico
  // solution_dot - dot_esatto se e' differenziale


  TrilinosWrappers::MPI::Vector phi_bem(solution.block(1));
  TrilinosWrappers::MPI::Vector dphi_dn_bem(solution.block(0));

  // settare i dirichlet/neumann bem
  prepare_bem_vectors(t);

  TrilinosWrappers::MPI::Vector tmp_rhs_bem(this_cpu_set, mpi_communicator);
  for(auto i : neumann_set_bem)
  {
    tmp_rhs_bem[i]=solution.block(0)[i];
  }
  for(auto i : dirichlet_set_bem)
  {
    tmp_rhs_bem[i]=solution.block(1)[i];
  }

  bem.solve(phi_bem, dphi_dn_bem, tmp_rhs_bem);

  // Where we impose Dirichlet BC we check that the solution is equal to the prescribed one
  for(auto i : dirichlet_set)
  {
    dst.block(1)[i]=solution.block(1)[i]-phi[i];
  }
  // Where we impose some kind of Neumann (dot or nor) BC we check that the solution is equal to the BEM one
  for(auto i : neumann_set_bem)
  {
    dst.block(1)[i]=solution.block(1)[i]-phi_bem[i];
  }
  // Where we impose Dirichlet dot BC we check that the solution is equal to the prescribed one
  for(auto i : dirichlet_dot_set)
  {
    dst.block(1)[i]=solution_dot.block(1)[i]-phi_dot[i];
  }
  // Where we impose Neumann BC we check that the solution is equal to the prescribed one
  for(auto i : neumann_set)
  {
    dst.block(0)[i]=solution.block(0)[i]-dphi_dn[i];
  }
  // Where we impose some kind of Dirichlet (dot or nor) BC we check that the solution is equal to the BEM one
  for(auto i : dirichlet_set_bem)
  {
    dst.block(0)[i]=solution.block(0)[i]-dphi_dn_bem[i];
  }
  // Where we impose Neumann dot BC we check that the solution is equal to the prescribed one
  for(auto i : neumann_dot_set)
  {
    dst.block(0)[i]=solution_dot.block(0)[i]-dphi_dn_dot[i];
  }



}







template <int dim>
void DAEBEM<dim>::declare_parameters(ParameterHandler &prm)
{

  prm.declare_entry("Output file name", "result", Patterns::Anything());

  prm.enter_subsection("potential_gradient function 2d");
  {
    Functions::ParsedFunction<2>::declare_parameters(prm, 2);
    prm.set("Function expression", "1; 1");
  }
  prm.leave_subsection();

  // prm.enter_subsection("Potential Gradient function 3d");
  // {
  //   Functions::ParsedFunction<3>::declare_parameters(prm, 3);
  //   prm.set("Variable names","x,y,z,t");
  //   prm.set("Function expression", "1; 1; 1");
  // }
  // prm.leave_subsection();


  // prm.enter_subsection("Potential Gradient dot function 3d");
  // {
  //   Functions::ParsedFunction<3>::declare_parameters(prm, 3);
  //   prm.set("Variable names","x,y,z,t");
  //   prm.set("Function expression", "1; 0; 0");
  // }
  // prm.leave_subsection();


  // prm.enter_subsection("Potential 2d");
  // {
  //   Functions::ParsedFunction<2>::declare_parameters(prm);
  //   prm.set("Function expression", "x+y");
  // }
  // prm.leave_subsection();
  //
  // prm.enter_subsection("Potential 3d");
  // {
  //   Functions::ParsedFunction<3>::declare_parameters(prm);
  //   prm.set("Variable names","x,y,z,t");
  //   prm.set("Function expression", "t*x+y+z");
  // }
  // prm.leave_subsection();
  //
  // prm.enter_subsection("Potential dot");
  // {
  //   Functions::ParsedFunction<3>::declare_parameters(prm);
  //   prm.set("Variable names","x,y,z,t");
  //   prm.set("Function expression", "x");
  // }
  // prm.leave_subsection();

  add_parameter(prm, &jacobian_direct_resolution, "Solve the Jacobian using direct method","false", Patterns::Bool());



}

template <int dim>
void DAEBEM<dim>::parse_parameters(ParameterHandler &prm)
{

  output_file_name = prm.get("Output file name");


  // prm.enter_subsection(std::string("Potential Gradient function")+
  //                      Utilities::int_to_string(dim)+std::string("d"));
  // {
  //   potential_gradient.parse_parameters(prm);
  // }
  // prm.leave_subsection();
  //
  // prm.enter_subsection(std::string("Potential Gradient dot function")+
  //                      Utilities::int_to_string(dim)+std::string("d"));
  // {
  //   potential_gradient_dot.parse_parameters(prm);
  // }
  // prm.leave_subsection();
  //
  // prm.enter_subsection(std::string("Potential")+
  //                      Utilities::int_to_string(dim)+std::string("d"));
  // {
  //   potential.parse_parameters(prm);
  // }
  //
  // prm.enter_subsection(std::string("Potential dot")+
  //                      Utilities::int_to_string(dim)+std::string("d"));
  // {
  //   potential_dot.parse_parameters(prm);
  // }
  // prm.leave_subsection();

}



/*template <int dim>
double* DAEBEM<dim>::initial_conditions() {

  initial_wave_shape.set_time(initial_time);
  initial_wave_potential.set_time(initial_time);
  potential_gradient.set_time(initial_time);

  Vector<double> instantpotential_gradientValue(dim);
  Point<dim> zero(0,0,0);
  potential_gradient.vector_value(zero,instantpotential_gradientValue);
  bem.pcout<<std::endl<<"Simulation time= "<<initial_time<<"   Vinf= ";
  instantpotential_gradientValue.print(cout,4,false,true);
  bem.pcout<<std::endl;

  dofs_number = bem.dh.n_dofs()+bem.gradient_dh.n_dofs();

  phi.reinit(bem.dh.n_dofs());
  dphi_dn.reinit(bem.dh.n_dofs());
  tmp_rhs.reinit(bem.dh.n_dofs());

  DXDt_and_DphiDt_vector.resize(n_dofs());

  std::vector<Point<dim> > support_points(bem.dh.n_dofs());
  DoFTools::map_dofs_to_support_points<dim-1, dim>( bem.mapping, bem.dh, support_points);
  std::vector<Point<dim> > gradient_support_points(bem.gradient_dh.n_dofs());
  DoFTools::map_dofs_to_support_points<dim-1, dim>( bem.mapping, bem.gradient_dh, gradient_support_points);


  unsigned int j = dim-1;
  for(unsigned int i=j; i<bem.gradient_dh.n_dofs(); i=i+dim)
      {
      DXDt_and_DphiDt_vector[i] = initial_wave_shape.value(gradient_support_points[i]);
      //bem.pcout<<DXDt_and_DphiDt_vector[i]<<std::endl;
      }

   for (unsigned int i=0; i<bem.dh.n_dofs(); i++)
      {
      DXDt_and_DphiDt_vector[i+bem.gradient_dh.n_dofs()] = initial_wave_potential.value(gradient_support_points[i]);
      //bem.pcout<<DXDt_and_DphiDt_vector[i+bem.gradient_dh.n_dofs()]<<std::endl;
      }

   max_y_coor_value = 0;
   for (unsigned int i=0; i < bem.dh.n_dofs(); i++)
       {
       //for printout
       //bem.pcout<<"Node "<<i<< "["<<support_points[i]<<"] "<<std::endl;
       max_y_coor_value = std::max(max_y_coor_value,std::abs(support_points[i](1)));
       }

  std::string filename = ( output_file_name + "_" +
         Utilities::int_to_string(0) +
         ".vtk" );

  output_results(filename);

  return &DXDt_and_DphiDt_vector[0];
} //*/




template <int dim>
void DAEBEM<dim>::solve_problem()
{


  potential.set_time(0);
  potential_gradient.set_time(0);

  const types::global_dof_index n_dofs =  bem.dh.n_dofs();
  std::vector<types::subdomain_id> dofs_domain_association(n_dofs);
  DoFTools::get_subdomain_association   (bem.dh,dofs_domain_association);
  this_cpu_set.clear();
  this_cpu_set = bem.this_cpu_set;
  this_cpu_set.compress();

  this_cpu_set_complete.resize(4);
  this_cpu_set_double.resize(2);

  for(unsigned int i=0; i<this_cpu_set_double.size(); ++i)
  {
    this_cpu_set_complete[i].clear();
    this_cpu_set_complete[i] = bem.this_cpu_set;
    this_cpu_set_complete[i].compress();

    this_cpu_set_double[i].clear();
    this_cpu_set_double[i] = bem.this_cpu_set;
    this_cpu_set_double[i].compress();


  }

  for(unsigned int i=this_cpu_set_double.size(); i<this_cpu_set_complete.size(); ++i)
  {
    this_cpu_set_complete[i].clear();
    this_cpu_set_complete[i] = bem.this_cpu_set;
    this_cpu_set_complete[i].compress();

  }



  phi.reinit(this_cpu_set,mpi_communicator);
  dphi_dn.reinit(this_cpu_set,mpi_communicator);
  phi_dot.reinit(this_cpu_set,mpi_communicator);
  dphi_dn_dot.reinit(this_cpu_set,mpi_communicator);

  tmp_rhs.reinit(this_cpu_set,mpi_communicator);

  solution.reinit(this_cpu_set_double, mpi_communicator);
  solution_dot.reinit(this_cpu_set_double, mpi_communicator);

  pcout<<"Computing normal vector"<<std::endl;
  bem.compute_normals();

  // ida(*this);
  ida.residual = this->residual;
   ida.setup_jacobian = this->setup_jacobian;
   ida.solver_should_restart = this->solver_should_restart;
   ida.solve_jacobian_system = this->solve_jacobian_system;
   ida.output_step = this->output_step;
   ida.differential_components = this->differential_components;
   ida.solve_dae(solution, solution_dot);
  // if (time_stepper == "ida")
  //   ida.start_ode(solution, solution_dot, max_time_iterations);
  // else if (time_stepper == "euler")
  //   euler.start_ode(solution, solution_dot);

  // prepare_bem_vectors();
  //
  //
  // bem.solve(phi, dphi_dn, tmp_rhs);

  // bem.compute_gradients(phi, dphi_dn);



}

template <int dim>
const TrilinosWrappers::MPI::Vector &DAEBEM<dim>::get_phi()
{
  return phi;
}

template <int dim>
const TrilinosWrappers::MPI::Vector &DAEBEM<dim>::get_dphi_dn()
{
  return dphi_dn;
}


template <int dim>
void DAEBEM<dim>::compute_dae_cache()
{
  cell_it
  cell = bem.dh.begin_active(),
  endc = bem.dh.end();

  const unsigned int   dofs_per_cell   = bem.fe->dofs_per_cell;
  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
  FEValues<dim-1,dim> fe_v(*bem.mapping, *bem.fe, *bem.quadrature,
                           update_values |
                           update_cell_normal_vectors |
                           update_quadrature_points |
                           update_JxW_values);


  for (cell = bem.dh.begin_active(); cell != endc; ++cell)
  {
    cell->get_dof_indices(local_dof_indices);
      if(cell->material_id() == 0)// Neumann node
      {
        for (unsigned int j=0; j<bem.fe->dofs_per_cell; ++j)
          if(bem.this_cpu_set.is_element(j))
          {
            algebraic_set.add_index(j);
            neumann_set.add_index(j);
            neumann_set_bem.add_index(j);
          }
      }
      else if(cell->material_id() == 1)//Dirichlet node
      {
        for (unsigned int j=0; j<bem.fe->dofs_per_cell; ++j)
          if(bem.this_cpu_set.is_element(j))
          {
            algebraic_set.add_index(j);
            dirichlet_set.add_index(j);
            dirichlet_set_bem.add_index(j);
          }

      }
      else if(cell->material_id() == 2)//Neumann differential node
      {
        for (unsigned int j=0; j<bem.fe->dofs_per_cell; ++j)
          if(bem.this_cpu_set.is_element(j))
          {
            differential_set.add_index(j);
            neumann_set_bem.add_index(j);
            neumann_dot_set.add_index(j);
          }

      }
      else if(cell->material_id() == 3)//Dirichlet differential node
      {
        for (unsigned int j=0; j<bem.fe->dofs_per_cell; ++j)
          if(bem.this_cpu_set.is_element(j))
          {
            differential_set.add_index(j);
            dirichlet_set_bem.add_index(j);
            dirichlet_dot_set.add_index(j);
          }

      }

  }

}

template <int dim>
void DAEBEM<dim>::prepare_bem_vectors(double t)
{
  Teuchos::TimeMonitor LocalTimer(*PrepareTimeDAE);
  // bem.compute_normals();
  const types::global_dof_index n_dofs =  bem.dh.n_dofs();

  phi.reinit(this_cpu_set,mpi_communicator);
  dphi_dn.reinit(this_cpu_set,mpi_communicator);
  phi_dot.reinit(this_cpu_set,mpi_communicator);
  dphi_dn_dot.reinit(this_cpu_set,mpi_communicator);
  // tmp_rhs.reinit(this_cpu_set,mpi_communicator);


  std::vector<Point<dim> > support_points(n_dofs);
  DoFTools::map_dofs_to_support_points<dim-1, dim>( *bem.mapping, bem.dh, support_points);

  std::vector<Point<dim> > vec_support_points(bem.gradient_dh.n_dofs());
  DoFTools::map_dofs_to_support_points<dim-1, dim>( *bem.mapping, bem.gradient_dh, vec_support_points);

  cell_it
  cell = bem.dh.begin_active(),
  endc = bem.dh.end();

  for(auto i : dirichlet_set)
  {
    phi[i] = potential.value(support_points[i]);
  }
  for(auto i : neumann_set)
  {
    Vector<double> imposed_pot_grad(dim);
    potential_gradient.vector_value(support_points[i],imposed_pot_grad);
    dphi_dn[i]=0.;
    types::global_dof_index dummy = bem.sub_wise_to_original[i];
    for(unsigned int d=0; d<dim; ++d)
    {
      types::global_dof_index vec_index = bem.vec_original_to_sub_wise[bem.gradient_dh.n_dofs()/dim*d+dummy];//bem.vector_start_per_process[this_mpi_process] + d*bem.this_cpu_set.n_elements() + local_dof_indices[j]-bem.start_per_process[this_mpi_process];//bem.gradient_dh.n_dofs()/dim*d+local_dof_indices[j];//bem.vector_start_per_process[this_mpi_process]+((local_dof_indices[j]-bem.start_per_process[this_mpi_process])*dim+d); //local_dof_indices[j]*dim+d;
      // std::cout<<this_mpi_process<<" "<<support_points[local_dof_indices[j]]<<" "<<vec_support_points[vec_index]<<std::endl;
      Assert(bem.vector_this_cpu_set.is_element(vec_index), ExcMessage("vector cpu set and cpu set are inconsistent"));
      // Assert(support_points[local_dof_indices[j]]==vec_support_points[vec_index], ExcMessage("the support points of dh and gradient_dh are different"));
      dphi_dn[i] += imposed_pot_grad[d]*bem.vector_normals_solution[vec_index];
    }
  }
  for(auto i : dirichlet_dot_set)
  {
    phi_dot[i] = potential_dot.value(support_points[i], 0);
  }
  for(auto i : neumann_dot_set)
  {
    dphi_dn_dot[i] = 0;
    Vector<double> imposed_pot_grad_dot(dim);
    potential_gradient.vector_value(support_points[i],imposed_pot_grad_dot);
    dphi_dn[i]=0.;
    types::global_dof_index dummy = bem.sub_wise_to_original[i];
    for(unsigned int d=0; d<dim; ++d)
    {
      types::global_dof_index vec_index = bem.vec_original_to_sub_wise[bem.gradient_dh.n_dofs()/dim*d+dummy];//bem.vector_start_per_process[this_mpi_process] + d*bem.this_cpu_set.n_elements() + local_dof_indices[j]-bem.start_per_process[this_mpi_process];//bem.gradient_dh.n_dofs()/dim*d+local_dof_indices[j];//bem.vector_start_per_process[this_mpi_process]+((local_dof_indices[j]-bem.start_per_process[this_mpi_process])*dim+d); //local_dof_indices[j]*dim+d;
      // std::cout<<this_mpi_process<<" "<<support_points[local_dof_indices[j]]<<" "<<vec_support_points[vec_index]<<std::endl;
      Assert(bem.vector_this_cpu_set.is_element(vec_index), ExcMessage("vector cpu set and cpu set are inconsistent"));
      // Assert(support_points[local_dof_indices[j]]==vec_support_points[vec_index], ExcMessage("the support points of dh and gradient_dh are different"));
      dphi_dn_dot[i] += imposed_pot_grad_dot[d]*bem.vector_normals_solution[vec_index];
    }

  }

}

template <int dim>
void DAEBEM<dim>::compute_errors()
{
  Teuchos::TimeMonitor LocalTimer(*ErrorsTimerDAE);

  // We still need to communicate our results to compute the errors.
  bem.compute_gradients(phi,dphi_dn);
  Vector<double> localized_gradient_solution(bem.vector_gradients_solution);//vector_gradients_solution
  Vector<double> localized_phi(phi);
  Vector<double> localized_dphi_dn(dphi_dn);
  Vector<double> localised_normals(bem.vector_normals_solution);
  // Vector<double> localised_alpha(bem.alpha);
  // We let only the first processor do the error computations
  if (this_mpi_process == 0)
    {
      // for(auto i : localised_alpha.locally_owned_elements())
      //   localised_alpha[i] -= 0.5;

      Vector<double> grad_difference_per_cell (comp_dom.tria.n_active_cells());
      std::vector<Point<dim> > support_points(bem.dh.n_dofs());
      double phi_max_error;// = localized_phi.linfty_norm();
      Vector<double> difference_per_cell (comp_dom.tria.n_active_cells());
      DoFTools::map_dofs_to_support_points<dim-1, dim>( *bem.mapping, bem.dh, support_points);

      if (!have_dirichlet_bc)
        {
          std::vector<double> exact_sol(bem.dh.n_dofs());
          Vector<double> exact_sol_deal(bem.dh.n_dofs());
          potential.value_list(support_points,exact_sol);
          for (auto i : exact_sol_deal.locally_owned_elements())
            exact_sol_deal[bem.original_to_sub_wise[i]] = exact_sol[bem.original_to_sub_wise[i]];
          auto exact_mean = VectorTools::compute_mean_value(*bem.mapping,bem.dh,QGauss<(dim-1)>(2*(2*bem.fe->degree+1)), exact_sol_deal, 0);
          exact_sol_deal.add(-exact_mean);
          auto my_mean = VectorTools::compute_mean_value(*bem.mapping,bem.dh,QGauss<(dim-1)>(2*(2*bem.fe->degree+1)), localized_phi, 0);
          localized_phi.add(-my_mean);
          localized_phi.sadd(1.,-1.,exact_sol_deal);
          std::cout<<"Extracting the mean value from phi solution"<<std::endl;
          VectorTools::integrate_difference (*bem.mapping, bem.dh, localized_phi,
                                             ZeroFunction<dim, double> (1),
                                             difference_per_cell,
                                             QGauss<(dim-1)>(2*(2*bem.fe->degree+1)),
                                             VectorTools::L2_norm);

          phi_max_error = localized_phi.linfty_norm();


        }
      else
        {
          VectorTools::integrate_difference (*bem.mapping, bem.dh, localized_phi,
                                             potential,
                                             difference_per_cell,
                                             QGauss<(dim-1)>(2*(2*bem.fe->degree+1)),
                                             VectorTools::L2_norm);

          phi_max_error = difference_per_cell.linfty_norm();

        }
      VectorTools::integrate_difference (*bem.mapping, bem.gradient_dh, localized_gradient_solution,
                                         potential_gradient,
                                         grad_difference_per_cell,
                                         QGauss<(dim-1)>(2*(2*bem.fe->degree+1)),
                                         VectorTools::L2_norm);
      const double grad_L2_error = grad_difference_per_cell.l2_norm();

      const double L2_error = difference_per_cell.l2_norm();


      Vector<double> vector_gradients_node_error(bem.gradient_dh.n_dofs());
      std::vector<Vector<double> > grads_nodes_errs(bem.dh.n_dofs(),Vector<double>(dim));
      potential_gradient.vector_value_list(support_points,grads_nodes_errs);
      for (types::global_dof_index d=0; d<dim; ++d)
        for (types::global_dof_index i=0; i<bem.dh.n_dofs(); ++i)
          vector_gradients_node_error(bem.vec_original_to_sub_wise[d*bem.dh.n_dofs()+i]) = grads_nodes_errs[bem.original_to_sub_wise[i]](d);
      vector_gradients_node_error*=-1.0;
      vector_gradients_node_error.add(1.,localized_gradient_solution);

      Vector<double> phi_node_error(bem.dh.n_dofs());
      std::vector<double> phi_nodes_errs(bem.dh.n_dofs());
      potential.value_list(support_points,phi_nodes_errs);
      for (types::global_dof_index i=0; i<bem.dh.n_dofs(); ++i)
        phi_node_error(i) = phi_nodes_errs[i];

      phi_node_error*=-1.0;
      phi_node_error.add(1.,localized_phi);

      Vector<double> dphi_dn_node_error(bem.dh.n_dofs());
      std::vector<Vector<double> > dphi_dn_nodes_errs(bem.dh.n_dofs(), Vector<double> (dim));
      potential_gradient.vector_value_list(support_points,dphi_dn_nodes_errs);
      dphi_dn_node_error = 0.;
      for (types::global_dof_index i=0; i<bem.dh.n_dofs(); ++i)
        {
          // dphi_dn_node_error[i] = 0.;
          for (unsigned int d=0; d<dim; ++d)
            {
              dphi_dn_node_error[bem.original_to_sub_wise[i]] += localised_normals[bem.vec_original_to_sub_wise[i+d*bem.dh.n_dofs()]] * dphi_dn_nodes_errs[bem.original_to_sub_wise[i]][d];


            }
        }
      dphi_dn_node_error*=-1.0;
      dphi_dn_node_error.add(1.,localized_dphi_dn);

      // dphi_dn_node_error.print(std::cout);
      Vector<double> difference_per_cell_2(comp_dom.tria.n_active_cells());
      VectorTools::integrate_difference (*bem.mapping, bem.dh, dphi_dn_node_error,
                                         ZeroFunction<dim, double> (1),
                                         difference_per_cell_2,
                                         QGauss<(dim-1)>(2*(2*bem.fe->degree+1)),
                                         VectorTools::L2_norm);
      const double dphi_dn_L2_error = difference_per_cell_2.l2_norm();

      const double grad_phi_max_error = vector_gradients_node_error.linfty_norm();
      const types::global_dof_index n_active_cells=comp_dom.tria.n_active_cells();
      const types::global_dof_index n_dofs=bem.dh.n_dofs();

      pcout << "   Number of active cells:       "
            << n_active_cells
            << std::endl
            << "   Number of degrees of freedom: "
            << n_dofs
            << std::endl
            ;

      pcout<<"Phi Nodes error L_inf norm: "<<phi_max_error<<std::endl;
      pcout<<"Phi Cells error L_2 norm: "<<L2_error<<std::endl;
      pcout<<"dPhidN Nodes error L_inf norm: "<<dphi_dn_node_error.linfty_norm()<<std::endl;
      pcout<<"dPhidN Nodes error L_2 norm: "<<dphi_dn_L2_error<<std::endl;
      pcout<<"Phi Nodes Gradient error L_inf norm: "<<grad_phi_max_error<<std::endl;
      pcout<<"Phi Cells Gradient  error L_2 norm: "<<grad_L2_error<<std::endl;
      // pcout<<"alpha Nodes error L_inf norm: "<<localised_alpha.linfty_norm()<<std::endl;
      // pcout<<"alpha Nodes error L_2 norm: "<<localised_alpha.l2_norm()<<std::endl;

      std::string filename_vector = "vector_error.vtu";
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation
      (dim, DataComponentInterpretation::component_is_part_of_vector);
      DataOut<dim-1, DoFHandler<dim-1, dim> > dataout_vector;
      dataout_vector.attach_dof_handler(bem.gradient_dh);
      dataout_vector.add_data_vector(vector_gradients_node_error, std::vector<std::string > (dim,"phi_gradient_error"), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);

      dataout_vector.build_patches(*bem.mapping,
                                   bem.mapping_degree,
                                   DataOut<dim-1, DoFHandler<dim-1, dim> >::curved_inner_cells);

      std::ofstream file_vector(filename_vector.c_str());

      dataout_vector.write_vtu(file_vector);

      std::string filename_scalar = "scalar_error.vtu";
      DataOut<dim-1, DoFHandler<dim-1, dim> > dataout_scalar;
      dataout_scalar.attach_dof_handler(bem.dh);
      dataout_scalar.add_data_vector(phi_node_error, std::vector<std::string > (1,"phi_error"), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data);
      dataout_scalar.add_data_vector(dphi_dn_node_error, std::vector<std::string > (1,"dphi_dn_error"), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data);
      dataout_scalar.build_patches(*bem.mapping,
                                   bem.mapping_degree,
                                   DataOut<dim-1, DoFHandler<dim-1, dim> >::curved_inner_cells);

      std::ofstream file_scalar(filename_scalar.c_str());
      dataout_scalar.write_vtu(file_scalar);

    }
}

template <int dim>
void DAEBEM<dim>::output_results(const std::string filename)
{
  Teuchos::TimeMonitor LocalTimer(*OutputTimerDAE);

  // At the time being the output is not running in parallel with saks
  // const Vector<double> localized_phi (phi);
  // const Vector<double> localized_dphi_dn (dphi_dn);
  // const Vector<double> localized_alpha (bem.alpha);
  // const Vector<double> localized_gradients (bem.vector_gradients_solution);
  // const Vector<double> localized_normals (bem.vector_normals_solution);
  //
  // if(this_mpi_process == 0)
  // {
  //   data_out_scalar.prepare_data_output(bem.dh);
  //   data_out_scalar.add_data_vector (localized_phi, "phi");
  //   data_out_scalar.add_data_vector (localized_dphi_dn, "dphidn");
  //   data_out_scalar.add_data_vector (localized_alpha, "alpha");
  //   data_out_scalar.write_data_and_clear("", bem.mapping);
  //
  //   data_out_vector.prepare_data_output(bem.gradient_dh);
  //   data_out_vector.add_data_vector (localized_gradients, "gradient");
  //   data_out_vector.add_data_vector (localized_normals, "normal");
  //   data_out_vector.write_data_and_clear("", bem.mapping);
  // }

  // Even for the output we need to serialize the code and then perform the
  // output only on the first processor.
  const Vector<double> localized_phi (phi);
  const Vector<double> localized_dphi_dn (dphi_dn);
  const Vector<double> localized_alpha (bem.alpha);
  const Vector<double> localized_gradients (bem.vector_gradients_solution);
  const Vector<double> localized_surf_gradients (bem.vector_surface_gradients_solution);
  const Vector<double> localized_normals (bem.vector_normals_solution);
  // localized_dphi_dn.print(std::cout);
  // localized_phi.print(std::cout);
  if (this_mpi_process == 0)
    {
      std::string filename_scalar, filename_vector;
      filename_scalar = filename + "_scalar_results" + ".vtu";
      filename_vector = filename + "_vector_results" + ".vtu";

      std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation
      (dim, DataComponentInterpretation::component_is_part_of_vector);

      DataOut<dim-1, DoFHandler<dim-1, dim> > dataout_scalar;
      DataOut<dim-1, DoFHandler<dim-1, dim> > dataout_vector;

      dataout_scalar.attach_dof_handler(bem.dh);
      dataout_vector.attach_dof_handler(bem.gradient_dh);



      dataout_scalar.add_data_vector(localized_phi, "phi", DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data);
      dataout_scalar.add_data_vector(localized_dphi_dn, "dphi_dn", DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data);
      dataout_scalar.add_data_vector(localized_alpha, "alpha", DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data);

      dataout_vector.add_data_vector(localized_gradients, std::vector<std::string > (dim,"phi_gradient"), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);
      dataout_vector.add_data_vector(localized_surf_gradients, std::vector<std::string > (dim,"phi_surf_gradient"), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);
      dataout_vector.add_data_vector(localized_normals, std::vector<std::string > (dim,"normals_at_nodes"), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);


      dataout_scalar.build_patches(*bem.mapping,
                                   bem.mapping_degree,
                                   DataOut<dim-1, DoFHandler<dim-1, dim> >::curved_inner_cells);

      std::ofstream file_scalar(filename_scalar.c_str());

      dataout_scalar.write_vtu(file_scalar);

      dataout_vector.build_patches(*bem.mapping,
                                   bem.mapping_degree,
                                   DataOut<dim-1, DoFHandler<dim-1, dim> >::curved_inner_cells);

      std::ofstream file_vector(filename_vector.c_str());

      dataout_vector.write_vtu(file_vector);

    }

}

template class DAEBEM<3>;
