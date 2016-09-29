
#include <dae_bem.h>
#include "Teuchos_TimeMonitor.hpp"
#include <ida/ida_impl.h>

using Teuchos::Time;
using Teuchos::TimeMonitor;
using Teuchos::RCP;

RCP<Time> PrepareTimeDAE = Teuchos::TimeMonitor::getNewTimer("PrepareBEMVectors");
RCP<Time> ErrorsTimerDAE = Teuchos::TimeMonitor::getNewTimer("Errors");
RCP<Time> OutputTimerDAE = Teuchos::TimeMonitor::getNewTimer("Output");
RCP<Time> ResidualTimerDAE = Teuchos::TimeMonitor::getNewTimer("Residual");



template <int dim>
int
DAEBEM<dim>::setup_jacobian(const double t,
                                            const TrilinosWrappers::MPI::BlockVector &src_yy,
                                            const TrilinosWrappers::MPI::BlockVector &src_yp,
                                            const double alpha)
{
  pcout<<"setting up Jacobian"<<std::endl;
  BlockDynamicSparsityPattern csp(src_yp.n_blocks(), src_yy.n_blocks());

  for(unsigned int i=0; i<src_yp.n_blocks(); ++i)
    for(unsigned int j=0; j<src_yy.n_blocks(); ++j)
    {
      csp.block(i,j).reinit(bem.dh.n_dofs(),bem.dh.n_dofs());
      DoFTools::make_sparsity_pattern (bem.dh,
                                       csp.block(i,j),
                                       bem.constraints);
                                     }
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
void
DAEBEM<dim>::set_dae_initial_conditions(TrilinosWrappers::MPI::BlockVector &xxx, TrilinosWrappers::MPI::BlockVector &xxx_dot)
{
    get_boundary_conditions(0.);
    TrilinosWrappers::MPI::Vector phi_bem(xxx.block(1));
    TrilinosWrappers::MPI::Vector dphi_dn_bem(xxx.block(0));

    TrilinosWrappers::MPI::Vector tmp_rhs_bem(this_cpu_set, mpi_communicator);
    for(auto i : neumann_set_bem)
    {
      tmp_rhs_bem[i]=dphi_dn[i];
      xxx.block(0)[i]=dphi_dn[i];
    }
    for(auto i : dirichlet_set_bem)
    {
      tmp_rhs_bem[i]=phi[i];
      xxx.block(1)[i]=phi[i];
    }
    // phi.print(std::cout);
    // dphi_dn.print(std::cout);
    // tmp_rhs_bem.print(std::cout);
    bem.solve(phi_bem, dphi_dn_bem, tmp_rhs_bem);
    // Where we impose Dirichlet BC we check that the solution is equal to the prescribed one
    for(auto i : dirichlet_set)
    {
      xxx.block(0)[i]=dphi_dn_bem[i];
    }
    // Where we impose some kind of Neumann (dot or nor) BC we check that the solution is equal to the BEM one
    for(auto i : neumann_set_bem)
    {
      xxx.block(1)[i]=phi_bem[i];
    }
    // Where we impose Dirichlet dot BC we check that the solution is equal to the prescribed one
    for(auto i : dirichlet_dot_set)
    {
      xxx_dot.block(1)[i]=phi_dot[i];
    }

    // Where we impose Neumann BC we check that the solution is equal to the prescribed one
    for(auto i : neumann_set)
    {
      xxx.block(0)[i]=dphi_dn[i];
    }
    // Where we impose some kind of Dirichlet (dot or nor) BC we check that the solution is equal to the BEM one
    for(auto i : dirichlet_set_bem)
    {
      xxx.block(0)[i]=dphi_dn_bem[i];
    }
    // Where we impose Neumann dot BC we check that the solution is equal to the prescribed one
    for(auto i : neumann_dot_set)
    {
      xxx_dot.block(0)[i]=dphi_dn_dot[i];
    }
    xxx.block(0).compress(VectorOperation::insert);
    xxx.block(1).compress(VectorOperation::insert);
    xxx_dot.block(0).compress(VectorOperation::insert);
    xxx_dot.block(1).compress(VectorOperation::insert);


}

template <int dim>
int DAEBEM<dim>::jacobian_vmult(const TrilinosWrappers::MPI::BlockVector &src,
                            TrilinosWrappers::MPI::BlockVector &dst) const
{
  std::cout<<jacobian_matrix.n_block_rows()<<" "<<dst.n_blocks()<<std::endl;
  std::cout<<jacobian_matrix.n_block_cols()<<" "<<src.n_blocks()<<std::endl;
    jacobian_matrix.vmult(dst,src);

    return 0;

}

template <int dim>
double DAEBEM<dim>::vector_norm (const  TrilinosWrappers::MPI::BlockVector &v) const
{
  return v.l2_norm();
}

template <int dim>
void DAEBEM<dim>::output_step(const double t,
                         const TrilinosWrappers::MPI::BlockVector &solution,
                         const TrilinosWrappers::MPI::BlockVector &solution_dot,
                         const unsigned int step_number)
{
  const Vector<double> localized_phi (solution.block(1));
  const Vector<double> localized_dphi_dn (solution.block(0));
  const Vector<double> localized_phi_dot (solution_dot.block(1));
  const Vector<double> localized_dphi_dn_dot (solution_dot.block(0));
  // localized_dphi_dn.print(std::cout);
  // localized_phi.print(std::cout);

  if (this_mpi_process == 0)
    {
      potential.set_time(t);
      Vector<double> phi_ex(bem.dh.n_dofs());
      std::vector<Point<dim> > support_points(bem.dh.n_dofs());
      DoFTools::map_dofs_to_support_points<dim-1, dim>( *bem.mapping, bem.dh, support_points);

      for(types::global_dof_index i = 0; i<bem.dh.n_dofs(); ++i)
        {
          phi_ex[i]=potential.value(support_points[i]);
        }
      std::string filename_ida = "dae_step_output_"+Utilities::to_string(t)+".vtu";//+Utilities::int_to_string(step_number)+".vtu";

      DataOut<dim-1, DoFHandler<dim-1, dim> > dataout_ida;

      dataout_ida.attach_dof_handler(bem.dh);



      dataout_ida.add_data_vector(localized_phi, "phi", DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data);
      dataout_ida.add_data_vector(phi_ex, "phi_ex", DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data);
      dataout_ida.add_data_vector(localized_dphi_dn, "dphi_dn", DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data);
      dataout_ida.add_data_vector(localized_phi_dot, "phi_dot", DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data);
      dataout_ida.add_data_vector(localized_dphi_dn_dot, "dphi_dn_dot", DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data);
      dataout_ida.build_patches(*bem.mapping,
                                   bem.mapping_degree,
                                   DataOut<dim-1, DoFHandler<dim-1, dim> >::curved_inner_cells);

      std::ofstream file_ida(filename_ida.c_str());

      dataout_ida.write_vtu(file_ida);
    }

  return;
}



template <int dim>
bool
DAEBEM<dim>::solver_should_restart(const double t,
                    TrilinosWrappers::MPI::BlockVector &sol,
                    TrilinosWrappers::MPI::BlockVector &sol_dot)
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
DAEBEM<dim>::solve_jacobian_system(const TrilinosWrappers::MPI::BlockVector &src,
                                         TrilinosWrappers::MPI::BlockVector &dst) const
{
  pcout<<"solving Jacobian"<<std::endl;
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
  get_boundary_conditions(t);

  TrilinosWrappers::MPI::Vector tmp_rhs_bem(this_cpu_set, mpi_communicator);
  for(auto i : neumann_set_bem)
  {
    tmp_rhs_bem[i]=solution.block(0)[i];
  }
  for(auto i : dirichlet_set_bem)
  {
    tmp_rhs_bem[i]=solution.block(1)[i];
  }
  // tmp_rhs_bem.print(std::cout);
  // solution_dot.block(0).print(std::cout);
  // dphi_dn.print(std::cout);
  // tmp_rhs_bem.print(std::cout);
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

  pcout<<t<<" : "<<dst.block(0).l2_norm()<<" : "<<dst.block(1).l2_norm()<<std::endl;

  return 0;

}

template <int dim>
TrilinosWrappers::MPI::BlockVector & DAEBEM<dim>::get_lumped_mass_matrix() const
{
  static TrilinosWrappers::MPI::BlockVector diag;
  diag.reinit(solution);

  diag.block(0)=1.;
  diag.block(1)=1.;

  return diag;

}


template <int dim>
TrilinosWrappers::MPI::BlockVector & DAEBEM<dim>::differential_components() const
{
  static TrilinosWrappers::MPI::BlockVector diff_comps;
  diff_comps.reinit(solution);
  for(auto i : differential_set)
  {
    diff_comps.block(0)[i]=1.;
  }
  for(auto i : differential_set)
  {
    diff_comps.block(1)[i]=1.;
  }
  diff_comps.compress(VectorOperation::insert);

  return diff_comps;
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

  add_parameter(prm, &time_stepper, "Time stepper solver","ida", Patterns::Selection("ida|imex"));

  add_parameter(prm, &output_file_name, "Output file name", "result", Patterns::Anything());


}

template <int dim>
void DAEBEM<dim>::parse_parameters(ParameterHandler &prm)
{

  ParameterAcceptor::parse_parameters(prm);
  // output_file_name = prm.get("Output file name");


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
  potential_dot.set_time(0);
  potential_gradient_dot.set_time(0);

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
    this_cpu_set_complete[i] = bem.this_cpu_set;

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
  compute_dae_cache();
  set_dae_initial_conditions(solution, solution_dot);
  TrilinosWrappers::MPI::BlockVector try_residual(solution);
  residual(0., solution, solution_dot, try_residual);
  for(auto i : this_cpu_set)
    pcout<<i<<" : "<<try_residual.block(0)[i]<<" : "<<try_residual.block(1)[i]<<std::endl;
  lambdas.set_functions_to_default();
  if(time_stepper == "ida")
  {
    pcout<<"solving using ida"<<std::endl;
    ida.residual = lambdas.residual;
    ida.setup_jacobian = lambdas.setup_jacobian;
    ida.solver_should_restart = lambdas.solver_should_restart;
    ida.solve_jacobian_system = lambdas.solve_jacobian_system;
    ida.output_step = lambdas.output_step;
    ida.create_new_vector = lambdas.create_new_vector;
    ida.differential_components = lambdas.differential_components;
    // TrilinosWrappers::MPI::BlockVector variation_1(this_cpu_set_double);
    // TrilinosWrappers::MPI::BlockVector variation_2(this_cpu_set_double);
    // TrilinosWrappers::MPI::BlockVector variation_3(this_cpu_set_double);
    // TrilinosWrappers::MPI::BlockVector variation_4(this_cpu_set_double);
    // for(auto i : this_cpu_set)
    // {
    //   double f=(double)rand()/RAND_MAX*2e-8 -1e-8;
    //   variation_1.block(0)[i]=solution.block(0)[i] + f;
    //   variation_2.block(0)[i]=solution_dot.block(0)[i] + f;
    //   variation_3.block(0)[i]=f;
    //   // pcout<<f<<" ";
    //   f=(double)rand()/RAND_MAX*2e-8 -1e-8;
    //   // pcout<<f<<" ";
    //   variation_1.block(1)[i]=solution.block(1)[i] + f;
    //   variation_2.block(1)[i]=solution_dot.block(1)[i] + f;
    //   variation_3.block(1)[i]=f;
    //   f=(double)rand()/RAND_MAX*2e-8 -1e-8;
    //   // pcout<<f<<" ";
    //   // variation_2.block(0)[i]=solution_dot.block(0)[i] + f;
    //   // variation_3.block(2)[i]=solution_dot.block(0)[i] + f;
    //   // f=(double)rand()/RAND_MAX*2e-8 -1e-8;
    //   // // pcout<<f<<std::endl;
    //   // variation_2.block(1)[i]=solution_dot.block(1)[i] + f;
    //   // variation_3.block(3)[i]=solution_dot.block(1)[i] + f;
    // }
    // TrilinosWrappers::MPI::BlockVector new_residual(try_residual);
    // TrilinosWrappers::MPI::BlockVector new_residual_Jac(try_residual);
    // ida.residual(0.,variation_1, variation_2,new_residual);
    // ida.setup_jacobian(0., solution, solution_dot, 1.);
    //
    // jacobian_vmult(variation_3, new_residual_Jac);
    // for(auto i : this_cpu_set)
    // {
    //   pcout<<" delta = "<<variation_3.block(0)[i]<<" : "<<variation_3.block(1)[i]<<std::endl;
    //   pcout<<" J delta = "<<new_residual_Jac.block(0)[i]<<" : "<<new_residual_Jac.block(1)[i]<<std::endl;
    //   pcout<<" f(x+delta) - f(x) = "<<-try_residual.block(0)[i]+new_residual.block(0)[i]<<" : "<<-try_residual.block(1)[i]+new_residual.block(1)[i]<<std::endl;
    //   pcout<<" f(x+delta) - f(x) - J delta = "<<std::abs(-try_residual.block(0)[i]+new_residual.block(0)[i]-new_residual_Jac.block(0)[i])<<" : "<<std::abs(-try_residual.block(1)[i]+new_residual.block(1)[i]-new_residual_Jac.block(1)[i])<<std::endl;
    // }
    // ida.solve_jacobian_system(new_residual_Jac, variation_4);
    // for(auto i : this_cpu_set)
    //   pcout<<"J-1 try : "<<variation_3.block(0)[i]<<" : "<<variation_3.block(0)[i]<<" --- "<<variation_4.block(0)[i]<<" : "<<variation_4.block(0)[i]<<std::endl;
    //
      ida.solve_dae(solution, solution_dot);
  }
  else if(time_stepper == "imex")
  {
    pcout<<"solving using imex"<<std::endl;
    imex.residual = lambdas.residual;
    imex.setup_jacobian = lambdas.setup_jacobian;
    imex.solver_should_restart = lambdas.solver_should_restart;
    imex.solve_jacobian_system = lambdas.solve_jacobian_system;
    imex.output_step = lambdas.output_step;
    imex.create_new_vector = lambdas.create_new_vector;
    imex.get_lumped_mass_matrix = lambdas.get_lumped_mass_matrix;
    // imex.differential_components = lambdas.differential_components;
    imex.solve_dae(solution, solution_dot);
  }

  dphi_dn=solution.block(0);
  phi=solution.block(1);
  dphi_dn_dot=solution.block(0);
  phi_dot=solution_dot.block(1);
  // if (time_stepper == "ida")
  //   ida.start_ode(solution, solution_dot, max_time_iterations);
  // else if (time_stepper == "euler")
  //   euler.start_ode(solution, solution_dot);

  // get_boundary_conditions();
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
  neumann_set.clear();
  dirichlet_set.clear();
  neumann_set_bem.clear();
  dirichlet_set_bem.clear();
  dirichlet_dot_set.clear();
  neumann_dot_set.clear();
  algebraic_set.clear();
  differential_set.clear();

  neumann_set.set_size(this_cpu_set.size());
  dirichlet_set.set_size(this_cpu_set.size());
  neumann_set_bem.set_size(this_cpu_set.size());
  dirichlet_set_bem.set_size(this_cpu_set.size());
  dirichlet_dot_set.set_size(this_cpu_set.size());
  neumann_dot_set.set_size(this_cpu_set.size());
  algebraic_set.set_size(this_cpu_set.size());
  differential_set.set_size(this_cpu_set.size());

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
        if(bem.this_cpu_set.is_element(local_dof_indices[j]))
        {
          algebraic_set.add_index(local_dof_indices[j]);
          neumann_set.add_index(local_dof_indices[j]);
          neumann_set_bem.add_index(local_dof_indices[j]);
        }
    }
    else if(cell->material_id() == 1)//Dirichlet node
    {
      for (unsigned int j=0; j<bem.fe->dofs_per_cell; ++j)
        if(bem.this_cpu_set.is_element(local_dof_indices[j]))
        {
          algebraic_set.add_index(local_dof_indices[j]);
          dirichlet_set.add_index(local_dof_indices[j]);
          dirichlet_set_bem.add_index(local_dof_indices[j]);
        }

    }
    else if(cell->material_id() == 2)//Neumann differential node
    {
      for (unsigned int j=0; j<bem.fe->dofs_per_cell; ++j)
        if(bem.this_cpu_set.is_element(local_dof_indices[j]))
        {
          differential_set.add_index(local_dof_indices[j]);
          neumann_set_bem.add_index(local_dof_indices[j]);
          neumann_dot_set.add_index(local_dof_indices[j]);
        }

    }
    else if(cell->material_id() == 3)//Dirichlet differential node
    {
      for (unsigned int j=0; j<bem.fe->dofs_per_cell; ++j)
        if(bem.this_cpu_set.is_element(local_dof_indices[j]))
        {
          differential_set.add_index(local_dof_indices[j]);
          dirichlet_set_bem.add_index(local_dof_indices[j]);
          dirichlet_dot_set.add_index(local_dof_indices[j]);
        }

    }

  }

  neumann_set.compress();
  dirichlet_set.compress();
  neumann_set_bem.compress();
  dirichlet_set_bem.compress();
  dirichlet_dot_set.compress();
  neumann_dot_set.compress();
  algebraic_set.compress();
  differential_set.compress();

  neumann_set.print(std::cout);
  neumann_dot_set.print(std::cout);
  neumann_set_bem.print(std::cout);


  dirichlet_set.print(std::cout);
  dirichlet_dot_set.print(std::cout);
  dirichlet_set_bem.print(std::cout);

}

template <int dim>
void DAEBEM<dim>::get_boundary_conditions(double t)
{
  Teuchos::TimeMonitor LocalTimer(*PrepareTimeDAE);
  // bem.compute_normals();
  const types::global_dof_index n_dofs =  bem.dh.n_dofs();

  potential.set_time(t);
  potential_gradient.set_time(t);
  potential_dot.set_time(t);
  potential_gradient_dot.set_time(t);

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

  for(auto i : this_cpu_set)
  {
    phi[i] = potential.value(support_points[i]);
  }
  for(auto i : this_cpu_set)
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
  for(auto i : this_cpu_set)
  {
    phi_dot[i] = potential_dot.value(support_points[i], 0);
  }
  for(auto i : this_cpu_set)
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
void DAEBEM<dim>::compute_errors(double t)
{
  Teuchos::TimeMonitor LocalTimer(*ErrorsTimerDAE);
  potential.set_time(t);
  potential_gradient.set_time(t);
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
