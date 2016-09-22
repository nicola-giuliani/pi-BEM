#ifndef __dae_lambdas_h_
#define __dae_lambdas_h_

// #include "lac/lac_type.h"
#include <deal2lkit/utilities.h>

using namespace deal2lkit;
// forward declaration
template <int dim> class DAEBEM;

/**
 * Lambdas class. A DAEBEM object has a Lambdas object and the
 * std::functions of the provided stepper (e.g., ida, imex)
 * are set to be equal to those here implemented.
 *
 * By default, the std::functions of this class call the
 * namesake functions of the DAEBEM object, which is specified
 * either when the Lambdas object is constructed or by calling the
 * Lambdas::initialize_dae() function.
 *
 * The aim of this class is to increase the flexibility of DAEBEM.
 * DAEBEM offers flexibility by itself thanks to the signals
 * to whom the user can connect in order to perform problem-specific
 * tasks. Whether the behavior of a function should be completely different,
 * the user can ovveride the functions declared here (without the need
 * of modifying DAEBEM' source code).
 */

template <int dim>
class Lambdas
{
public:

  /**
   * Default constructor. Initialize the Lambdas object without
   * a reference to a particular DAEBEM object. You will later have
   * to call initialize() to provide this reference to the DAEBEM
   * object.
   */
  Lambdas ();

  /**
   * Create a Lambdas object that is already initialized for
   * a particular DAEBEM.
   */
  Lambdas (DAEBEM<dim> &dae_obj);

  /**
   * Destructor. Does nothing.
   */
  ~Lambdas ();

  /**
   * Initialize this class for a given dae.
   *
   * @param dae_object A reference to the main dae object.
   */
  void initialize_dae (DAEBEM<dim> &dae_obj);

  /**
   * Set the default behavior of the functions of this class,
   * which is call the namesake functions of the dae_bject.
   */
  void set_functions_to_default();

  /**
   * Return a shared_ptr<VECTOR_TYPE>. A shared_ptr is needed in order to
   * keep the pointed vector alive, without the need to use a static variable.
   */
  std::function<shared_ptr<TrilinosWrappers::MPI::BlockVector>()> create_new_vector;

  /**
   * Compute residual.
   */
  std::function<int(const double t,
                    const TrilinosWrappers::MPI::BlockVector &y,
                    const TrilinosWrappers::MPI::BlockVector &y_dot,
                    TrilinosWrappers::MPI::BlockVector &res)> residual;

  /**
   * Compute Jacobian.
   */
  std::function<int(const double t,
                    const TrilinosWrappers::MPI::BlockVector &y,
                    const TrilinosWrappers::MPI::BlockVector &y_dot,
                    const double alpha)> setup_jacobian;

  /** Solve linear system. */
  std::function<int(const TrilinosWrappers::MPI::BlockVector &rhs, TrilinosWrappers::MPI::BlockVector &dst)> solve_jacobian_system;

  /**
   * Store solutions to file.
   */
  std::function<void (const double t,
                      const TrilinosWrappers::MPI::BlockVector &sol,
                      const TrilinosWrappers::MPI::BlockVector &sol_dot,
                      const unsigned int step_number)> output_step;

  /**
   * Evaluate wether the mesh should be refined or not. If so,
   * it refines and interpolate the solutions from the old to the
   * new mesh.
   */
  std::function<bool (const double t,
                      TrilinosWrappers::MPI::BlockVector &sol,
                      TrilinosWrappers::MPI::BlockVector &sol_dot)> solver_should_restart;

  /**
   * Return a vector whose component are 1 if the corresponding
   * dof is differential, 0 if algebraic. This function is needed
   * by the IDAInterface stepper.
   */
  std::function<TrilinosWrappers::MPI::BlockVector&()> differential_components;

  /**
   * Return a vector whose components are the weights used by
   * IDA to compute the vector norm. By default this function is not
   * implemented.
   */
  std::function<TrilinosWrappers::MPI::BlockVector&()> get_local_tolerances;

  /**
   * Return a vector which is a lumped mass matrix. This function
   * is used by Kinsol (through imex) for setting the weights used
   * for computing the norm a vector.
   */
  // std::function<TrilinosWrappers::MPI::BlockVector&()> get_lumped_mass_matrix;

  /**
   * Compute the matrix-vector product Jacobian times @p src,
   * and the result is put in @p dst. It is required by Kinsol.
   */
  std::function<int(const TrilinosWrappers::MPI::BlockVector &src,
                    TrilinosWrappers::MPI::BlockVector &dst)> jacobian_vmult;

  /**
   * Return the norm of @p vector, which is used by IMEXStepper.
   */
  std::function<double(const TrilinosWrappers::MPI::BlockVector &vector)> vector_norm;

  /**
   * A pointer to the dae object to which we want to get
   * access.
   */
  DAEBEM<dim> *dae;
};

#endif
