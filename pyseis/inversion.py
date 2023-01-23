"""
This module provides the Fwi class for running Full Waveform Inversion (FWI)

Can be used with wave equation solvers provided in the pyseis python package
which currently include the acoustic,constant density, isotrpoic wave equation
and the elastic, isotropic wave equation, both two and three dimensions. In
other words, the Fwi class can be used to solve acoustic or elastic FWI in 2D
and 3D. This module also includes several sub-modules such as pyProblem,
pyStepper, pyStopper, pyOperator, pyNonLinearSolver, and genericIO which are
taken from Ettore Biondi's pysolver module. The Fwi class allows to setup and
run the inversion problem with various solvers and steppers.
"""
from datetime import datetime
import pathlib
import os
import sys_util

import pyProblem as Prblm
import pyStepper as Stepper
import pyStopper as Stopper
import pyOperator as Operator
import pyNonLinearSolver
import genericIO

SOLVERS = {
    'nlcg': pyNonLinearSolver.NLCGsolver,
    'lbfgs': pyNonLinearSolver.LBFGSsolver,
    'lbfgsb': pyNonLinearSolver.LBFGSBsolver
}
STEPPERS = {
    'parabolic': Stepper.ParabolicStep,
    'linear': Stepper.ParabolicStep,
    'parabolicNew': Stepper.ParabolicStepConst
}


class Fwi():

  def __init__(self,
               wave_eq_solver,
               obs_data,
               starting_model,
               num_iter,
               model_bounds=None,
               solver_type='lbfgs',
               stepper_type='parabolic',
               max_steps=None,
               work_dir='wrk',
               prefix=None,
               iterations_per_save=10,
               iterations_in_buffer=3):
    """
    Initialize the FWI (Full Waveform Inversion) class. 
    It is used to set up all the attributes of the FWI object
    that are required to run the inversion.
    
    Parameters:
      wave_eq_solver (class): the wave equation solver
      obs_data (numpy.ndarray): observed data
      starting_model (numpy.ndarray): starting model
      num_iter (int): number of iterations
      model_bounds (List[numpy.ndarray], optional): Bounds of the model, if any.
        Defaults to None.
      solver_type (str, optional): solver to use, options are 'nlcg', 'lbfgs',
        or 'lbfgsb'. Defaults to 'lbfgs'. 
      stepper_type (str, optional): stepper to use options are 'parabolic', 
        'linear', 'parabolicNew'. Defaults to 'parabolic'.
      max_steps (int, optional): maximum number of steps to take. Defaults to None.
      work_dir (str, optional): working directory where history of the inversion is saved.
        Defaults to 'wrk'.
      prefix (str, optional): prefix for saving history in the working directory. Defaults to None.
      iterations_per_save (int, optional): Number of iterations to save.
        Defaults to 10.
      iterations_in_buffer (int, optional): Number of iterations to store in buffer.
        Defaults to 3.
    """
    # check that the observed data matches the data space of the wave_eq_solver
    if wave_eq_solver._get_data().shape != obs_data.shape:
      raise RuntimeError(
          'the data space of the provided wave_eq_solver must match the shape of the provided obs_data'
      )

    # check that the initial model the model space of the wave_eq_solver
    if wave_eq_solver._get_model().shape != starting_model.shape:
      raise RuntimeError(
          'the model space of the provided wave_eq_solver must match the shape of the provided starting_model'
      )

    self.wave_eq_solver = wave_eq_solver

    # create the fwi operator which includes the nonlinear and linearized wave eq
    # self.fwi_op = self._make_fwi_op(self.wave_eq_solver)

    # create inversion problem
    self.wave_eq_solver._set_data(obs_data)
    self.wave_eq_solver._set_model(starting_model)
    self.problem = self._make_problem(self.wave_eq_solver,
                                      self.wave_eq_solver._operator,
                                      model_bounds=model_bounds)

    # create solver
    self.solver = self._make_solver(num_iter,
                                    solver_type=solver_type,
                                    stepper_type=stepper_type,
                                    max_steps=max_steps,
                                    work_dir=work_dir,
                                    prefix=prefix,
                                    iterations_per_save=iterations_per_save,
                                    iterations_in_buffer=iterations_in_buffer)

  def run(self, verbose=1):
    """
    Run the inversion problem with the given attributes.
    
    Parameters:
      verbose (int, optional): level of verbosity. Defaults to 1.
    """
    self.solver.run(self.problem, verbose=verbose)
    self.history = load_history(self.work_dir, self.prefix,
                                self.wave_eq_solver._truncate_model)

    return self.history

  def _make_fwi_op(self, wave_eq_solver):
    """
    Helper function to create the FWI operator which includes
    the nonlinear and linearized wave equation.
    
    Parameters:
        wave_eq_solver (class): the wave equation solver
    """
    if wave_eq_solver._jac_wave_op is None:
      wave_eq_solver._setup_jac_wave_op()

    return Operator.NonLinearOperator(
        wave_eq_solver._nl_wave_op, wave_eq_solver._jac_wave_op,
        wave_eq_solver._jac_wave_op.set_background)

  def _make_problem(self, wave_eq_solver, operator, model_bounds=None):
    """
    Helper function to create inversion problem

    Parameters:
      wave_eq_solver (class): the wave equation solver
      operator (class): FWI operator
      model_bounds (List[numpy.ndarray], optional): Bounds of the model, if any.
        Defaults to None.
    """
    if model_bounds is None:
      min_bound = None
      max_bound = None
    else:
      min_bound = self._make_bounds(wave_eq_solver, model_bounds[0])
      max_bound = self._make_bounds(wave_eq_solver, model_bounds[1])

    return Prblm.ProblemL2NonLinear(wave_eq_solver.model_sep,
                                    wave_eq_solver.data_sep,
                                    operator,
                                    minBound=min_bound,
                                    maxBound=max_bound)

  def _make_bounds(self, wave_eq_solver, bound):
    """
    Helper function to create bounds for the inversion problem.
    
    Parameters:
      wave_eq_solver (class): The wave equation solver
      bounds (numpy.ndarray): The bounds on the model to be used.
    
    Returns:
      bounds (numpy.ndarray): The bounds that are set.
    """
    bound_sep = wave_eq_solver.model_sep.clone()
    bound_arr = bound_sep.getNdArray()
    bound_arr[bound_arr != 0.0] == bound

    return bound_sep

  def _make_solver(self,
                   num_iter,
                   solver_type='lbfgs',
                   stepper_type='parabolic',
                   max_steps=None,
                   work_dir='./.wrk',
                   prefix=None,
                   iterations_per_save=10,
                   iterations_in_buffer=3,
                   overwrite=True):
    """
    Helper function to create a solver for the inversion problem.

    Parameters:
      num_iter (int): The number of iterations to run the solver.
      solver_type (str, optional): The type of solver to use. Defaults to 
        'lbfgs'.
      stepper_type (str, optional): The type of stepper to use. Defaults to
        'parabolic'.
      max_steps (int, optional): Maximum number of steps the solver can take.
        Defaults to None.
      work_dir (str, optional): The working directory where the history of the
        inversion is saved. 
        Defaults to 'wrk'.
      prefix (str, optional): A prefix for saving history in the working
        directory. Defaults to None.
      iterations_per_save (int, optional): The number of iterations to save.
        Defaults to 10.
      iterations_in_buffer (int, optional): The number of iterations to store
        in buffer. Defaults to 3.
    """
    if solver_type not in SOLVERS.keys():
      raise RuntimeError(
          f'{solver_type} not a valid solver type. Options are: { SOLVERS.keys()}'
      )
    # set working directory
    self.work_dir = pathlib.Path(work_dir)
    self.work_dir.mkdir(exist_ok=overwrite)

    # set prefix
    self.prefix = prefix
    if prefix is None:
      self.prefix = datetime.now().strftime('%Y-%m-%dT%H%M%SZ')

    # make logger
    self.logger_filename = self.work_dir / (self.prefix + '.log')
    self.logger = sys_util.logger(str(self.logger_filename))

    # make stopper
    self.stopper = Stopper.BasicStopper(niter=num_iter)

    # make solver
    solver = SOLVERS[solver_type](self.stopper, self.logger)

    # set solver parameters
    if max_steps is not None:
      solver.max_steps = max_steps
    if stepper_type not in STEPPERS.keys():
      raise RuntimeError(
          f'{stepper_type} not a valid stepper type. Options are: { STEPPERS.keys()}'
      )
    solver.stepper = STEPPERS[stepper_type]()
    if stepper_type == 'linear':
      solver.stepper.eval_parab = False

    # set history defaults
    solver.setDefaults(save_obj=1,
                       save_res=1,
                       save_grad=1,
                       save_model=1,
                       prefix=str(self.work_dir / self.prefix),
                       iter_buffer_size=iterations_in_buffer,
                       iter_sampling=iterations_per_save,
                       flush_memory=1)

    return solver


def load_history(work_dir, prefix, truncate_padding_func=None):
  """
  Load the history of the inversion from the working directory.

  Parameters:
    work_dir (str): The working directory where the history is saved.
    prefix (str): The prefix for the history files.
    truncate_model (callable): A callable that truncates the model if needed.
  
  Returns:
    history (List[dict]): The history of the inversion.
  """
  model_space_keys = ['inv_mod', 'gradient', 'model']
  other_keys = ['residual', 'obj']

  history = {}

  # for model space history files, truncate padding by default
  for key in model_space_keys:
    arr = None
    filename = work_dir / (prefix + f'_{key}.H')
    if filename.is_file():
      arr = _load_sep_to_np(filename)
      if truncate_padding_func:
        arr = truncate_padding_func(arr)
    history[key] = arr

  # laod other history files
  for key in other_keys:
    arr = None
    filename = work_dir / (prefix + f'_{key}.H')
    if filename.is_file():
      arr = _load_sep_to_np(filename)
    history[key] = arr

  return history


def _load_sep_to_np(sep_fn):
  return genericIO.defaultIO.getVector(str(sep_fn)).getNdArray()