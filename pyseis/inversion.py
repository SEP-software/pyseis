"""
This module provides wave equation based inversion classes.

Can be used with wave equation solvers provided in the pyseis python package
which currently include the acoustic, constant density, isotropic wave equation
and the elastic, isotropic wave equation, both two and three dimensions. In
other words, the classes can be used to solve acoustic or elastic waveform
inversions in 2D and 3D. This module also includes several sub-modules such as
pyProblem, pyStepper, pyStopper, pyOperator, pyNonLinearSolver, and genericIO
which are taken from Ettore Biondi's python-solver module. Inversions can be run
with various inversion solvers and steppers.
"""
import pathlib
import os
import sys_util
import numpy as np

from datetime import datetime
from typing import Callable, List, Tuple

import pyProblem as Prblm
import pyStepper as Stepper
import pyStopper as Stopper
import pyOperator as Operator
import pyNonLinearSolver
import pyLinearSolver
import genericIO
import SepVector

from pyseis.wave_equations import wave_equation

SOLVERS = {
    'nlcg': pyNonLinearSolver.NLCGsolver,
    'lbfgs': pyNonLinearSolver.LBFGSsolver,
    'lbfgsb': pyNonLinearSolver.LBFGSBsolver,
    'cg': pyLinearSolver.LCGsolver,
    'lsqr': pyLinearSolver.LSQRsolver
}
STEPPERS = {
    'parabolic': Stepper.ParabolicStep,
    'linear': Stepper.ParabolicStep,
    'parabolicNew': Stepper.ParabolicStepConst
}


class WaveEquationInversion():
  """
  A purely absract class that defines the interface for wave equation based
  inversions. This class is not meant to be instantiated directly, but rather
  used as a base class for other classes that implement the abstract methods.
  """

  def __init__(self,
               wave_eq_solver: wave_equation.WaveEquation,
               seismic_data: np.ndarray,
               earth_model: np.ndarray,
               num_iter: int,
               solver_type: str,
               stepper_type: str = None,
               max_steps: int = None,
               work_dir: str = 'wrk',
               prefix: str = None,
               iterations_per_save: int = 10,
               iterations_in_buffer: int = 3) -> None:
    """Abstract constructor for wave equation based inversions.

    Parameters:
      wave_eq_solver (wave_equation.WaveEquatio): the wave equation solver. See
        pyseis/wave_equations.
      seismic_data (numpy.ndarray): seismic data to be used in inversion.
      earth_model (numpy.ndarray): earth model to be in wave prop
      num_iter (int): number of iterations
      solver_type (str, optional): solver to use, options are 'nlcg', 'lbfgs',
        or 'lbfgsb'. Defaults to 'nlcg'. 
      stepper_type (str, optional): stepper to use options are 'parabolic',
        'linear', 'parabolicNew'. Defaults to 'parabolic'.
      max_steps (int, optional): maximum number of steps to take. Defaults to
        None.
      work_dir (str, optional): working directory where history of the inversion
        is saved.
        Defaults to 'wrk'.
      prefix (str, optional): prefix for saving history in the working directory.
        Defaults to None.
      iterations_per_save (int, optional): Number of iterations to save.
        Defaults to 10.
      iterations_in_buffer (int, optional): Number of iterations to store in buffer.
        Defaults to 3.
    """
    # check that the observed data matches the data space of the wave_eq_solver
    if wave_eq_solver._get_data().shape != seismic_data.shape:
      raise RuntimeError(
          'the data space of the provided wave_eq_solver must match the shape of the provided obs_data'
      )

    # check that the initial model the model space of the wave_eq_solver
    if wave_eq_solver._get_model().shape != earth_model.shape:
      raise RuntimeError(
          'the model space of the provided wave_eq_solver must match the shape of the provided earth_model'
      )

    # check that that the starting model will not cause dispersion
    if not wave_eq_solver.check_dispersion(
        wave_eq_solver._get_velocities(earth_model),
        wave_eq_solver.model_sampling, wave_eq_solver.fd_param['f_max']):
      min_vel = wave_eq_solver.find_min_vel(wave_eq_solver.model_sampling,
                                            wave_eq_solver.fd_param['f_max'])
      raise RuntimeError(
          f"The provided earth model will cause dispersion because the minumum velocity value is too low. Given the current max frequency in the source wavelet, {wave_eq_solver.fd_param['f_max']}Hz, and max spatial sampling, {max(wave_eq_solver.model_sampling)}m, the minimum allowed velocity is {min_vel} m/s"
      )

    self.wave_eq_solver = wave_eq_solver

    # create solver
    self.solver = self._make_solver(num_iter,
                                    solver_type=solver_type,
                                    stepper_type=stepper_type,
                                    max_steps=max_steps,
                                    work_dir=work_dir,
                                    prefix=prefix,
                                    iterations_per_save=iterations_per_save,
                                    iterations_in_buffer=iterations_in_buffer)

  def run(self, verbose: int = 1) -> dict:
    """
    Run the inversion problem with the given attributes.
    
    Parameters:
      verbose (int, optional): level of verbosity. Defaults to 1.
    """
    self.solver.run(self.problem, verbose=verbose)
    self.history = load_history(self.work_dir, self.prefix,
                                self.wave_eq_solver._truncate_model)

    return self.history

  def _make_grad_mask(self, gradient_mask, wave_eq_solver):
     # pad gradient mask
    gradient_mask = wave_eq_solver._pad_model(gradient_mask,
                                              wave_eq_solver.model_padding,
                                              wave_eq_solver._FAT)
    gradient_mask_sep = wave_eq_solver.model_sep.clone()
    gradient_mask_sep.getNdArray()[:] = gradient_mask
    
    return gradient_mask_sep
    
  def _make_nl_problem(
      self,
      model: SepVector.floatVector,
      data: SepVector.floatVector,
      operator: Operator,
      gradient_mask: SepVector.floatVector = None,
      model_bounds: List[float] = None) -> Prblm.ProblemL2NonLinear:
    """Helper function to create inversion problem

    Args:
        model (SepVector.floatVector): model space of inversion problem
        data (SepVector.floatVector): data space of inversion problem
        operator (Operator): nonlinear operator of inversion problem, mapping
          model space to data space.
        model_bounds (List[float], optional): upper and lower bounds of model
          space values. Defaults to None.

    Returns:
        Prblm.ProblemL2NonLinear: nonlinear L2 inversion problem
    """
    if model_bounds is None:
      min_bound = None
      max_bound = None
    else:
      min_bound = self._make_bounds(model, model_bounds[0])
      max_bound = self._make_bounds(model, model_bounds[1])

    return Prblm.ProblemL2NonLinear(model,
                                    data,
                                    operator,
                                    minBound=min_bound,
                                    maxBound=max_bound,
                                    grad_mask=gradient_mask)

  def _make_lin_problem(self, model: SepVector.floatVector, data: SepVector.floatVector, operator: Operator,
      gradient_mask: SepVector.floatVector = None,) -> Prblm.ProblemL2Linear:
    """Helper function to create inversion problem

    Args:
        model (SepVector): model space of inversion problem
        data (SepVector): data space of inversion problem
        operator (Operator): linear operator of inversion problem, mapping model
          space to data space.

    Returns:
        Prblm.ProblemL2Linear: linear L2 inversion problem
    """

    return Prblm.ProblemL2Linear(model, data, operator,grad_mask=gradient_mask)

  def _make_bounds(self, model_sep: SepVector.floatVector,
                   bound: float) -> SepVector.floatVector:
    """
    Helper function to create bounds for the inversion problem.
    
    Parameters:
      model_sep (SepVector.floatVector): Model space to be used.
      bounds (numpy.ndarray): The bound on the model to be used.
    
    Returns:
      bounds (numpy.ndarray): A numpy array of the same shape as the model space
    """
    bound_sep = model_sep.clone()
    bound_arr = bound_sep.getNdArray()
    bound_arr[bound_arr != 0.0] == bound

    return bound_sep

  def _make_solver(self,
                   num_iter: int,
                   solver_type: str = 'lbfgs',
                   stepper_type: str = 'parabolic',
                   max_steps: int = None,
                   work_dir: str = './.wrk',
                   prefix: str = None,
                   iterations_per_save: int = 10,
                   iterations_in_buffer: int = 3,
                   overwrite: bool = True):
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
      overwrite (bool, optional): whether to overwrite working directory.
        Defaults to True.
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
    solver = SOLVERS[solver_type](self.stopper, logger=self.logger)

    # set maxsteps
    if max_steps is not None:
      solver.max_steps = max_steps

    # set stepper, if solver_type is linear, this doesnt change anything
    if stepper_type is not None:
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


class Lsrtm(WaveEquationInversion):

  def __init__(self,
               wave_eq_solver: wave_equation.WaveEquation,
               linear_data: np.ndarray,
               migration_model: np.ndarray,
               num_iter: int,
               starting_model: np.ndarray = None,
               solver_type: str = 'cg',
               work_dir: str = 'wrk',
               prefix: str = None,
               iterations_per_save: int = 10,
               iterations_in_buffer: int = 3,
               gradient_mask: np.ndarray = None) -> None:
    """Initialize the Least Squares Reverse Time Migration (LSRTM) class.
    
    Used to set up all the attributes of the Lsrtm object that are required to
    run an inversion.

    Parameters:
        wave_eq_solver (wave_equation.WaveEquation): the wave equation solver
        linear_data (np.ndarray): seimic data to be inverted
        migration_model (np.ndarray): the background velocity model around which
          the wave equation is linearized
        num_iter (int): number of inversion iterations
        starting_model (np.ndarray, optional): starting image. Defaults to None.
        solver_type (str, optional): type of inversion solver. Defaults to 'cg'.
        work_dir (str, optional): working directory where history of the
          inversion is saved. Defaults to './wrk'.
        prefix (str, optional): prefix for saving history in the working
          directory. Defaults to None.
        iterations_per_save (int, optional): Number of iterations to save.
          Defaults to 10.
        iterations_in_buffer (int, optional): Number of iterations to store in
          memory buffer. Defaults to 3.
        gradient_mask: (np.ndarry, optional): Multiplied with gradient of FWI
          before computing step size. Must be the shape of wave_eq_solver model
          space (which is also the same shape as the starting_model).
    """
    # call parent constructor
    super().__init__(wave_eq_solver=wave_eq_solver,
                     seismic_data=linear_data,
                     earth_model=migration_model,
                     num_iter=num_iter,
                     solver_type=solver_type,
                     work_dir=work_dir,
                     prefix=prefix,
                     iterations_per_save=iterations_per_save,
                     iterations_in_buffer=iterations_in_buffer)

    # make fwi operator which contains the jacobian operator
    if self.wave_eq_solver._fwi_operator is None:
      self.wave_eq_solver._fwi_operator = self.wave_eq_solver._setup_fwi_op()

    # make gradient mask
    if gradient_mask is not None:
      gradient_mask = self._make_grad_mask(gradient_mask, self.wave_eq_solver)

    # create inversion problem
    self.wave_eq_solver._set_data(linear_data)
    self.wave_eq_solver._set_background_model(migration_model)
    if starting_model is None:
      starting_model = np.zeros_like(migration_model)
    wave_eq_solver._set_lin_model(starting_model)
    self.problem = self._make_lin_problem(self.wave_eq_solver.lin_model_sep,
                                          self.wave_eq_solver.data_sep,
                                          self.wave_eq_solver._jac_operator,
                                          gradient_mask=gradient_mask)


class Fwi(WaveEquationInversion):

  def __init__(self,
               wave_eq_solver: wave_equation.WaveEquation,
               obs_data: np.ndarray,
               starting_model: np.ndarray,
               num_iter: int,
               model_bounds: Tuple[float, float] = None,
               solver_type: str = 'nlcg',
               stepper_type: str = 'parabolic',
               max_steps: int = None,
               work_dir: str = 'wrk',
               prefix: str = None,
               iterations_per_save: int = 10,
               iterations_in_buffer: int = 3,
               gradient_mask: np.ndarray = None) -> None:
    """
    Initialize the FWI (Full Waveform Inversion) class.
    
    Used to set up all the attributes of the FWI object that are required to run
    an inversion.
    
    Parameters:
      wave_eq_solver (wave_equation.WaveEquatio): the wave equation solver. See
        pyseis/wave_equations.
      obs_data (numpy.ndarray): observed data
      starting_model (numpy.ndarray): starting model
      num_iter (int): number of iterations
      model_bounds (Tuple[float, float], optional): [min, max] model bounds of
        the  model, if any. Defaults to None.
      solver_type (str, optional): solver to use, options are 'nlcg', 'lbfgs',
        or 'lbfgsb'. Defaults to 'nlcg'. 
      stepper_type (str, optional): stepper to use options are 'parabolic',
        'linear', 'parabolicNew'. Defaults to 'parabolic'.
      max_steps (int, optional): maximum number of steps to take. Defaults to
        None.
      work_dir (str, optional): working directory where history of the inversion
        is saved. Defaults to 'wrk'.
      prefix (str, optional): prefix for saving history in the working
        directory. Defaults to None.
      iterations_per_save (int, optional): Number of iterations to save.
        Defaults to 10.
      iterations_in_buffer (int, optional): Number of iterations to store in
        memory buffer. Defaults to 3.
      gradient_mask: (np.ndarry, optional): Multiplied with gradient of FWI
        before computing step size. Must be the shape of wave_eq_solver model
        space (which is also the same shape as the starting_model).
    """
    # call parent constructor
    super().__init__(wave_eq_solver=wave_eq_solver,
                     seismic_data=obs_data,
                     earth_model=starting_model,
                     num_iter=num_iter,
                     solver_type=solver_type,
                     stepper_type=stepper_type,
                     max_steps=max_steps,
                     work_dir=work_dir,
                     prefix=prefix,
                     iterations_per_save=iterations_per_save,
                     iterations_in_buffer=iterations_in_buffer)

    # setup fwi operator
    if self.wave_eq_solver._fwi_operator is None:
      self.wave_eq_solver._fwi_operator = self.wave_eq_solver._setup_fwi_op()

    # make gradient mask
    if gradient_mask is not None:
      gradient_mask = self._make_grad_mask(gradient_mask, self.wave_eq_solver)

    # create inversion problem
    self.wave_eq_solver._set_data(obs_data)
    self.wave_eq_solver._set_model(starting_model)
    self.problem = self._make_nl_problem(self.wave_eq_solver.model_sep,
                                         self.wave_eq_solver.data_sep,
                                         self.wave_eq_solver._fwi_operator,
                                         gradient_mask=gradient_mask,
                                         model_bounds=model_bounds)


def load_history(work_dir: str,
                 prefix: str,
                 truncate_padding_func: Callable = None) -> dict:
  """
  Load the history of the inversion from the working directory.

  Parameters:
    work_dir (str): The working directory where the history is saved.
    prefix (str): The prefix for the history files.
    truncate_model (callable): A callable that truncates the model if needed.
  
  Returns:
    history (List[dict]): The history of the inversion.
  """
  # check if truncate_padding_func is callable
  if truncate_padding_func is not None:
    if not callable(truncate_padding_func):
      raise RuntimeError('Provided truncate_padding_func is not callable!')

  model_space_keys = ['inv_mod', 'gradient', 'model']
  other_keys = ['residual', 'obj']

  history = {}

  work_dir = pathlib.Path(work_dir)
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


def _load_sep_to_np(sep_fn: str) -> np.ndarray:
  """Loads a SepVector from disk to numpy array

  Args:
      sep_fn (str): filename of Seplib .H file

  Returns:
      np.ndarray: SepVector in numpy array format
  """
  sep_vec = genericIO.defaultIO.getVector(str(sep_fn))
  return genericIO.defaultIO.getVector(str(sep_fn)).getNdArray()