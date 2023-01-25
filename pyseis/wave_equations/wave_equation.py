"""Defines the abstract WaveEquation class.

A WaveEquation object is able to forward time march a finite difference wave
equation in two or three dimensions and sample the resulting wavefield(s) at
receiver locations. AcousticIsotropic and ElasticIsotropic child classes
implement different wave equations but share common methods that are defined in
WaveEquation.

"""
import numpy as np
import abc
import math
import typing

import genericIO
import pyOperator as Operator
from pyElastic_iso_float_nl_3D import ostream_redirect
from pyseis.wavelets import Wavelet

SEP_PARAM_CYPHER = {
    'ny': 'n_y',
    'dy': 'd_y',
    'nx': 'n_x',
    'dx': 'd_x',
    'nz': 'n_z',
    'dz': 'd_z',
    'yPad': 'y_pad',
    'xPadMinus': 'x_pad_minus',
    'xPadPlus': 'x_pad_plus',
    'zPadMinus': 'z_pad_minus',
    'zPadPlus': 'z_pad_plus',
    'mod_par': 'mod_par',
    'dts': 'd_t',
    'nts': 'n_t',
    'fMax': 'f_max',
    'sub': 'sub',
    'nShot': 'n_src',
    'nExp': 'n_src',
    'iGpu': 'gpus',
    'block_size': '_BLOCK_SIZE',
    'blockSize': '_BLOCK_SIZE',
    'fat': '_FAT',
    'ginsu': 'ginsu',
    'freeSurface': 'free_surface',
    'surfaceCondition': 'free_surface'
}
COURANT_LIMIT = 0.45


class WaveEquation(abc.ABC):

  def __init__(self,
               model,
               model_sampling,
               wavelet,
               d_t,
               src_locations,
               rec_locations,
               gpus,
               model_padding=None,
               model_origins=None,
               subsampling=None,
               free_surface=False):
    if free_surface and not self._FREE_SURFACE_AVAIL:
      raise NotImplementedError(
          'free surface condition has not been implemented yet!')

    self.model_sep = None
    self.data_sep = None
    self.fd_param = {
        '_BLOCK_SIZE': self._BLOCK_SIZE,
        '_FAT': self._FAT,
        'mod_par': 1,
        'free_surface': int(free_surface)
    }
    self._operator = None

    # default model padding
    if not model_padding:
      model_padding = (self._DEFAULT_PADDING,) * len(model_sampling)

    # default origins are at 0.0
    if not model_origins:
      model_origins = (0.0,) * len(model_sampling)

    self._make(model, wavelet, d_t, src_locations, rec_locations, gpus,
               model_padding, model_origins, model_sampling, subsampling,
               free_surface)

  def _make(self,
            model,
            wavelet,
            d_t,
            src_locations,
            rec_locations,
            gpus,
            model_padding,
            model_origins,
            model_sampling,
            subsampling=None,
            free_surface=False):
    # pads model, makes self.model_sep, and updates self.fd_param
    self.model_sep, self.model_padding = self._setup_model(
        model, model_padding, model_sampling, model_origins, free_surface)

    # make and set wavelet
    self.wavelet_nl_sep, self.wavelet_lin_sep = self._setup_wavelet(
        wavelet, d_t)

    # make and set source devices
    self.src_devices = self._setup_src_devices(src_locations,
                                               self.fd_param['n_t'])

    # make and set rec devices
    self.rec_devices = self._setup_rec_devices(rec_locations,
                                               self.fd_param['n_t'])

    # make and set data space
    self.data_sep = self._setup_data(self.fd_param['n_t'], d_t)

    # calculate and find subsampling. pass only the velocity componenets
    velocities = self._get_velocities(model)
    self._setup_subsampling(velocities, d_t, model_sampling, subsampling)

    # set gpus list
    self.fd_param['gpus'] = str(gpus)[1:-1]

    #set ginsu to zero for now
    self.fd_param['ginsu'] = 0

    # make and set sep par
    self.sep_param = self._setup_sep_par(self.fd_param)

    # make operator
    self._operator = self._setup_operators(self.data_sep, self.model_sep,
                                           self.sep_param, self.src_devices,
                                           self.rec_devices,
                                           self.wavelet_nl_sep,
                                           self.wavelet_lin_sep)

    self.data_sep = self._operator.range.clone()

  def forward(self, model):
    self._set_model(model)
    self._operator.nl_op.forward(0, self.model_sep, self.data_sep)
    return np.copy(self.data_sep.getNdArray())

  def jacobian(self, lin_model, background_model=None):
    self._set_lin_model(lin_model)
    if background_model is not None:
      self._set_background_model(background_model)

    self._operator.lin_op.forward(0, self.lin_model_sep, self.data_sep)
    return np.copy(self.data_sep.getNdArray())

  def jacobian_adjoint(self, lin_data, background_model=None):
    self._set_data(lin_data)
    if background_model is not None:
      self._set_background_model(background_model)

    self._operator.lin_op.adjoint(0, self.lin_model_sep, self.data_sep)
    return np.copy(self._truncate_model(self.lin_model_sep.getNdArray()))

  def dot_product_test(self, verb=False, tolerance=0.00001):
    with ostream_redirect():
      return self._operator.lin_op.dotTest(verb, tolerance)

  def _setup_wavelet(self, wavelet, d_t):
    self.fd_param['d_t'] = d_t
    self.fd_param['n_t'] = wavelet.shape[-1]
    self.fd_param['f_max'] = Wavelet.calc_max_freq(wavelet, d_t)
    return self._make_sep_wavelets(wavelet, d_t)

  def _setup_sep_par(self, fd_param):
    sep_param_dict = {}
    for required_sep_param in self.required_sep_params:
      fd_param_key = SEP_PARAM_CYPHER[required_sep_param]
      if fd_param_key not in fd_param:
        raise RuntimeError(f'{fd_param_key} was not set.')
      sep_param_dict[required_sep_param] = fd_param[fd_param_key]

    return self._make_sep_par(sep_param_dict)

  def _make_sep_par(self, dict):
    """Turn a dictionary of kwargs into a genericIO par object needed for wave prop

    Args:
        dict (dictionary): a dictionary of args passed to nonlinearPropShotsGpu
          constructor

    Returns:
        genericIO.io object:
    """
    kwargs_str = {
        key: str(value)[1:-1] if isinstance(value, list) else str(value)
        for key, value in dict.items()
    }

    return genericIO.io(params=kwargs_str)

  def _setup_nl_wave_op(self, data_sep, model_sep, sep_par, src_devices,
                        rec_devices, wavelet_nl_sep):
    _nl_wave_op = _NonlinearWaveCppOp(model_sep, data_sep, sep_par, src_devices,
                                      rec_devices, wavelet_nl_sep,
                                      self._nl_wave_pybind_class)
    self.lin_model_sep = self.model_sep.clone()
    _jac_wave_op = _JacobianWaveCppOp(self.lin_model_sep, self.data_sep,
                                      self.model_sep, self.sep_param,
                                      self.src_devices, self.rec_devices,
                                      self.wavelet_lin_sep,
                                      self._jac_wave_pybind_class)
    return Operator.NonLinearOperator(_nl_wave_op, _jac_wave_op,
                                      _jac_wave_op.set_background)

  def _set_model(self, model):
    # pad model
    model = self._pad_model(model, self.model_padding, self._FAT)

    # set sep model
    self.model_sep.getNdArray()[:] = model

  def _set_lin_model(self, lin_model):
    # pad model
    lin_model = self._pad_model(lin_model, self.model_padding, self._FAT)

    # set sep model
    self.lin_model_sep.getNdArray()[:] = lin_model

  def _set_background_model(self, background_model):
    background_model = self._pad_model(background_model, self.model_padding,
                                       self._FAT)
    background_model_sep = self.model_sep.clone()
    background_model_sep.getNdArray()[:] = background_model
    self._operator.lin_op.set_background(background_model_sep)

  def _set_data(self, data):
    self.data_sep.getNdArray()[:] = data

  def _get_data(self):
    return self.data_sep.getNdArray()

  def _get_model(self, padded=False):
    model = self.model_sep.getNdArray()
    if not padded:
      model = self._truncate_model(model)

    return model

  def _setup_model(self,
                   model,
                   model_padding,
                   model_sampling,
                   model_origins=None,
                   free_surface=False):
    model_shape = self._get_model_shape(model)
    padding, padded_model_shape, padded_model_origins = self._calc_pad_params(
        model_shape, model_padding, model_sampling, self._FAT, model_origins,
        free_surface)
    self._set_padding_params(padding, padded_model_shape, padded_model_origins,
                             model_sampling)
    model = self._pad_model(model, padding, self._FAT)

    # make SepVector model
    model_sep = self._make_sep_vector_model_space(padded_model_shape,
                                                  padded_model_origins,
                                                  model_sampling)
    model_sep.getNdArray()[:] = model

    return model_sep, padding

  def _calc_pad_params(
      self,
      model_shape: tuple,
      model_padding: tuple,
      model_sampling: tuple,
      fat: int,
      model_origins: tuple = None,
      free_surface: bool = False) -> typing.Tuple[tuple, tuple, tuple]:
    """
    Returns the padding, new shape, and new origins of the model based on the given parameters.
   
    Parameters:
        model_shape (tuple): Tuple of integers representing the shape of the model
        model_padding (tuple): Tuple of integers representing the padding of the model in each axis
        model_sampling (tuple): Tuple of floats representing the sampling rate of the model in each axis
        model_origins (tuple, optional): Tuple of floats representing the origin of the model in each axis
        free_surface (bool, optional): Flag indicating whether the model has a free surface or not
    
    Returns:
        Tuple of tuples: Tuple of padding, new shape, and new origins of the model
    """

    # check that dimensions are consistent
    if len(model_shape) != len(model_padding):
      raise ValueError(
          f'Shape of model_padding has {len(model_padding)} dimension but model has {len(model_shape)}'
      )
    if len(model_shape) != len(model_sampling):
      raise ValueError(
          f'Shape of model_sampling has {len(model_sampling)} dimension but model has {len(model_shape)}'
      )
    if model_origins and len(model_shape) != len(model_origins):
      raise ValueError(
          f'Shape of model_origins has {len(model_origins)} dimension but model has {len(model_shape)}'
      )
    elif not model_origins:
      model_origins = (0.0,) * len(model_shape)

    padding = []

    # y axis if it exists
    if len(model_shape) == 3:
      padding.append((model_padding[0], model_padding[0]))

    # x axis
    padding.append(
        self._calc_pad_minus_plus(model_shape[-2], model_padding[-2], fat))

    # z axis
    padding.append(
        self._calc_pad_minus_plus(model_shape[-1],
                                  model_padding[-1],
                                  fat,
                                  free_surface=free_surface))

    # new origins
    new_origins = [
        old_origin - (fat + pad_minus) * sampling for old_origin, (
            pad_minus,
            _), sampling in zip(model_origins, padding, model_sampling)
    ]

    # new shape
    new_shape = [
        fat + pad_minus + shape + pad_plus + fat
        for shape, (pad_minus, pad_plus) in zip(model_shape, padding)
    ]

    return tuple(padding), tuple(new_shape), tuple(new_origins)

  def _calc_pad_minus_plus(
      self,
      axis_size: int,
      axis_padding: int,
      fat: int,
      free_surface: bool = False) -> typing.Tuple[int, int]:
    """
    Returns the pad minus and pad plus values for the given axis size and padding.
   
    Parameters:
        axis_size (int): Size of the axis
        axis_padding (int): Padding of the axis
        free_surface (bool, optional): Flag indicating whether the model has a free surface or not
    
    Returns:
        Tuple of integers: Tuple of pad minus and pad plus values for the given axis
    """
    if free_surface:
      pad_minus = self._get_free_surface_pad_minus()
      axis_total_samples = pad_minus + axis_size + axis_padding
    else:
      axis_total_samples = axis_padding * 2 + axis_size
      pad_minus = axis_padding

    n_gpu_blocks = math.ceil(axis_total_samples / self._BLOCK_SIZE)
    pad_plus = n_gpu_blocks * self._BLOCK_SIZE - axis_size - pad_minus

    return (pad_minus, pad_plus)

  def _set_padding_params(self, padding, model_shape, model_origins,
                          model_sampling):
    if len(model_shape) == 3:
      self.fd_param['n_y'] = model_shape[0]
      self.fd_param['o_y'] = model_origins[0]
      self.fd_param['d_y'] = model_sampling[0]
      self.fd_param['y_pad'] = padding[0][0]

    self.fd_param['n_x'] = model_shape[-2]
    self.fd_param['o_x'] = model_origins[-2]
    self.fd_param['d_x'] = model_sampling[-2]
    self.fd_param['x_pad_minus'] = padding[-2][0]
    self.fd_param['x_pad_plus'] = padding[-2][1]

    self.fd_param['n_z'] = model_shape[-1]
    self.fd_param['o_z'] = model_origins[-1]
    self.fd_param['d_z'] = model_sampling[-1]
    self.fd_param['z_pad_minus'] = padding[-1][0]
    self.fd_param['z_pad_plus'] = padding[-1][1]

  def _setup_subsampling(self,
                         vel_models,
                         d_t,
                         model_sampling,
                         subsampling=None):
    # caclulate subsampling minimum
    min_sub = self._calc_subsampling(vel_models, d_t, model_sampling)
    # if subsampling was already used to init a pybind operator, check that the newly caclulated subsampling is lesser
    if 'sub' in self.fd_param:
      if min_sub > self.fd_param['sub']:
        raise RuntimeError(
            'Newly set model requires greater subsampling than what wave equation operator was initialized with. This is currently not allowed.'
        )
    # user specified subsampling must be greater than calculated subsampling
    if subsampling is not None:
      if subsampling < min_sub:
        raise RuntimeError(
            f"User specified subsampling={subsampling} that does not satisfy Courant condition. subsampling must be >={min_sub}"
        )
    else:
      subsampling = min_sub
    self.fd_param['sub'] = subsampling

  def _calc_subsampling(self, vel_models, d_t, model_sampling):
    """Find time downsampling needed during propagation to remain stable.

    Args:
        models (nd nparray): model or models that will be propagated in. Should
          not include padding
        d_t (float): initial sampling rate

    Returns:
        int: amount that input  d_t is to be downsampled in order for nl prop to remain stable
    """
    max_vel = np.amax(vel_models)
    d_t_sub = math.ceil(max_vel * d_t / (min(model_sampling) * COURANT_LIMIT))
    return d_t_sub

  @abc.abstractmethod
  def _make_sep_wavelets(self, wavelet, d_t):
    pass

    # @abc.abstractmethod
    # def _calc_subsampling(self, model, d_t, model_sampling):
    raise NotImplementedError(
        '_calc_subsampling not overwritten by WaveEquation child class')

  @abc.abstractmethod
  def _pad_model(self, model: np.ndarray, padding: typing.Tuple,
                 fat: int) -> np.ndarray:
    raise NotImplementedError(
        '_pad_model not overwritten by WaveEquation child class')

  @abc.abstractmethod
  def _get_model_shape(self, model):
    pass

  @abc.abstractmethod
  def _setup_src_devices(self, src_locations, n_t):
    pass

  @abc.abstractmethod
  def _setup_rec_devices(self, rec_locations, n_t):
    pass

  @abc.abstractmethod
  def _setup_data(self, n_t, d_t, data=None):
    pass

  @abc.abstractmethod
  def _get_velocities(self, model):
    pass

  @abc.abstractmethod
  def _setup_operators(self, data_sep, model_sep, sep_par, src_devices,
                       rec_devices, wavelet_nl_sep, wavelet_lin_sep):
    pass

  @abc.abstractmethod
  def _get_free_surface_pad_minus(self):
    pass


class _NonlinearWaveCppOp(abc.ABC, Operator.Operator):
  """Wrapper encapsulating PYBIND11 module for the nonlinear wave propagator"""

  def __init__(self, model_sep, data_sep, sep_par, src_devices, rec_devices,
               wavelet_nl_sep, _nl_wave_pybind_class):
    self.setDomainRange(model_sep, data_sep)
    if not isinstance(src_devices[0], list):
      src_devices = [src_devices]
    if not isinstance(rec_devices[0], list):
      rec_devices = [rec_devices]
    self.wave_prop_operator = _nl_wave_pybind_class(model_sep.getCpp(),
                                                    sep_par.param, *src_devices,
                                                    *rec_devices)
    self.wavelet_nl_sep = wavelet_nl_sep

  def forward(self, add, model_sep, data_sep):
    #Setting elastic model parameters
    self.set_background(model_sep)
    with ostream_redirect():
      self.wave_prop_operator.forward(add, self.wavelet_nl_sep.getCpp(),
                                      data_sep.getCpp())

  def set_background(self, model_sep):
    with ostream_redirect():
      self.wave_prop_operator.setBackground(model_sep.getCpp())


class _JacobianWaveCppOp(abc.ABC, Operator.Operator):
  """Wrapper encapsulating PYBIND11 module for the Jocobian aka Born wave propagator"""

  def __init__(self, lin_model_sep, data_sep, velocity_sep, sep_par,
               src_devices, rec_devices, wavelet_lin_sep,
               _jac_wave_pybind_class):

    self.setDomainRange(lin_model_sep, data_sep)
    if not isinstance(src_devices[0], list):
      src_devices = [src_devices]
    if not isinstance(rec_devices[0], list):
      rec_devices = [rec_devices]

    self.wave_prop_operator = _jac_wave_pybind_class(velocity_sep.getCpp(),
                                                     sep_par.param,
                                                     *src_devices,
                                                     wavelet_lin_sep,
                                                     *rec_devices)

  def forward(self, add, model_sep, data_sep):
    with ostream_redirect():
      self.wave_prop_operator.forward(add, model_sep.getCpp(),
                                      data_sep.getCpp())

  def adjoint(self, add, model_sep, data_sep):
    with ostream_redirect():
      self.wave_prop_operator.adjoint(add, model_sep.getCpp(),
                                      data_sep.getCpp())

  def set_background(self, model_sep):
    with ostream_redirect():
      self.wave_prop_operator.setBackground(model_sep.getCpp())

  # def dot_product_test(self, verb=False, tolerance=1e-3):
  #   """Method to call the Cpp class dot-product test"""
  #   with ostream_redirect():
  #     result = self.dotTest(verb, tolerance)
  #   return result
