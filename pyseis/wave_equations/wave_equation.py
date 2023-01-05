"""Defines the abstract WaveEquation class.

A WaveEquation object is able to forward time march a finite difference wave
equation in two or three dimensions and sample the resulting wavefield(s) at
receiver locations. AcousticIsotropic and ElasticIsotropic child classes
implement different wave equations but share common methods that are defined in
WaveEquation.

"""
import numpy as np
import abc
from math import ceil
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
    'surfaceCondition': 'surface_condition'
}
COURANT_LIMIT = 0.45


class WaveEquation(abc.ABC):

  def __init__(self):
    self.model_sep = None
    self.fd_param = {'_BLOCK_SIZE': self._BLOCK_SIZE, '_FAT': self._FAT}
    self._nl_wave_op = None
    self._jac_wave_op = None

  def forward(self, model):
    self._setup_model(model)
    self._nl_wave_op.forward(0, self.model_sep, self.data_sep)
    return np.copy(self.data_sep.getNdArray())

  def jacobian(self, lin_model, background_model=None):
    self._setup_lin_model(lin_model)
    if background_model is not None:
      self._setup_model(background_model)
    if self._jac_wave_op is None:
      self._setup_jac_wave_op(self.data_sep, self.model_sep, self.sep_param,
                              self.src_devices, self.rec_devices,
                              self.wavelet_lin_sep, lin_model)
    # self._jac_wave_op.set_background(self.model_sep)
    self._jac_wave_op.forward(0, self.lin_model_sep, self.data_sep)
    return np.copy(self.data_sep.getNdArray())

  def jacobian_adjoint(self, lin_data, background_model=None):
    self._set_data(lin_data)
    if background_model is not None:
      self._setup_model(background_model)
    if self._jac_wave_op is None:
      self._setup_jac_wave_op(self.data_sep, self.model_sep, self.sep_param,
                              self.src_devices, self.rec_devices,
                              self.wavelet_lin_sep)
    # self._jac_wave_op.set_background(self.model_sep)
    self._jac_wave_op.adjoint(0, self.lin_model_sep, self.data_sep)
    return np.copy(self._truncate_model(self.lin_model_sep.getNdArray()))

  def dot_product_test(self, verb=False, tolerance=0.00001):
    if self._jac_wave_op is None:
      self._setup_jac_wave_op(self.data_sep, self.model_sep, self.sep_param,
                              self.src_devices, self.rec_devices,
                              self.wavelet_lin_sep)

    return self._jac_wave_op.dot_product_test(verb, tolerance)

  def _setup_wavelet(self, wavelet, d_t):
    self.fd_param['d_t'] = d_t
    self.fd_param['n_t'] = wavelet.shape[-1]
    self.fd_param['f_max'] = Wavelet.calc_max_freq(wavelet, d_t)
    self.wavelet_nl_sep, self.wavelet_lin_sep = self._make_sep_wavelets(
        wavelet, d_t)

  def _setup_sep_par(self, fd_param):
    sep_param_dict = {}
    for required_sep_param in self.required_sep_params:
      fd_param_key = SEP_PARAM_CYPHER[required_sep_param]
      if fd_param_key not in fd_param:
        raise RuntimeError(f'{fd_param_key} was not set.')
      sep_param_dict[required_sep_param] = fd_param[fd_param_key]

    self.sep_param = self._make_sep_par(sep_param_dict)

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
    self._nl_wave_op = _NonlinearWaveCppOp(model_sep, data_sep, sep_par,
                                           src_devices, rec_devices,
                                           wavelet_nl_sep,
                                           self._nl_wave_pybind_class)

  def _setup_jac_wave_op(self,
                         data_sep,
                         model_sep,
                         sep_par,
                         src_devices,
                         rec_devices,
                         wavelet_lin_sep,
                         lin_model=None):
    self.lin_model_sep = model_sep.clone()
    if lin_model is None:
      self.lin_model_sep.zero()
    else:
      lin_model = self._pad_model(lin_model, self.model_sampling,
                                  self.model_padding, self.model_origins,
                                  self.free_surface)[0]
      self.lin_model_sep.getNdArray()[:] = lin_model
    self._jac_wave_op = _JacobianWaveCppOp(self.lin_model_sep, data_sep,
                                           model_sep, sep_par, src_devices,
                                           rec_devices, wavelet_lin_sep,
                                           self._jac_wave_pybind_class)

  def _set_data(self, data):
    self.data_sep.getNdArray()[:] = data

  @abc.abstractmethod
  def _make_sep_wavelets(self, wavelet, d_t):
    pass

  @abc.abstractmethod
  def _calc_subsampling(self, model, d_t, model_sampling):
    raise NotImplementedError(
        '_calc_subsampling not overwritten by WaveEquation child class')

  @abc.abstractmethod
  def _pad_model(self, model, model_sampling, model_padding):
    raise NotImplementedError(
        '_pad_model not overwritten by WaveEquation child class')

  @abc.abstractmethod
  def _setup_model(self,
                   model,
                   model_sampling=None,
                   model_padding=None,
                   model_origins=None):
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

  def dot_product_test(self, verb=False, tolerance=1e-3):
    """Method to call the Cpp class dot-product test"""
    with ostream_redirect():
      result = self.dotTest(verb, tolerance)
    return result
