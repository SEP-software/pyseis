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
import wavelets.Wavelet as Wavelet

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
    'ginsu': 'ginsu'
}
COURANT_LIMIT = 0.45


class WaveEquation(abc.ABC, Operator.Operator):

  def __init__(self):
    self.model_sep = None
    self.fd_param = {'_BLOCK_SIZE': self._BLOCK_SIZE, '_FAT': self._FAT}

  def fwd(self, model):
    self._setup_model(model)
    self.wave_prop_cpp_op.forward(0, self.model_sep, self.data_sep)
    return self.data_sep.getNdArray()

  def forward(self, add, model_sep, data_sep):
    with self.ostream_redirect():
      self.wave_prop_cpp_op.forward(0, model_sep, data_sep)

  def _setup_wavelet(self, wavelet, d_t):
    self.fd_param['d_t'] = d_t
    self.fd_param['n_t'] = wavelet.shape[-1]
    self.fd_param['f_max'] = Wavelet.calc_max_freq(wavelet, d_t)
    self.wavelet_sep = self._make_sep_wavelet(wavelet, d_t)

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

  def _setup_wave_prop_operator(self, data_sep, model_sep, sep_par, src_devices,
                                rec_devices, wavelet_sep):
    self.wave_prop_cpp_op = self.wave_prop_cpp_op_class(model_sep, data_sep,
                                                        sep_par, src_devices,
                                                        rec_devices,
                                                        wavelet_sep)
    self.setDomainRange(model_sep, data_sep)

  @abc.abstractmethod
  def _make_sep_wavelet(self, wavelet, d_t):
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
  def _setup_data(self, n_t, d_t):
    pass


class _WavePropCppOp(abc.ABC, Operator.Operator):
  """Wrapper encapsulating PYBIND11 module for the wave propagator"""

  def __init__(self, model_sep, data_sep, sep_par, src_devices, rec_devices,
               wavelet_sep):
    self.setDomainRange(model_sep, data_sep)
    if not isinstance(src_devices[0], list):
      src_devices = [src_devices]
    if not isinstance(rec_devices[0], list):
      rec_devices = [rec_devices]
    self.wave_prop_operator = self.wave_prop_module(model_sep.getCpp(),
                                                    sep_par.param, *src_devices,
                                                    *rec_devices)
    self.wavelet_sep = wavelet_sep

  def forward(self, add, model_sep, data_sep):
    #Setting elastic model parameters
    self.set_background(model_sep)
    with ostream_redirect():
      self.wave_prop_operator.forward(add, self.wavelet_sep.getCpp(),
                                      data_sep.getCpp())

  @abc.abstractmethod
  def set_background(self, model_sep):
    pass