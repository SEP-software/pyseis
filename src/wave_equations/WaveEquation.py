import numpy as np
import abc
from math import ceil
import genericIO

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
    'block_size': 'block_size',
    'blockSize': 'block_size',
    'fat': 'fat',
    'ginsu': 'ginsu'
}
COURANT_LIMIT = 0.45


class WaveEquation(abc.ABC):

  def __init__(self):
    self.model_sep = None
    # self.gpu_operator = None
    self.fd_param = {'block_size': self.block_size, 'fat': self.fat}
    # self.ostream_redirect = None

  def set_wavelet(self, wavelet, d_t):
    wavelet = self.wavelet_module(wavelet, d_t)
    self.wavelet_sep = wavelet.get_sep()
    self.fd_param['d_t'] = d_t
    self.fd_param['n_t'] = wavelet.n_t
    self.fd_param['f_max'] = wavelet.f_max

  def get_sep_param(self):
    return self.sep_param

  def set_sep_par(self, fd_param):
    sep_param_dict = {}
    for required_sep_param in self.required_sep_params:
      fd_param_key = SEP_PARAM_CYPHER[required_sep_param]
      if fd_param_key not in fd_param:
        raise RuntimeError(f'{fd_param_key} was not set.')
      sep_param_dict[required_sep_param] = fd_param[fd_param_key]

    self.sep_param = self.to_sep(sep_param_dict)

  def to_sep(self, dict):
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

  def get_subsampling(self):
    if 'sub' not in self.fd_param:
      raise RuntimeError('subsampling has not been set')
    return self.fd_param['sub']

  def get_model_sep(self):
    return self.model_sep

  def get_wavelet_sep(self):
    return self.wavelet_sep

  def get_src_devices(self):
    return self.src_devices

  def get_rec_devices(self):
    return self.rec_devices

  def get_data_sep(self):
    return self.data_sep

  def set_gpu_operator(self, data_sep, model_sep, sep_par, src_devices,
                       rec_devices):
    if not isinstance(src_devices[0], list):
      src_devices = [src_devices]
    if not isinstance(rec_devices[0], list):
      rec_devices = [rec_devices]
    self.gpu_operator = self.gpu_module(model_sep.getCpp(), sep_par.param,
                                        *src_devices, *rec_devices)

  def fwd(self, model):
    self.set_model(model)
    self.set_background(self.get_model_sep())
    if self.gpu_operator is None:
      raise RuntimeError("self.gpu_operator has not been made")

    with self.ostream_redirect():
      self.gpu_operator.forward(0,
                                self.get_wavelet_sep().getCpp(),
                                self.get_data_sep().getCpp())
    return self.get_data_sep().getNdArray()

  @abc.abstractmethod
  def set_background(self, model):
    pass

  @abc.abstractmethod
  def find_subsampling(self, model, d_t, model_sampling):
    raise NotImplementedError(
        'find_subsampling not overwritten by Wavelet child class')

  @abc.abstractmethod
  def pad_model(self, model, model_sampling, model_padding):
    raise NotImplementedError(
        'pad_model not overwritten by Wavelet child class')

  @abc.abstractmethod
  def set_model(self,
                model,
                model_sampling=None,
                model_padding=None,
                model_origins=None):
    pass

  @abc.abstractmethod
  def set_src_devices(self, src_locations, n_t):
    pass

  @abc.abstractmethod
  def set_rec_devices(self, rec_locations, n_t):
    pass

  @abc.abstractmethod
  def set_data(self, n_t, d_t):
    pass
