import numpy as np
from math import ceil
from wave_equations.WaveEquation import WaveEquation, COURANT_LIMIT


class AcousticIsotropic(WaveEquation):
  block_size = 16
  fat = 5

  def __init__(self):
    super().__init__()

  def make(self, model, wavelet, d_t, src_locations, rec_locations, gpus):
    # pads model, makes self.model_sep, and updates self.fd_param
    self.set_model(model)

    # make and set wavelet
    self.set_wavelet(wavelet, d_t)
    n_t = wavelet.shape[-1]

    # make and set source devices
    self.set_src_devices(src_locations, n_t)

    # make and set rec devices
    self.set_rec_devices(rec_locations, n_t)

    # make and set data space
    self.set_data(n_t, d_t)

    # calculate and find subsampling
    self.set_subsampling(model, d_t, self.model_sampling)

    # set gpus list
    self.fd_param['gpus'] = str(gpus)[1:-1]

    #set ginsu
    self.fd_param['ginsu'] = 0

    # make and set sep par
    self.set_sep_par(self.fd_param)

    # make and set gpu operator
    self.set_gpu_operator(self.get_data_sep(), self.get_model_sep(),
                          self.get_sep_param(), self.get_src_devices(),
                          self.get_rec_devices())

  def get_subsampling(self):
    if 'sub' not in self.fd_param:
      raise RuntimeError('subsampling has not been set')
    return self.fd_param['sub']

  def set_subsampling(self, model, d_t, model_sampling):
    sub = self.find_subsampling(model, d_t, model_sampling)
    if 'sub' in self.fd_param:
      if sub > self.fd_param['sub']:
        raise RuntimeError(
            'Newly set model requires greater subsampling than what wave equation operator was initialized with. This is currently not allowed.'
        )
    self.fd_param['sub'] = sub

  def find_subsampling(self, model, d_t, model_sampling):
    """Find time downsampling needed during propagation to remain stable.

    Args:
        models (nd nparray): model or models that will be propagated in. Should
          not include padding
        d_t (float): initial sampling rate

    Returns:
        int: amount that input  d_t is to be downsampled in order for nl prop to remain stable
    """
    max_vel = np.amax(model)
    # d_t_sub = COURANT_LIMIT * min(model_sampling) / max_vel
    d_t_sub = ceil(max_vel * d_t / (min(model_sampling) * COURANT_LIMIT))
    return d_t_sub