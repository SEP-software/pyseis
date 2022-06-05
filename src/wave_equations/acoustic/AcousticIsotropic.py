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
    self.setup_model(model)

    # make and set wavelet
    self.setup_wavelet(wavelet, d_t)
    n_t = wavelet.shape[-1]

    # make and set source devices
    self.setup_src_devices(src_locations, n_t)

    # make and set rec devices
    self.setup_rec_devices(rec_locations, n_t)

    # make and set data space
    self.setup_data(n_t, d_t)

    # calculate and find subsampling
    self.setup_subsampling(model, d_t, self.model_sampling)

    # set gpus list
    self.fd_param['gpus'] = str(gpus)[1:-1]

    #set ginsu
    self.fd_param['ginsu'] = 0

    # make and set sep par
    self.setup_sep_par(self.fd_param)

    # make and set gpu operator
    self.setup_wave_prop_operator(self.data_sep, self.model_sep, self.sep_param,
                                  self.src_devices, self.rec_devices,
                                  self.wavelet_sep)

  def setup_subsampling(self, model, d_t, model_sampling):
    sub = self.calc_subsampling(model, d_t, model_sampling)
    if 'sub' in self.fd_param:
      if sub > self.fd_param['sub']:
        raise RuntimeError(
            'Newly set model requires greater subsampling than what wave equation operator was initialized with. This is currently not allowed.'
        )
    self.fd_param['sub'] = sub

  def calc_subsampling(self, model, d_t, model_sampling):
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