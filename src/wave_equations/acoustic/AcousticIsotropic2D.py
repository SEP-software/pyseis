from math import ceil
import numpy as np
import Hypercube, SepVector
from pyAcoustic_iso_float_nl import deviceGpu as device_gpu
from pyAcoustic_iso_float_nl import nonlinearPropShotsGpu, ostream_redirect
from wave_equations.acoustic.AcousticIsotropic import AcousticIsotropic
from wavelets.acoustic import Acoustic2D


class AcousticIsotropic2D(AcousticIsotropic):

  def __init__(self,
               model,
               model_sampling,
               wavelet,
               d_t,
               src_locations,
               rec_locations,
               gpus,
               model_padding=(50, 50),
               model_origins=(0.0, 0.0)):
    super().__init__()
    self.required_sep_params = [
        'nx', 'dx', 'nz', 'dz', 'xPadMinus', 'xPadPlus', 'zPadMinus',
        'zPadPlus', 'dts', 'nts', 'fMax', 'sub', 'nShot', 'iGpu', 'blockSize',
        'fat'
    ]
    self.wavelet_module = Acoustic2D.Acoustic2D
    self.gpu_module = nonlinearPropShotsGpu
    self.ostream_redirect = ostream_redirect

    self.model_sampling = model_sampling
    self.model_padding = model_padding
    self.model_origins = model_origins
    self.make(model, wavelet, d_t, src_locations, rec_locations, gpus)

  def set_model(self, model):
    model, x_pad, x_pad_plus, new_o_x, z_pad, z_pad_plus, new_o_z = self.pad_model(
        model, self.model_sampling, self.model_padding, self.model_origins)
    self.model = model
    self.model_sep = SepVector.getSepVector(
        Hypercube.hypercube(axes=[
            Hypercube.axis(
                n=model.shape[1], o=new_o_z, d=self.model_sampling[1]),
            Hypercube.axis(
                n=model.shape[0], o=new_o_x, d=self.model_sampling[0])
        ]))
    self.model_sep.getNdArray()[:] = self.model
    self.fd_param['n_x'] = model.shape[0]
    self.fd_param['n_z'] = model.shape[1]
    self.fd_param['d_x'] = self.model_sampling[0]
    self.fd_param['d_z'] = self.model_sampling[1]
    self.fd_param['x_pad_minus'] = x_pad
    self.fd_param['x_pad_plus'] = x_pad_plus
    self.fd_param['z_pad_minus'] = z_pad
    self.fd_param['z_pad_plus'] = z_pad_plus

  def pad_model(self, model, model_sampling, model_padding, model_origins=None):
    """Pad 2d model.

    Finds the correct padding on either end of the axis so both directions are
    divisible by block_size for optimal gpu computation.

    Args:
        model (2d np array): 2d model to be padded. Should have shape
          (n_x,n_z).
        d_x (float): sampling rate of x axis.
        d_z (float): sampling rate of z axis
        x_pad (int): desired padding for beginning and end of x axis.
        z_pad ([type]): desired padding for beginning and end of z axis.

    Returns:
        2d np array: padded 2d model.
    """
    #get dimensions
    n_x = model.shape[0]
    d_x = model_sampling[0]
    x_pad = model_padding[0]
    n_z = model.shape[1]
    d_z = model_sampling[1]
    z_pad = model_padding[1]
    if model_origins is None:
      model_origins = (0.0, 0.0)

    # Compute size of z_pad_plus
    n_z_total = z_pad * 2 + n_z
    ratio_z = n_z_total / self.block_size
    nb_blockz = ceil(ratio_z)
    z_pad_plus = nb_blockz * self.block_size - n_z - z_pad

    # Compute sixe of x_pad_plus
    n_x_total = x_pad * 2 + n_x
    ratio_x = n_x_total / self.block_size
    nb_blockx = ceil(ratio_x)
    x_pad_plus = nb_blockx * self.block_size - n_x - x_pad

    # pad
    model = np.pad(np.pad(model, ((x_pad, x_pad_plus), (z_pad, z_pad_plus)),
                          mode='edge'),
                   ((self.fat, self.fat), (self.fat, self.fat)),
                   mode='constant')

    # Compute new origins
    new_o_x = model_origins[0] - (self.fat + x_pad) * d_x
    new_o_z = model_origins[1] - (self.fat + z_pad) * d_z

    return model, x_pad, x_pad_plus, new_o_x, z_pad, z_pad_plus, new_o_z

  def set_src_devices(self, src_locations, n_t):
    """
    src_locations - [n_src,(x_pos,z_pos)]
    """
    if self.get_model_sep() is None:
      raise RuntimeError('self.model_sep must be set to set src devices')

    if 'n_src' in self.fd_param:
      assert self.fd_param['n_src'] == len(src_locations)

    source_devices = []
    for src_location in src_locations:
      x_loc = src_location[0]
      z_loc = src_location[1]
      x_coord_sep = SepVector.getSepVector(ns=[1]).set(x_loc)
      z_coord_sep = SepVector.getSepVector(ns=[1]).set(z_loc)
      source_devices.append(
          device_gpu(z_coord_sep.getCpp(), x_coord_sep.getCpp(),
                     self.get_model_sep().getCpp(), int(n_t), 0, 0, 0))

    self.fd_param['n_src'] = len(source_devices)
    self.src_devices = source_devices

  def set_rec_devices(self, rec_locations, n_t):
    """
    src_locations - [n_rec,(x_pos,z_pos)] OR [n_src,n_rec,(x_pos,z_pos)]
    """
    if self.get_model_sep() is None:
      raise RuntimeError('self.model_sep must be set to set src devices')

    if len(rec_locations.shape) == 3:
      n_src = rec_locations.shape[0]
      n_rec = rec_locations.shape[1]
      if 'n_src' in self.fd_param:
        if self.fd_param['n_src'] != n_src:
          raise RuntimeError(
              'number of shots in rec_locations does not match n_src from set_src_locations'
          )
    else:
      n_rec = rec_locations.shape[0]
      if 'n_src' not in self.fd_param:
        raise RuntimeError('to make rec devices, src devices must be set first')
      n_src = self.fd_param['n_src']
      rec_locations = np.repeat(np.expand_dims(rec_locations, axis=0),
                                n_src,
                                axis=0)

    rec_devices = []
    for rec_location in rec_locations:
      x_coord_sep = SepVector.getSepVector(ns=[n_rec])
      x_coord_sep.getNdArray()[:] = rec_location[:, 0]
      z_coord_sep = SepVector.getSepVector(ns=[n_rec])
      z_coord_sep.getNdArray()[:] = rec_location[:, 1]
      rec_devices.append(
          device_gpu(z_coord_sep.getCpp(), x_coord_sep.getCpp(),
                     self.get_model_sep().getCpp(), int(n_t), 0, 0, 0))

    self.fd_param['n_src'] = len(rec_devices)
    self.fd_param['n_rec'] = n_rec
    self.rec_devices = rec_devices

  def set_data(self, n_t, d_t):
    if 'n_src' not in self.fd_param:
      raise RuntimeError('self.fd_param[\'n_src\'] must be set to set set_data')
    if 'n_rec' not in self.fd_param:
      raise RuntimeError('self.fd_param[\'n_rec\'] must be set to set set_data')

    data_sep = SepVector.getSepVector(
        Hypercube.hypercube(axes=[
            Hypercube.axis(n=n_t, o=0.0, d=d_t),
            Hypercube.axis(n=self.fd_param['n_rec'], o=0.0, d=1.0),
            Hypercube.axis(n=self.fd_param['n_src'], o=0.0, d=1.0)
        ]))

    self.data_sep = data_sep

  def set_background(self, model):
    if "getCpp" in dir(model):
      model = model.getCpp()
    with self.ostream_redirect():
      self.gpu_operator.setVel(model)
