"""Defines the 2D and 3D AcousticIsotropic wave equation classes.

The AcousticIsotropic2D and AcousticIsotropic3D inherit from the abstract
AcousticIsotropic class. AcousticIsotropic2D and AcousticIsotropic3D can model
the acoustic, isotropic, constant-density wave equation in two and three
dimensions, respectively. With a pressure-wave velocity model, source wavelet,
source positions, and receiver positions, a user can forward model the wave
wave equation and sample at receiver locations. Pybind11 is used to wrap C++
code which then uses CUDA kernels to execute finite difference operations.
Current implementation parallelizes shots over gpus. 

  Typical usage example:
    #### 2D ##### 
    acoustic_2d = acoustic_isotropic.AcousticIsotropic2D(
      vp_model_nd_array,
      (d_x, d_z),
      wavelet_nd_array,
      d_t,
      src_locations_nd_array,
      rec_locations_nd_array,
      gpus=[0,1,2,4])
    data = acoustic_2d.fwd(vp_model_half_space)
"""
import numpy as np
from math import ceil
import Hypercube
import SepVector
from wave_equations import wave_equation
# 2d pybind11 modules
from pyAcoustic_iso_float_nl import deviceGpu as device_gpu_2d
from pyAcoustic_iso_float_nl import nonlinearPropShotsGpu, ostream_redirect
# 3d pybind11 modules
from pyAcoustic_iso_float_nl_3D import deviceGpu_3D as device_gpu_3d
from pyAcoustic_iso_float_nl_3D import nonlinearPropShotsGpu_3D 


class AcousticIsotropic(wave_equation.WaveEquation):
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
    d_t_sub = ceil(max_vel * d_t /
                   (min(model_sampling) * wave_equation.COURANT_LIMIT))
    return d_t_sub


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
    self.wave_prop_cpp_op_class = _Aco2dWavePropCppOp
    self.ostream_redirect = ostream_redirect

    self.model_sampling = model_sampling
    self.model_padding = model_padding
    self.model_origins = model_origins
    self.make(model, wavelet, d_t, src_locations, rec_locations, gpus)

  def setup_model(self, model):
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

  def setup_src_devices(self, src_locations, n_t):
    """
    src_locations - [n_src,(x_pos,z_pos)]
    """
    if self.model_sep is None:
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
          device_gpu_2d(z_coord_sep.getCpp(), x_coord_sep.getCpp(),
                        self.model_sep.getCpp(), int(n_t), 0, 0, 0))

    self.fd_param['n_src'] = len(source_devices)
    self.src_devices = source_devices

  def setup_rec_devices(self, rec_locations, n_t):
    """
    src_locations - [n_rec,(x_pos,z_pos)] OR [n_src,n_rec,(x_pos,z_pos)]
    """
    if self.model_sep is None:
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
          device_gpu_2d(z_coord_sep.getCpp(), x_coord_sep.getCpp(),
                        self.model_sep.getCpp(), int(n_t), 0, 0, 0))

    self.fd_param['n_src'] = len(rec_devices)
    self.fd_param['n_rec'] = n_rec
    self.rec_devices = rec_devices

  def setup_data(self, n_t, d_t):
    if 'n_src' not in self.fd_param:
      raise RuntimeError(
          'self.fd_param[\'n_src\'] must be set to set setup_data')
    if 'n_rec' not in self.fd_param:
      raise RuntimeError(
          'self.fd_param[\'n_rec\'] must be set to set setup_data')

    data_sep = SepVector.getSepVector(
        Hypercube.hypercube(axes=[
            Hypercube.axis(n=n_t, o=0.0, d=d_t),
            Hypercube.axis(n=self.fd_param['n_rec'], o=0.0, d=1.0),
            Hypercube.axis(n=self.fd_param['n_src'], o=0.0, d=1.0)
        ]))

    self.data_sep = data_sep

  def make_sep_wavelet(self, wavelet, d_t):
    n_t = wavelet.shape[-1]
    wavelet_sep = SepVector.getSepVector(
        Hypercube.hypercube(axes=[
            Hypercube.axis(n=n_t, o=0.0, d=d_t),
            Hypercube.axis(n=1),
            Hypercube.axis(n=1)
        ]))
    wavelet_sep.getNdArray()[:] = wavelet

    return wavelet_sep


class AcousticIsotropic3D(AcousticIsotropic):
  fat = 4

  def __init__(self,
               model,
               model_sampling,
               wavelet,
               d_t,
               src_locations,
               rec_locations,
               gpus,
               model_padding=(30, 30, 30),
               model_origins=(0.0, 0.0, 0.0)):
    super().__init__()
    self.required_sep_params = [
        'nx', 'dx', 'nz', 'dz', 'ny', 'dy', 'xPadMinus', 'xPadPlus',
        'zPadMinus', 'zPadPlus', 'yPad', 'dts', 'nts', 'fMax', 'sub', 'nShot',
        'iGpu', 'blockSize', 'fat', 'ginsu'
    ]
    self.wave_prop_cpp_op_class = _Aco3dWavePropCppOp
    self.ostream_redirect = ostream_redirect

    self.model_sampling = model_sampling
    self.model_padding = model_padding
    self.model_origins = model_origins
    self.make(model, wavelet, d_t, src_locations, rec_locations, gpus)

  def setup_model(self, model):

    model, y_pad, y_pad_plus, new_o_y, x_pad, x_pad_plus, new_o_x, z_pad, z_pad_plus, new_o_z = self.pad_model(
        model, self.model_sampling, self.model_padding, self.model_origins)
    self.model = model
    self.model_sep = SepVector.getSepVector(
        Hypercube.hypercube(axes=[
            Hypercube.axis(
                n=model.shape[2], o=new_o_z, d=self.model_sampling[2]),
            Hypercube.axis(
                n=model.shape[1], o=new_o_x, d=self.model_sampling[1]),
            Hypercube.axis(
                n=model.shape[0], o=new_o_y, d=self.model_sampling[0])
        ]))
    self.model_sep.getNdArray()[:] = self.model
    self.fd_param['n_y'] = model.shape[0]
    self.fd_param['n_x'] = model.shape[1]
    self.fd_param['n_z'] = model.shape[2]
    self.fd_param['d_y'] = self.model_sampling[0]
    self.fd_param['d_x'] = self.model_sampling[1]
    self.fd_param['d_z'] = self.model_sampling[2]
    self.fd_param['y_pad'] = y_pad
    self.fd_param['x_pad_minus'] = x_pad
    self.fd_param['x_pad_plus'] = x_pad_plus
    self.fd_param['z_pad_minus'] = z_pad
    self.fd_param['z_pad_plus'] = z_pad_plus

  def pad_model(self, model, model_sampling, model_padding, model_origins=None):
    """Pad 3d model.

    Finds the correct padding on either end of the axis so both directions are
    divisible by block_size for optimal gpu computation.

    Args:
        model (3d np array): 3d model to be padded. Should have shape
          (n_x,n_z).

    Returns:
        3d np array: padded 2d model.
    """
    #get dimensions
    n_y = model.shape[0]
    d_y = model_sampling[0]
    y_pad = model_padding[0]
    n_x = model.shape[1]
    d_x = model_sampling[1]
    x_pad = model_padding[1]
    n_z = model.shape[2]
    d_z = model_sampling[2]
    z_pad = model_padding[2]
    if model_origins is None:
      model_origins = (0.0, 0.0, 0.0)

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

    # compute y axis padding
    y_pad_plus = y_pad

    # pad
    model = np.pad(np.pad(model, ((y_pad, y_pad_plus), (x_pad, x_pad_plus),
                                  (z_pad, z_pad_plus)),
                          mode='edge'),
                   ((self.fat, self.fat), (self.fat, self.fat),
                    (self.fat, self.fat)),
                   mode='constant')

    # Compute new origins
    new_o_y = model_origins[0] - (self.fat + y_pad) * d_y
    new_o_x = model_origins[1] - (self.fat + x_pad) * d_x
    new_o_z = model_origins[2] - (self.fat + z_pad) * d_z

    return model, y_pad, y_pad_plus, new_o_y, x_pad, x_pad_plus, new_o_x, z_pad, z_pad_plus, new_o_z

  def setup_src_devices(self, src_locations, n_t):
    """
    src_locations - [n_src,(x_pos,z_pos)]
    """
    if self.model_sep is None:
      raise RuntimeError('self.model_sep must be set to set src devices')

    if 'n_src' in self.fd_param:
      assert self.fd_param['n_src'] == len(src_locations)

    sep_par = self.make_sep_par({
        'fat': self.fd_param['fat'],
        'zPadMinus': self.fd_param['z_pad_minus'],
        'zPadPlus': self.fd_param['z_pad_plus'],
        'xPadMinus': self.fd_param['x_pad_minus'],
        'xPadPlus': self.fd_param['x_pad_plus'],
        'yPad': self.fd_param['y_pad']
    })

    source_devices = []
    for src_location in src_locations:
      y_loc = src_location[0]
      x_loc = src_location[1]
      z_loc = src_location[2]
      y_coord_sep = SepVector.getSepVector(ns=[1]).set(y_loc)
      x_coord_sep = SepVector.getSepVector(ns=[1]).set(x_loc)
      z_coord_sep = SepVector.getSepVector(ns=[1]).set(z_loc)

      source_devices.append(
          device_gpu_3d(z_coord_sep.getCpp(), x_coord_sep.getCpp(),
                        y_coord_sep.getCpp(), self.model_sep.getCpp(), int(n_t),
                        sep_par.param, 0, 0, 0, 0, 'linear', 1))

    self.fd_param['n_src'] = len(source_devices)
    self.src_devices = source_devices

  def setup_rec_devices(self, rec_locations, n_t):
    """
    src_locations - [n_rec,(y_pos,x_pos,z_pos)] OR [n_shot,n_rec,(y_pos,x_pos,z_pos)]
    """
    if self.model_sep is None:
      raise RuntimeError('self.model_sep must be set to set src devices')

    if len(rec_locations.shape) == 3:
      n_src = rec_locations.shape[0]
      n_rec = rec_locations.shape[1]
      if 'n_src' in self.fd_param:
        if self.fd_param['n_src'] != n_src:
          raise RuntimeError(
              'number of shots in rec_locations does not match n_src from set_src_locations'
          )
    else:  # fixed receivers
      n_rec = rec_locations.shape[0]
      if 'n_src' not in self.fd_param:
        raise RuntimeError('to make rec devices, src devices must be set first')
      n_src = self.fd_param['n_src']
      rec_locations = np.repeat(np.expand_dims(rec_locations, axis=0),
                                n_src,
                                axis=0)

    n_rec = rec_locations.shape[1]
    rec_devices = []
    for rec_location in rec_locations:
      y_coord_sep = SepVector.getSepVector(ns=[n_rec])
      y_coord_sep.getNdArray()[:] = rec_location[:, 0]
      x_coord_sep = SepVector.getSepVector(ns=[n_rec])
      x_coord_sep.getNdArray()[:] = rec_location[:, 1]
      z_coord_sep = SepVector.getSepVector(ns=[n_rec])
      z_coord_sep.getNdArray()[:] = rec_location[:, 2]
      sep_par = self.make_sep_par({
          'fat': self.fd_param['fat'],
          'zPadMinus': self.fd_param['z_pad_minus'],
          'zPadPlus': self.fd_param['z_pad_plus'],
          'xPadMinus': self.fd_param['x_pad_minus'],
          'xPadPlus': self.fd_param['x_pad_plus'],
          'yPad': self.fd_param['y_pad']
      })
      rec_devices.append(
          device_gpu_3d(z_coord_sep.getCpp(), x_coord_sep.getCpp(),
                        y_coord_sep.getCpp(), self.model_sep.getCpp(), int(n_t),
                        sep_par.param, 0, 0, 0, 0, 'linear', 1))

    self.fd_param['n_rec'] = n_rec
    self.rec_devices = rec_devices

  def setup_data(self, n_t, d_t):
    if 'n_src' not in self.fd_param:
      raise RuntimeError(
          'self.fd_param[\'n_shots\'] must be set to set setup_data')
    if 'n_rec' not in self.fd_param:
      raise RuntimeError(
          'self.fd_param[\'n_rec\'] must be set to set setup_data')

    data_sep = SepVector.getSepVector(
        Hypercube.hypercube(axes=[
            Hypercube.axis(n=n_t, o=0.0, d=d_t),
            Hypercube.axis(n=self.fd_param['n_rec'], o=0.0, d=1.0),
            Hypercube.axis(n=self.fd_param['n_src'], o=0.0, d=1.0)
        ]))

    self.data_sep = data_sep

  def make_sep_wavelet(self, wavelet, d_t):
    n_t = wavelet.shape[-1]
    wavelet_sep = SepVector.getSepVector(
        Hypercube.hypercube(
            axes=[Hypercube.axis(n=n_t, o=0.0, d=d_t),
                  Hypercube.axis(n=1)]))
    wavelet_sep.getNdArray()[:] = wavelet

    return wavelet_sep


class _Aco2dWavePropCppOp(wave_equation._WavePropCppOp):
  """Wrapper encapsulating PYBIND11 module for the wave propagator"""

  wave_prop_module = nonlinearPropShotsGpu

  def set_background(self, model_sep):
    with ostream_redirect():
      self.wave_prop_operator.setVel(model_sep.getCpp())


class _Aco3dWavePropCppOp(wave_equation._WavePropCppOp):
  """Wrapper encapsulating PYBIND11 module for the wave propagator"""

  wave_prop_module = nonlinearPropShotsGpu_3D

  def set_background(self, model_sep):
    with ostream_redirect():
      self.wave_prop_operator.setVel_3D(model_sep.getCpp())