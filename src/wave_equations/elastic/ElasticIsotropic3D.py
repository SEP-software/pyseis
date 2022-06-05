from math import ceil
import numpy as np
import Hypercube, SepVector
from pyElastic_iso_float_nl_3D import spaceInterpGpu_3D as device_gpu
from pyElastic_iso_float_nl_3D import nonlinearPropElasticShotsGpu_3D, ostream_redirect
from wave_equations.elastic.ElasticIsotropic import ElasticIsotropic, convert_to_lame, convert_to_vel
from dataCompModule_3D import ElasticDatComp_3D as _ElasticDatComp_3D
import pyOperator as Operator
from wave_equations.WaveEquation import _WavePropCppOp


class ElasticIsotropic3D(ElasticIsotropic):
  N_WFLD_COMPONENTS = 9
  N_MODEL_PARAMETERS = 3

  def __init__(self,
               model,
               model_sampling,
               wavelet,
               d_t,
               src_locations,
               rec_locations,
               gpus,
               model_padding=(30, 30, 30),
               model_origins=(0.0, 0.0, 0.0),
               lame_model=False,
               recording_components=[
                   'vx', 'vy', 'vz', 'sxx', 'syy', 'szz', 'sxz', 'sxy', 'syz'
               ]):
    super().__init__()
    self.required_sep_params = [
        'ny', 'dy', 'nx', 'dx', 'nz', 'dz', 'yPad', 'xPadMinus', 'xPadPlus',
        'zPadMinus', 'zPadPlus', 'mod_par', 'dts', 'nts', 'fMax', 'sub', 'nExp',
        'iGpu', 'blockSize', 'fat'
    ]
    self.wave_prop_cpp_op_class = _Ela3dWavePropCppOp

    self.model_sampling = model_sampling
    self.model_padding = model_padding
    self.model_origins = model_origins
    self.make(model, wavelet, d_t, src_locations, rec_locations, gpus,
              recording_components, lame_model)

  def setup_model(self, model, lame_model=False):
    if not lame_model:
      model = convert_to_lame(model)

    self.fd_param['mod_par'] = 1
    model, y_pad, y_pad_plus, new_o_y, x_pad, x_pad_plus, new_o_x, z_pad, z_pad_plus, new_o_z = self.pad_model(
        model, self.model_sampling, self.model_padding, self.model_origins)
    self.model = model
    self.model_sep = SepVector.getSepVector(
        Hypercube.hypercube(axes=[
            Hypercube.axis(
                n=model.shape[3], o=new_o_z, d=self.model_sampling[2]),
            Hypercube.axis(
                n=model.shape[2], o=new_o_x, d=self.model_sampling[1]),
            Hypercube.axis(
                n=model.shape[1], o=new_o_y, d=self.model_sampling[0]),
            Hypercube.axis(n=model.shape[0], o=0.0, d=1.0)
        ]))
    self.model_sep.getNdArray()[:] = self.model
    self.fd_param['n_y'] = model.shape[1]
    self.fd_param['n_x'] = model.shape[2]
    self.fd_param['n_z'] = model.shape[3]
    self.fd_param['o_y'] = new_o_y
    self.fd_param['o_x'] = new_o_x
    self.fd_param['o_z'] = new_o_z
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
        d_x (float): sampling rate of x axis.
        d_z (float): sampling rate of z axis
        x_pad (int): desired padding for beginning and end of x axis.
        z_pad ([type]): desired padding for beginning and end of z axis.

    Returns:
        3d np array: padded 3d model.
    """
    #get dimensions
    n_y = model.shape[1]
    d_y = model_sampling[0]
    y_pad = model_padding[0]
    n_x = model.shape[2]
    d_x = model_sampling[1]
    x_pad = model_padding[1]
    n_z = model.shape[3]
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
    model = np.pad(np.pad(model, ((0, 0), (y_pad, y_pad_plus),
                                  (x_pad, x_pad_plus), (z_pad, z_pad_plus)),
                          mode='edge'),
                   ((0, 0), (self.fat, self.fat), (self.fat, self.fat),
                    (self.fat, self.fat)),
                   mode='constant')

    # Compute new origins
    new_o_y = model_origins[0] - (self.fat + y_pad) * d_y
    new_o_x = model_origins[1] - (self.fat + x_pad) * d_x
    new_o_z = model_origins[2] - (self.fat + z_pad) * d_z

    return model, y_pad, y_pad_plus, new_o_y, x_pad, x_pad_plus, new_o_x, z_pad, z_pad_plus, new_o_z

  def setup_src_devices(self, src_locations, n_t, interp_method='linear'):
    """
    src_locations - [n_src,(x_pos,z_pos)]
    """
    if self.model_sep is None:
      raise RuntimeError('self.model_sep must be set to set src devices')

    if 'n_src' in self.fd_param:
      assert self.fd_param['n_src'] == len(src_locations)

    staggered_grid_hypers = self.make_staggered_grid_hypers(
        self.fd_param['n_y'], self.fd_param['n_x'], self.fd_param['n_z'],
        self.fd_param['o_y'], self.fd_param['o_x'], self.fd_param['o_z'],
        self.fd_param['d_y'], self.fd_param['d_x'], self.fd_param['d_z'])

    sep_par = self.make_sep_par({
        'fat': self.fd_param['fat'],
        'zPadMinus': self.fd_param['z_pad_minus'],
        'zPadPlus': self.fd_param['z_pad_plus'],
        'xPadMinus': self.fd_param['x_pad_minus'],
        'xPadPlus': self.fd_param['x_pad_plus'],
        'yPad': self.fd_param['y_pad']
    })

    src_devices_staggered_grids = [[], [], [], [], [], [], []]
    for src_location in src_locations:
      y_loc = src_location[0]
      x_loc = src_location[1]
      z_loc = src_location[2]
      y_coord_sep = SepVector.getSepVector(ns=[1]).set(y_loc)
      x_coord_sep = SepVector.getSepVector(ns=[1]).set(x_loc)
      z_coord_sep = SepVector.getSepVector(ns=[1]).set(z_loc)
      for src_devices_staggered_grid, staggered_grid_hyper in zip(
          src_devices_staggered_grids, staggered_grid_hypers):
        src_devices_staggered_grid.append(
            device_gpu(z_coord_sep.getCpp(), x_coord_sep.getCpp(),
                       y_coord_sep.getCpp(), staggered_grid_hyper.getCpp(),
                       int(n_t), sep_par.param, 0, 0, 0, 0, interp_method, 1))

    self.fd_param['n_src'] = len(src_locations)
    self.src_devices = src_devices_staggered_grids

  def setup_rec_devices(self,
                        rec_locations,
                        n_t,
                        interp_method='linear',
                        interp_n_filters=4):
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
      # rec_locations = np.repeat(np.expand_dims(rec_locations, axis=0),
      #                           n_src,
      #                           axis=0)
      rec_locations = np.expand_dims(rec_locations, axis=0)

    staggered_grid_hypers = self.make_staggered_grid_hypers(
        self.fd_param['n_y'], self.fd_param['n_x'], self.fd_param['n_z'],
        self.fd_param['o_y'], self.fd_param['o_x'], self.fd_param['o_z'],
        self.fd_param['d_y'], self.fd_param['d_x'], self.fd_param['d_z'])

    sep_par = self.make_sep_par({
        'fat': self.fd_param['fat'],
        'zPadMinus': self.fd_param['z_pad_minus'],
        'zPadPlus': self.fd_param['z_pad_plus'],
        'xPadMinus': self.fd_param['x_pad_minus'],
        'xPadPlus': self.fd_param['x_pad_plus'],
        'yPad': self.fd_param['y_pad']
    })

    rec_devices_staggered_grids = [[], [], [], [], [], [], []]
    for rec_location in rec_locations:
      y_coord_sep = SepVector.getSepVector(ns=[n_rec])
      y_coord_sep.getNdArray()[:] = rec_location[:, 0]
      x_coord_sep = SepVector.getSepVector(ns=[n_rec])
      x_coord_sep.getNdArray()[:] = rec_location[:, 1]
      z_coord_sep = SepVector.getSepVector(ns=[n_rec])
      z_coord_sep.getNdArray()[:] = rec_location[:, 2]

      for rec_devices_staggered_grid, staggered_grid_hyper in zip(
          rec_devices_staggered_grids, staggered_grid_hypers):
        rec_devices_staggered_grid.append(
            device_gpu(z_coord_sep.getCpp(), x_coord_sep.getCpp(),
                       y_coord_sep.getCpp(), staggered_grid_hyper.getCpp(),
                       int(n_t), sep_par.param, 0, 0, 0, 0, interp_method, 1))

    self.fd_param['n_rec'] = n_rec
    self.rec_devices = rec_devices_staggered_grids

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
            Hypercube.axis(n=self.N_WFLD_COMPONENTS, o=0.0, d=1),
            Hypercube.axis(n=self.fd_param['n_src'], o=0.0, d=1.0)
        ]))

    self.data_sep = data_sep

  def make_sep_wavelet(self, wavelet, d_t):
    n_t = wavelet.shape[-1]
    wavelet_sep = SepVector.getSepVector(
        Hypercube.hypercube(axes=[
            Hypercube.axis(n=n_t, o=0.0, d=d_t),
            Hypercube.axis(n=1),
            Hypercube.axis(n=self.N_WFLD_COMPONENTS),
            Hypercube.axis(n=1)
        ]))
    wavelet_sep.getNdArray().flat[:] = wavelet

    return wavelet_sep

  def make_staggered_grid_hypers(self, n_y, n_x, n_z, o_y, o_x, o_z, d_y, d_x,
                                 d_z):
    z_axis = Hypercube.axis(n=n_z, o=o_z, d=d_z)
    z_axis_staggered = Hypercube.axis(n=n_z, o=o_z - 0.5 * d_z, d=d_z)

    x_axis = Hypercube.axis(n=n_x, o=o_x, d=d_x)
    x_axis_staggered = Hypercube.axis(n=n_x, o=o_x - 0.5 * d_x, d=d_x)

    y_axis = Hypercube.axis(n=n_y, o=o_y, d=d_y)
    y_axis_staggered = Hypercube.axis(n=n_y, o=o_y - 0.5 * d_y, d=d_y)

    param_axis = Hypercube.axis(n=self.N_MODEL_PARAMETERS, o=0.0, d=1)

    center_grid_hyper = Hypercube.hypercube(
        axes=[z_axis, x_axis, y_axis, param_axis])
    x_staggered_grid_hyper = Hypercube.hypercube(
        axes=[z_axis, x_axis_staggered, y_axis, param_axis])
    y_staggered_grid_hyper = Hypercube.hypercube(
        axes=[z_axis, x_axis, y_axis_staggered, param_axis])
    z_staggered_grid_hyper = Hypercube.hypercube(
        axes=[z_axis_staggered, x_axis, y_axis, param_axis])
    xz_staggered_grid_hyper = Hypercube.hypercube(
        axes=[z_axis_staggered, x_axis_staggered, y_axis, param_axis])
    xy_staggered_grid_hyper = Hypercube.hypercube(
        axes=[z_axis, x_axis_staggered, y_axis_staggered, param_axis])
    yz_staggered_grid_hyper = Hypercube.hypercube(
        axes=[z_axis_staggered, x_axis, y_axis_staggered, param_axis])

    return center_grid_hyper, x_staggered_grid_hyper, y_staggered_grid_hyper, z_staggered_grid_hyper, xz_staggered_grid_hyper, xy_staggered_grid_hyper, yz_staggered_grid_hyper

  def setup_wavefield_sampling_operator(self, recording_components, data_sep):
    # _ElasticDatComp_3D expects a string of comma seperated values
    recording_components = ",".join(recording_components)
    #make sampling opeartor
    wavefield_sampling_operator = _ElasticDatComp_3D(recording_components,
                                                     data_sep)
    self.data_sep = wavefield_sampling_operator.range.clone()

    self.wave_prop_cpp_op = Operator.ChainOperator(self.wave_prop_cpp_op,
                                                   wavefield_sampling_operator)


class _Ela3dWavePropCppOp(_WavePropCppOp):
  """Wrapper encapsulating PYBIND11 module for the wave propagator"""

  wave_prop_module = nonlinearPropElasticShotsGpu_3D

  def set_background(self, model_sep):
    with ostream_redirect():
      self.wave_prop_operator.setBackground(model_sep.getCpp())