"""Defines the 2D and 3D ElasticIsotropic wave equation classes.

The ElasticIsotropic2D and ElasticIsotropic3D inherit from the abstract
ElasticIsotropic class. ElasticIsotropic2D and ElasticIsotropic3D can model
the elastic, isotropic wave equation in two and three dimensions, respectively,
using a staggered-grid implementation. With an elastic earth model parameterized
by (v_p, v_s, and rho), a source wavelet, source positions, and receiver
positions, a user can forward model the wave wave equation and sample at
receiver locations. Pybind11 is used to wrap C++ code which then uses CUDA 
kernels to execute finite difference operations. Current implementation
parallelizes shots over gpus. 

  Typical usage example:
    #### 2D ##### 
    Elastic_2d = Elastic_isotropic.ElasticIsotropic2D(
      vp_vs_rho_model_nd_array,
      (d_x, d_z),
      wavelet_nd_array,
      d_t,
      src_locations_nd_array,
      rec_locations_nd_array,
      gpus=[0,1,2,4])
    data = Elastic_2d.forward(vp_model_half_space)
"""
import numpy as np
from math import ceil
import abc
import typing

import Hypercube
import SepVector
import pyOperator as Operator
from pyseis.wave_equations import wave_equation
# 2d pybind modules
from pyElastic_iso_float_nl import spaceInterpGpu as device_gpu_2d
from pyElastic_iso_float_nl import nonlinearPropElasticShotsGpu, ostream_redirect
from pyElastic_iso_float_born import BornElasticShotsGpu
from dataCompModule import ElasticDatComp as _ElasticDatComp2D
from elasticParamConvertModule import ElasticConv as _ElasticModelConv2D
from elasticParamConvertModule import ElasticConvJab as _ElasticModelConvJac2D
# 3d pybind modules
from pyElastic_iso_float_nl_3D import spaceInterpGpu_3D as device_gpu_3d
from pyElastic_iso_float_nl_3D import nonlinearPropElasticShotsGpu_3D
from pyElastic_iso_float_born_3D import BornElasticShotsGpu_3D
from dataCompModule_3D import ElasticDatComp_3D as _ElasticDatComp_3D
from elasticParamConvertModule_3D import ElasticConv_3D as _ElasticModelConv3D
from elasticParamConvertModule_3D import ElasticConvJab_3D as _ElasticModelConvJac3D


class ElasticIsotropic(wave_equation.WaveEquation):
  _BLOCK_SIZE = 16
  _N_MODEL_PARAMETERS = 3

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
               recording_components=None,
               subsampling=None,
               free_surface=False):
    if not recording_components:
      recording_components = self._DEFAULT_RECORDING_COMPONENTS
    self.recording_components = recording_components

    super().__init__(model, model_sampling, wavelet, d_t, src_locations,
                     rec_locations, gpus, model_padding, model_origins,
                     subsampling, free_surface)

  def _setup_wavefield_sampling_operator(self, _operator, recording_components,
                                         data_sep):
    # _ElasticDatComp_nD expects a string of comma seperated values
    recording_components = ",".join(recording_components)
    #make sampling opeartor
    wavefield_sampling_operator = self._wavefield_sampling_class(
        recording_components, data_sep)
    wavefield_sampling_operator = Operator.NonLinearOperator(
        wavefield_sampling_operator, wavefield_sampling_operator)

    return Operator.CombNonlinearOp(_operator, wavefield_sampling_operator)

    # self.data_sep = self._operator.range.clone()

  def _get_model_shape(self, model):
    return model.shape[1:]

  def _pad_model(self, model: np.ndarray, padding: typing.Tuple,
                 fat: int) -> np.ndarray:
    """
    Add padding to the model.

    Parameters:
        model (np.ndarray): 3D model to be padded
        padding (Tuple[int, int]): Tuple of integers representing the padding of the model in each axis

    Returns:
        np.ndarray: Padded 3D model
    """
    fat_pad = ((fat, fat),) * len(padding)
    return np.pad(np.pad(model, ((0, 0), *padding), mode='edge'),
                  ((0, 0), *fat_pad),
                  mode='edge')

  def _make_sep_vector_model_space(self, shape, origins, sampling):
    #need to reverse order for SepVector constructor
    ns = list(shape)[::-1]
    os = list(origins)[::-1]
    ds = list(sampling)[::-1]
    #add elastic parameter axis
    ns.append(self._N_MODEL_PARAMETERS)
    os.append(0.0)
    ds.append(1.0)
    return SepVector.getSepVector(ns=ns, os=os, ds=ds)

  def _get_velocities(self, model):
    return model[:2]

  def _setup_operators(self, data_sep, model_sep, sep_par, src_devices,
                       rec_devices, wavelet_nl_sep, wavelet_lin_sep):
    # setup model sampling operator
    nl_op = self._nl_elastic_param_conv_class(model_sep, 1)
    jac_op = self._jac_elastic_param_conv_class(model_sep, model_sep, 1)
    _param_convert_op = Operator.NonLinearOperator(nl_op, jac_op,
                                                   jac_op.setBackground)
    tmp_model = model_sep.clone()
    _param_convert_op.nl_op.forward(0, tmp_model, model_sep)

    # make and set gpu operator
    _operator = self._setup_nl_wave_op(data_sep, model_sep, sep_par,
                                       src_devices, rec_devices, wavelet_nl_sep)

    _operator = Operator.CombNonlinearOp(_param_convert_op, _operator)

    # append wavefield sampling to gpu operator
    _operator = self._setup_wavefield_sampling_operator(
        _operator, self.recording_components, data_sep)

    return _operator

  def _get_free_surface_pad_minus(self):
    return self._FAT


class ElasticIsotropic2D(ElasticIsotropic):
  _N_WFLD_COMPONENTS = 5
  _FAT = 4
  _nl_wave_pybind_class = nonlinearPropElasticShotsGpu
  _jac_wave_pybind_class = BornElasticShotsGpu
  _wavefield_sampling_class = _ElasticDatComp2D
  _nl_elastic_param_conv_class = _ElasticModelConv2D
  _jac_elastic_param_conv_class = _ElasticModelConvJac2D
  required_sep_params = [
      'nx', 'dx', 'nz', 'dz', 'xPadMinus', 'xPadPlus', 'zPadMinus', 'zPadPlus',
      'mod_par', 'dts', 'nts', 'fMax', 'sub', 'nExp', 'iGpu', 'blockSize',
      'fat', 'surfaceCondition'
  ]
  _FREE_SURFACE_AVAIL = True
  _DEFAULT_PADDING = 50
  _DEFAULT_RECORDING_COMPONENTS = [
      'vx',
      'vz',
      'sxx',
      'szz',
      'sxz',
  ]

  def _truncate_model(self, model):
    x_pad = self.fd_param['x_pad_minus'] + self._FAT
    x_pad_plus = self.fd_param['x_pad_plus'] + self._FAT
    z_pad = self.fd_param['z_pad_minus'] + self._FAT
    z_pad_plus = self.fd_param['z_pad_plus'] + self._FAT
    return model[:, x_pad:-x_pad_plus:, z_pad:-z_pad_plus:]

  def _setup_src_devices(self,
                         src_locations,
                         n_t,
                         interp_method='linear',
                         interp_n_filters=4):
    """
    src_locations - [n_src,(x_pos,z_pos)]
    """
    if self.model_sep is None:
      raise RuntimeError('self.model_sep must be set to set src devices')

    if 'n_src' in self.fd_param:
      assert self.fd_param['n_src'] == len(src_locations)

    staggered_grid_hypers = self._make_staggered_grid_hypers(
        self.fd_param['n_x'], self.fd_param['n_z'], self.fd_param['o_x'],
        self.fd_param['o_z'], self.fd_param['d_x'], self.fd_param['d_z'])

    src_devices_staggered_grids = [[], [], [], []]
    for src_location in src_locations:
      x_loc = src_location[0]
      z_loc = src_location[1]
      x_coord_sep = SepVector.getSepVector(ns=[1]).set(x_loc)
      z_coord_sep = SepVector.getSepVector(ns=[1]).set(z_loc)
      for src_devices_staggered_grid, staggered_grid_hyper in zip(
          src_devices_staggered_grids, staggered_grid_hypers):
        src_devices_staggered_grid.append(
            device_gpu_2d(z_coord_sep.getCpp(), x_coord_sep.getCpp(),
                          staggered_grid_hyper.getCpp(), int(n_t),
                          interp_method, interp_n_filters, 0, 0, 0))

    self.fd_param['n_src'] = len(src_locations)
    # self.src_devices = src_devices_staggered_grids
    return src_devices_staggered_grids

  def _setup_rec_devices(self,
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
      rec_locations = np.repeat(np.expand_dims(rec_locations, axis=0),
                                n_src,
                                axis=0)

    staggered_grid_hypers = self._make_staggered_grid_hypers(
        self.fd_param['n_x'], self.fd_param['n_z'], self.fd_param['o_x'],
        self.fd_param['o_z'], self.fd_param['d_x'], self.fd_param['d_z'])

    rec_devices_staggered_grids = [[], [], [], []]
    for rec_location in rec_locations:
      x_coord_sep = SepVector.getSepVector(ns=[n_rec])
      x_coord_sep.getNdArray()[:] = rec_location[:, 0]
      z_coord_sep = SepVector.getSepVector(ns=[n_rec])
      z_coord_sep.getNdArray()[:] = rec_location[:, 1]
      for rec_devices_staggered_grid, staggered_grid_hyper in zip(
          rec_devices_staggered_grids, staggered_grid_hypers):
        rec_devices_staggered_grid.append(
            device_gpu_2d(z_coord_sep.getCpp(), x_coord_sep.getCpp(),
                          staggered_grid_hyper.getCpp(), int(n_t),
                          interp_method, interp_n_filters, 0, 0, 0))

    self.fd_param['n_rec'] = n_rec
    # self.rec_devices = rec_devices_staggered_grids
    return rec_devices_staggered_grids

  def _setup_data(self, n_t, d_t):
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
            Hypercube.axis(n=self._N_WFLD_COMPONENTS, o=0.0, d=1),
            Hypercube.axis(n=self.fd_param['n_src'], o=0.0, d=1.0)
        ]))

    return data_sep

  def _make_sep_wavelets(self, wavelet, d_t):
    n_t = wavelet.shape[-1]
    wavelet_nl_sep = SepVector.getSepVector(
        Hypercube.hypercube(axes=[
            Hypercube.axis(n=n_t, o=0.0, d=d_t),
            Hypercube.axis(n=1),
            Hypercube.axis(n=self._N_WFLD_COMPONENTS),
            Hypercube.axis(n=1)
        ]))
    wavelet_nl_sep.getNdArray()[0, :, 0, :] = wavelet

    wavelet_lin_sep = SepVector.getSepVector(
        Hypercube.hypercube(axes=[
            Hypercube.axis(n=n_t, o=0.0, d=d_t),
            Hypercube.axis(n=self._N_WFLD_COMPONENTS),
            Hypercube.axis(n=1)
        ]))
    wavelet_lin_sep.getNdArray()[:] = wavelet
    wavelet_lin_sep = [wavelet_lin_sep.getCpp()]

    return wavelet_nl_sep, wavelet_lin_sep

  def _make_staggered_grid_hypers(self, n_x, n_z, o_x, o_z, d_x, d_z):
    z_axis = Hypercube.axis(n=n_z, o=o_z, d=d_z)
    z_axis_staggered = Hypercube.axis(n=n_z, o=o_z - 0.5 * d_z, d=d_z)

    x_axis = Hypercube.axis(n=n_x, o=o_x, d=d_x)
    x_axis_staggered = Hypercube.axis(n=n_x, o=o_x - 0.5 * d_x, d=d_x)

    param_axis = Hypercube.axis(n=self._N_MODEL_PARAMETERS, o=0.0, d=1)

    center_grid_hyper = Hypercube.hypercube(axes=[z_axis, x_axis, param_axis])
    x_staggered_grid_hyper = Hypercube.hypercube(
        axes=[z_axis, x_axis_staggered, param_axis])
    z_staggered_grid_hyper = Hypercube.hypercube(
        axes=[z_axis_staggered, x_axis, param_axis])
    xz_staggered_grid_hyper = Hypercube.hypercube(
        axes=[z_axis_staggered, x_axis_staggered, param_axis])

    return center_grid_hyper, x_staggered_grid_hyper, z_staggered_grid_hyper, xz_staggered_grid_hyper


class ElasticIsotropic3D(ElasticIsotropic):
  _N_WFLD_COMPONENTS = 9
  _FAT = 4
  required_sep_params = [
      'ny', 'dy', 'nx', 'dx', 'nz', 'dz', 'yPad', 'xPadMinus', 'xPadPlus',
      'zPadMinus', 'zPadPlus', 'mod_par', 'dts', 'nts', 'fMax', 'sub', 'nExp',
      'iGpu', 'blockSize', 'fat', 'freeSurface'
  ]
  _nl_wave_pybind_class = nonlinearPropElasticShotsGpu_3D
  _jac_wave_pybind_class = BornElasticShotsGpu_3D
  _wavefield_sampling_class = _ElasticDatComp_3D
  _nl_elastic_param_conv_class = _ElasticModelConv3D
  _jac_elastic_param_conv_class = _ElasticModelConvJac3D
  _FREE_SURFACE_AVAIL = False
  _DEFAULT_PADDING = 30
  _DEFAULT_RECORDING_COMPONENTS = [
      'vx', 'vy', 'vz', 'sxx', 'syy', 'szz', 'sxz', 'sxy', 'syz'
  ]

  def _truncate_model(self, model):
    y_pad = self.fd_param['y_pad'] + self._FAT
    x_pad = self.fd_param['x_pad_minus'] + self._FAT
    x_pad_plus = self.fd_param['x_pad_plus'] + self._FAT
    z_pad = self.fd_param['z_pad_minus'] + self._FAT
    z_pad_plus = self.fd_param['z_pad_plus'] + self._FAT
    return model[:, y_pad:-y_pad:, x_pad:-x_pad_plus:, z_pad:-z_pad_plus:]

  def _setup_src_devices(self, src_locations, n_t, interp_method='linear'):
    """
    src_locations - [n_src,(x_pos,z_pos)]
    """
    if self.model_sep is None:
      raise RuntimeError('self.model_sep must be set to set src devices')

    if 'n_src' in self.fd_param:
      assert self.fd_param['n_src'] == len(src_locations)

    staggered_grid_hypers = self._make_staggered_grid_hypers(
        self.fd_param['n_y'], self.fd_param['n_x'], self.fd_param['n_z'],
        self.fd_param['o_y'], self.fd_param['o_x'], self.fd_param['o_z'],
        self.fd_param['d_y'], self.fd_param['d_x'], self.fd_param['d_z'])

    sep_par = self._make_sep_par({
        'fat': self.fd_param['_FAT'],
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
            device_gpu_3d(z_coord_sep.getCpp(), x_coord_sep.getCpp(),
                          y_coord_sep.getCpp(), staggered_grid_hyper.getCpp(),
                          int(n_t), sep_par.param, 0, 0, 0, 0, interp_method,
                          1))

    self.fd_param['n_src'] = len(src_locations)
    # self.src_devices = src_devices_staggered_grids
    return src_devices_staggered_grids

  def _setup_rec_devices(self,
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

    staggered_grid_hypers = self._make_staggered_grid_hypers(
        self.fd_param['n_y'], self.fd_param['n_x'], self.fd_param['n_z'],
        self.fd_param['o_y'], self.fd_param['o_x'], self.fd_param['o_z'],
        self.fd_param['d_y'], self.fd_param['d_x'], self.fd_param['d_z'])

    sep_par = self._make_sep_par({
        'fat': self.fd_param['_FAT'],
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
            device_gpu_3d(z_coord_sep.getCpp(), x_coord_sep.getCpp(),
                          y_coord_sep.getCpp(), staggered_grid_hyper.getCpp(),
                          int(n_t), sep_par.param, 0, 0, 0, 0, interp_method,
                          1))

    self.fd_param['n_rec'] = n_rec
    # self.rec_devices = rec_devices_staggered_grids
    return rec_devices_staggered_grids

  def _setup_data(self, n_t, d_t):
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
            Hypercube.axis(n=self._N_WFLD_COMPONENTS, o=0.0, d=1),
            Hypercube.axis(n=self.fd_param['n_src'], o=0.0, d=1.0)
        ]))

    return data_sep

  def _make_sep_wavelets(self, wavelet, d_t):
    n_t = wavelet.shape[-1]
    wavelet_nl_sep = SepVector.getSepVector(
        Hypercube.hypercube(axes=[
            Hypercube.axis(n=n_t, o=0.0, d=d_t),
            Hypercube.axis(n=1),
            Hypercube.axis(n=self._N_WFLD_COMPONENTS),
            Hypercube.axis(n=1)
        ]))
    wavelet_nl_sep.getNdArray().flat[:] = wavelet

    wavelet_lin_sep = SepVector.getSepVector(
        Hypercube.hypercube(axes=[
            Hypercube.axis(n=n_t, o=0.0, d=d_t),
            Hypercube.axis(n=1),
            Hypercube.axis(n=self._N_WFLD_COMPONENTS)
        ]))
    wavelet_lin_sep.getNdArray().flat[:] = wavelet
    wavelet_lin_sep = [wavelet_lin_sep.getCpp()]

    return wavelet_nl_sep, wavelet_lin_sep

  def _make_staggered_grid_hypers(self, n_y, n_x, n_z, o_y, o_x, o_z, d_y, d_x,
                                  d_z):
    z_axis = Hypercube.axis(n=n_z, o=o_z, d=d_z)
    z_axis_staggered = Hypercube.axis(n=n_z, o=o_z - 0.5 * d_z, d=d_z)

    x_axis = Hypercube.axis(n=n_x, o=o_x, d=d_x)
    x_axis_staggered = Hypercube.axis(n=n_x, o=o_x - 0.5 * d_x, d=d_x)

    y_axis = Hypercube.axis(n=n_y, o=o_y, d=d_y)
    y_axis_staggered = Hypercube.axis(n=n_y, o=o_y - 0.5 * d_y, d=d_y)

    param_axis = Hypercube.axis(n=self._N_MODEL_PARAMETERS, o=0.0, d=1)

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


def convert_to_lame(model):
  """convert elastic model to rho, lame, and mu

  Args:
      model (nd numpy array): (n_params, ...)
  """

  converted_model = np.zeros_like(model)

  #VpVsRho to RhoLameMu (m/s|m/s|kg/m3 -> kg/m3|Pa|Pa)
  converted_model[0] = model[2]  #rho
  converted_model[1] = model[2] * (
      model[0] * model[0] - 2.0 * model[1] * model[1])  #lame
  converted_model[2] = model[2] * model[1] * model[1]  #mu

  return converted_model


def convert_to_vel(model):
  converted_model = np.zeros_like(model)
  #RhoLameMu to VpVsRho (kg/m3|Pa|Pa -> m/s|m/s|kg/m3)
  converted_model[0] += np.sqrt(np.divide((model[1] + 2 * model[2]),
                                          model[0]))  #vp
  converted_model[1] += np.sqrt(np.divide(model[2], model[0]))  #vs
  converted_model[2] += model[0]  #rho

  return converted_model