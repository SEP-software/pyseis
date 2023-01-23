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
      gpus=[0,1,2,3])
    data = acoustic_2d.forward(vp_model_half_space)
"""
import numpy as np
from math import ceil
import typing

import Hypercube
import SepVector
from pyseis.wave_equations import wave_equation
# 2d pybind11 modules
from pyAcoustic_iso_float_nl import deviceGpu as device_gpu_2d
from pyAcoustic_iso_float_nl import nonlinearPropShotsGpu, ostream_redirect
from pyAcoustic_iso_float_born import BornShotsGpu
# 3d pybind11 modules
from pyAcoustic_iso_float_nl_3D import deviceGpu_3D as device_gpu_3d
from pyAcoustic_iso_float_nl_3D import nonlinearPropShotsGpu_3D
from pyAcoustic_iso_float_Born_3D import BornShotsGpu_3D


class AcousticIsotropic(wave_equation.WaveEquation):
  _BLOCK_SIZE = 16
  _FREE_SURFACE_AVAIL = True

  def _get_model_shape(self, model):
    return model.shape

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
    return np.pad(np.pad(model, padding, mode='edge'), fat_pad, mode='constant')

  def _make_sep_vector_model_space(self, shape, origins, sampling):
    #need to reverse order for SepVector constructor
    ns = list(shape)[::-1]
    os = list(origins)[::-1]
    ds = list(sampling)[::-1]
    return SepVector.getSepVector(ns=ns, os=os, ds=ds)

  def _get_velocities(self, model):
    return model

  def _setup_operators(self,
                       data_sep,
                       model_sep,
                       sep_par,
                       src_devices,
                       rec_devices,
                       wavelet_nl_sep,
                       wavelet_lin_sep,
                       recording_components=None):
    # make and set gpu operator
    return self._setup_nl_wave_op(data_sep, model_sep, sep_par, src_devices,
                                  rec_devices, wavelet_nl_sep)

  def _get_free_surface_pad_minus(self):
    return 0


class AcousticIsotropic2D(AcousticIsotropic):
  _FAT = 5
  required_sep_params = [
      'nx', 'dx', 'nz', 'dz', 'xPadMinus', 'xPadPlus', 'zPadMinus', 'zPadPlus',
      'dts', 'nts', 'fMax', 'sub', 'nShot', 'iGpu', 'blockSize', 'fat', 'ginsu',
      'freeSurface'
  ]
  _nl_wave_pybind_class = nonlinearPropShotsGpu
  _jac_wave_pybind_class = BornShotsGpu
  _DEFAULT_PADDING = 50

  def _truncate_model(self, model):
    x_pad = self.fd_param['x_pad_minus'] + self._FAT
    x_pad_plus = self.fd_param['x_pad_plus'] + self._FAT
    z_pad = self.fd_param['z_pad_minus'] + self._FAT
    z_pad_plus = self.fd_param['z_pad_plus'] + self._FAT
    return model[x_pad:-x_pad_plus:, z_pad:-z_pad_plus:]

  def _setup_src_devices(self, src_locations, n_t):
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
    # self.src_devices = source_devices
    return source_devices

  def _setup_rec_devices(self, rec_locations, n_t):
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
    # self.rec_devices = rec_devices
    return rec_devices

  def _setup_data(self, n_t, d_t, data=None):
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

    return data_sep

  def _make_sep_wavelets(self, wavelet, d_t):
    '''for backwards combatibility with the c++ codebase, the nonlinear and the 
    linear operators expect different shaped wavelet
    '''
    #the 2d acoustic code uses a 3d sepvector for nonlinear prop and a list
    #containing a single, 2d sepvector for linear prop.
    n_t = wavelet.shape[-1]
    #make wavelet for nonlinear wave prop
    wavelet_nl_sep = SepVector.getSepVector(
        Hypercube.hypercube(axes=[
            Hypercube.axis(n=n_t, o=0.0, d=d_t),
            Hypercube.axis(n=1),
            Hypercube.axis(n=1)
        ]))
    wavelet_nl_sep.getNdArray()[:] = wavelet
    #make wavelet for linear wave prop
    wavelet_lin_sep = SepVector.getSepVector(
        Hypercube.hypercube(axes=[
            Hypercube.axis(n=n_t, o=0.0, d=d_t),
            Hypercube.axis(n=1),
        ]))
    wavelet_lin_sep.getNdArray()[:] = wavelet
    wavelet_lin_sep = [wavelet_lin_sep.getCpp()]

    return wavelet_nl_sep, wavelet_lin_sep


class AcousticIsotropic3D(AcousticIsotropic):
  _FAT = 4
  required_sep_params = [
      'nx', 'dx', 'nz', 'dz', 'ny', 'dy', 'xPadMinus', 'xPadPlus', 'zPadMinus',
      'zPadPlus', 'yPad', 'dts', 'nts', 'fMax', 'sub', 'nShot', 'iGpu',
      'blockSize', 'fat', 'ginsu', 'freeSurface'
  ]
  _nl_wave_pybind_class = nonlinearPropShotsGpu_3D
  _jac_wave_pybind_class = BornShotsGpu_3D
  _DEFAULT_PADDING = 30

  def _truncate_model(self, model):
    y_pad = self.fd_param['y_pad'] + self._FAT
    x_pad = self.fd_param['x_pad_minus'] + self._FAT
    x_pad_plus = self.fd_param['x_pad_plus'] + self._FAT
    z_pad = self.fd_param['z_pad_minus'] + self._FAT
    z_pad_plus = self.fd_param['z_pad_plus'] + self._FAT
    return model[y_pad:-y_pad:, x_pad:-x_pad_plus:, z_pad:-z_pad_plus:]

  def _setup_src_devices(self, src_locations, n_t):
    """
    src_locations - [n_src,(y_pos,x_pos,z_pos)]
    """
    if self.model_sep is None:
      raise RuntimeError('self.model_sep must be set to set src devices')

    if 'n_src' in self.fd_param:
      assert self.fd_param['n_src'] == len(src_locations)

    sep_par = self._make_sep_par({
        'fat': self.fd_param['_FAT'],
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
    # self.src_devices = source_devices
    return source_devices

  def _setup_rec_devices(self, rec_locations, n_t):
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
      sep_par = self._make_sep_par({
          'fat': self.fd_param['_FAT'],
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
    # self.rec_devices = rec_devices
    return rec_devices

  def _setup_data(self, n_t, d_t, data=None):
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

    return data_sep

  def _make_sep_wavelets(self, wavelet, d_t):
    n_t = wavelet.shape[-1]
    wavelet_nl_sep = SepVector.getSepVector(
        Hypercube.hypercube(
            axes=[Hypercube.axis(n=n_t, o=0.0, d=d_t),
                  Hypercube.axis(n=1)]))
    wavelet_nl_sep.getNdArray()[:] = wavelet

    #the 3d acoustic code uses the same wavelet object for nonlinear and linear
    wavelet_lin_sep = wavelet_nl_sep.clone()
    wavelet_lin_sep = wavelet_lin_sep.getCpp()

    return wavelet_nl_sep, wavelet_lin_sep
