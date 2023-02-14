"""Defines the 2D and 3D AcousticIsotropic wave equation classes.

The AcousticIsotropic2D and AcousticIsotropic3D inherit from the abstract
AcousticIsotropic class. AcousticIsotropic2D and AcousticIsotropic3D can model
the acoustic, isotropic, constant-density wave equation in two and three
dimensions, respectively. With a pressure-wave velocity model, source wavelet,
source positions, and receiver positions, a user can forward model the wave
wave equation and sample at receiver locations. Pybind11 is used to wrap C++
code which then uses CUDA kernels to execute finite difference operations.
Current implementation parallelizes shots over gpus. Absorbing boundaries are
used to handle edge effects.

Typical usage example:
  #### 2D ##### 
  from pyseis.wave_equations import acoustic_isotropic
  
  acoustic_2d = acoustic_isotropic.AcousticIsotropic2D(
    vp_model_nd_array,
    (d_x, d_z),
    wavelet_nd_array,
    d_t,
    src_locations_nd_array,
    rec_locations_nd_array,
    gpus=[0,1,2,3])
  data = acoustic_2d.forward(vp_model_nd_array)
"""
import numpy as np
from math import ceil
from typing import Tuple, List

import genericIO
import Hypercube
import SepVector
import pyOperator as Operator
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
  """Abstract acoustic wave equation solver class.

  Contains all methods and variables common to and acoustic 2D and 3D
  wave equation solvers. See wave_equation.WaveEquation for __init__
  """
  _BLOCK_SIZE = 16
  _FREE_SURFACE_AVAIL = True

  def _get_model_shape(self, model: np.ndarray) -> Tuple[int, ...]:
    """Helper function to get shape of model space.

    Args:
        model (np.ndarray): model space

    Returns:
        Tuple[int, ...]: shape of model space
    """
    return model.shape

  def _pad_model(self, model: np.ndarray, padding: Tuple,
                 fat: int) -> np.ndarray:
    """Helper to pad earth models before wave prop.

    Args:
        model (np.ndarray):  earth model without padding (2D or 3D). For the
          acoustic wave equation, this must be pressure wave velocity in (m/s). If
          elastic, the first axis must be the three elastic parameters (pressure
        wave velocity (m/s), shear wave velocity (m/s), and density (kg/m^3)).
          - a 3D acoustic model will have shape (n_y, n_x, n_z)
          - a 2D acoustic model will have shape (n_x, n_z)
        padding (Tuple[Tuple[int, int], ...]): the padding (minus, plus)
          tuples that was applied to each axis.
        fat (int): the additional boundary to be added around the padding. Used
          to make laplacian computation more easily self adjoint.

    Returns:
        np.ndarray: padded earth model
    """
    fat_pad = ((fat, fat),) * len(padding)
    return np.pad(np.pad(model, padding, mode='edge'), fat_pad, mode='constant')

  def _make_sep_vector_model_space(
      self, shape: Tuple[int, ...], origins: Tuple[float, ...],
      sampling: Tuple[float, ...]) -> SepVector.floatVector:
    """Helper to make empty SepVector moddel space

    Args:
        shape (Tuple[int, ...]): The spatial shape of the unpadded model.
          In the elastic case this should not include the elastic parameter axis.
        origins (Tuple[int, ...], optional): Origin of each axis of the earth
          model. Just like the sampling arguement, must be same length as
          the number of dimensions of model. If None, the origins are assumed to
          be 0.0. Defaults to None.
        sampling (Tuple[float, ...]): spatial sampling of provided earth
          model axes. Must be same length as number of dimensions of model.
            - (d_y, d_x, d_z) with 3D models
            - (d_x, d_z) with 2D models
      
    Raises:
        NotImplementedError: if not implemented by child class

    Returns:
        SepVector.floatVector: empty model space
    """
    #need to reverse order for SepVector constructor
    ns = list(shape)[::-1]
    os = list(origins)[::-1]
    ds = list(sampling)[::-1]
    return SepVector.getSepVector(ns=ns, os=os, ds=ds)

  def _get_velocities(self, model: np.ndarray) -> np.ndarray:
    """Helper function to return only the velocity components of a model.
    
    In acoustic case just returns passed model.

    Args:
        model (np.ndarray): earth model

    Returns:
        np.ndarray: velocity components of model
    """
    return model

  def _setup_data(self,
                  n_t: int,
                  d_t: float,
                  data: np.ndarray = None) -> SepVector.floatVector:
    """Helper to setup the SepVector data space.
    
    Args:
        n_t (int): number of time samples in the data space
        d_t (float): time sampling in the data space (s)
        data (np.ndarray, optional): Data to fill SepVector. If None, filled
          with zeros. Defaults to None.

    Returns:
        SepVector.floatVector: data space
    """
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

  def _setup_nonlinear_operator(
      self, data_sep: SepVector.floatVector, model_sep: SepVector.floatVector,
      sep_par: genericIO.io, src_devices: List, rec_devices: List,
      wavelet_nl_sep: SepVector.floatVector) -> Operator.Operator:
    """Helper function to set up the nonlinear operator
    
    nonlinear: f(m) = d
    jacobian: Fm = d 

    Args:
        data_sep (SepVector.floatVector): data space in SepVector format
        model_sep (SepVector.floatVector): model space in SepVector format
        sep_par (genericIO.io): io object containing all fd_params
        src_devices (List): list of pybind11 source device classes
        rec_devices (List): list of pybind11 receiver device classes
        wavelet_nl_sep (SepVector.floatVector): the wavelet for nonlinear wave
          prop in SepVector format
        wavelet_lin_sep (SepVector.floatVector): the wavelets for linear wave
          prop in SepVector format

    Returns:
        Operator.Operator: all combined operators
    """
    # make and set gpu operator
    return self._setup_nl_wave_op(data_sep, model_sep, sep_par, src_devices,
                                  rec_devices, wavelet_nl_sep)

  def _setup_jacobian_operator(
      self, data_sep: SepVector.floatVector, model_sep: SepVector.floatVector,
      sep_par: genericIO.io, src_devices: List, rec_devices: List,
      wavelet_lin_sep: SepVector.floatVector) -> Operator.Operator:
    """Helper function to set up the jacobian operator
    
    Args:
        data_sep (SepVector.floatVector): data space in SepVector format
        model_sep (SepVector.floatVector): model space in SepVector format
        sep_par (genericIO.io): io object containing all fd_params
        src_devices (List): list of pybind11 source device classes
        rec_devices (List): list of pybind11 receiver device classes
        wavelet_lin_sep (SepVector.floatVector): the wavelets for linear wave prop
        
    Returns:
        Operator.Operator: all combined operators
    """
    return self._setup_jac_wave_op(data_sep, model_sep, sep_par, src_devices,
                                   rec_devices, wavelet_lin_sep)

  def _combine_nl_and_jac_operators(self) -> Operator.NonLinearOperator:
    """Combine the nonlinear and jacobian operators into one nonlinear operator

    Returns:
        Operator.NonlinearOperator: a nonlinear operator that combines the
          nonlienar and jacobian operators
    """
    return Operator.NonLinearOperator(self._nl_operator, self._jac_operator,
                                      self._jac_operator.set_background)

  def _get_free_surface_pad_minus(self) -> int:
    """Abstract helper function to get the amount of padding to add to the free surface

    Raises:
        NotImplementedError: if not implemented by child class

    Returns:
        int: number of cells to pad
    """
    return 0


class AcousticIsotropic2D(AcousticIsotropic):
  """2D acoustic, isotropic wave equation solver class.

  Wavefield paramterized by pressure. Expects model space to be paramterized by
  Vp. Free surface condition is avialable.
  
  Typical usage example (see examples directory for more):
    from pyseis.wave_equations import acoustic_isotropic
    
    acoustic_2d = acoustic_isotropic.AcousticIsotropic2D(
      vp_vs_rho_model_nd_array,
      (d_y, d_x, d_z),
      wavelet_nd_array,
      d_t,
      src_locations_nd_array,
      rec_locations_nd_array,
      gpus=[0,1,2,4])
    data = acoustic_2d.forward(vp_vs_rho_model_nd_array)
  """
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

  """
  src_locations - [n_src,(x_pos,z_pos)]
  """

  def _setup_src_devices(self, src_locations: np.ndarray, n_t: float) -> List:
    """Helper function to setup source devices needed for wave prop.

    Args:
        src_locations (np.ndarray): location of each source device. Should have
          shape (n_src, 2) where 2 is the number of dimensions, x and z.
        n_t (float): number of time steps in the wavelet/data space

    Raises:
        NotImplementedError: if not implemented by child class

    Returns:
        List: list of pybind11 device classes
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

  def _setup_rec_devices(self, rec_locations: np.ndarray, n_t: float) -> List:
    """Helper function to setup receiver devices needed for wave prop.

    Args:
        rec_locations (np.ndarray): location of each source device. Should have
          shape (n_rec, n_dim) or (n_src, n_rec, 2) where 2 is the number of
          dimensions, x and z. The latter allows for different receiver
          positions for each shot (aka streamer geometry).
        n_t (float): number of time steps in the wavelet/data space

    Raises:
        NotImplementedError: if not implemented by child class

    Returns:
        List: list of pybind11 device classes
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

  def _make_sep_wavelets(
      self, wavelet: np.ndarray,
      d_t: float) -> Tuple[SepVector.floatVector, SepVector.floatVector]:
    """Helper function to make SepVector wavelets needed for wave prop

    The acoustic 2D prop uses a 3d SepVector for nonlinear prop and a list
    containing a single, 2d SepVector for linear prop. THIS IS FOR BACKWARDS
    COMPATIBILITY WITH PYBIND11 C++ CODE.
    
    Args:
      wavelet (np.ndarray): : source signature of wave equation. Also defines
        the recording duration of the seismic data space.
        - In 2D and 3D acoustic case, wavelet will have shape (n_t)
          nine source components (Fy, Fx, Fz, syy, sxx, szz, syx, syz, sxz)
      d_t (float): temporal sampling rate (s)

    Returns:
        Tuple: 3d SepVector.floatVector (wavelet for 2d nonlinear acoustic prop)
          and list with a single 2d SepVector.floatVector (wavelet acosutic
          linear wave prop).
    """
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
  """3D acoustic, isotropic wave equation solver class.

  Wavefield paramterized by pressure. Expects model space to be paramterized by
  Vp. Free surface condition is avialable.
  
  Typical usage example (see examples directory for more):
    from pyseis.wave_equations import acoustic_isotropic
    
    acoustic_3d = acoustic_isotropic.AcousticIsotropic3D(
      vp_vs_rho_model_nd_array,
      (d_y, d_x, d_z),
      wavelet_nd_array,
      d_t,
      src_locations_nd_array,
      rec_locations_nd_array,
      gpus=[0,1,2,4])
    data = acoustic_3d.forward(vp_vs_rho_model_nd_array)
  """
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
    """Helper function to setup source devices needed for wave prop.

    Args:
        src_locations (np.ndarray): location of each source device. Should have
          shape (n_src, 3) where 3 is the number of dimensions, x, y, and z.
        n_t (float): number of time steps in the wavelet/data space

    Raises:
        NotImplementedError: if not implemented by child class

    Returns:
        List: list of pybind11 device classes
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
    """Helper function to setup receiver devices needed for wave prop.

    Args:
        rec_locations (np.ndarray): location of each source device. Should have
          shape (n_rec, n_dim) or (n_src, n_rec, 3) where 3 is the number of
          dimensions, x, y,  and z. The latter allows for different receiver
          positions for each shot (aka streamer geometry).
        n_t (float): number of time steps in the wavelet/data space

    Raises:
        NotImplementedError: if not implemented by child class

    Returns:
        List: list of pybind11 device classes
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

  def _make_sep_wavelets(
      self, wavelet: np.ndarray,
      d_t: float) -> Tuple[SepVector.floatVector, SepVector.floatVector]:
    """Helper function to make SepVector wavelets needed for wave prop

    The acoustic 3D prop uses a 2d SepVector for nonlinear and linear prop. THIS
    IS FOR BACKWARDS COMPATIBILITY WITH PYBIND11 C++ CODE.
    
    Args:
      wavelet (np.ndarray): : source signature of wave equation. Also defines
        the recording duration of the seismic data space.
        - In 2D and 3D acoustic case, wavelet will have shape (n_t)
        - In 2D, elastic case, wavelet will have shape (5, n_t) describing the
          five source components (Fx, Fz, sxx, szz, sxz)
        - In 3D, elastic case, wavelet will have shape (9, n_t) describing the
          nine source components (Fy, Fx, Fz, syy, sxx, szz, syx, syz, sxz)
      d_t (float): temporal sampling rate (s)


    Returns:
      Tuple: 2d SepVector.floatVector (wavelet for 3d nonlinear acoustic prop)
          and 2d SepVector.floatVector (wavelet acoustic linear wave prop).
    """
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
