"""Defines the 2D and 3D ElasticIsotropic wave equation classes.

The ElasticIsotropic2D and ElasticIsotropic3D inherit from the abstract
ElasticIsotropic class. ElasticIsotropic2D and ElasticIsotropic3D can model
the elastic, isotropic wave equation in two and three dimensions, respectively,
using velocity/stress wavefields and a staggered-grid implementation. With an
elastic earth model parameterized by (v_p, v_s, and rho), a source wavelet,
source positions, and receiver positions, a user can forward model the wave wave
equation and sample at receiver locations. Pybind11 is used to wrap C++ code
which then uses CUDA kernels to execute finite difference operations. Current
implementation parallelizes shots over gpus. Absorbing boundaries are used to
handle edge effects. Domain decompisition not vailable. Free surface available
in 2D.

Typical usage example:
  #### 2D ##### 
  from pyseis.wave_equations import elastic_isotropic
  
  elastic_2d = elastic_isotropic.ElasticIsotropic2D(
    vp_vs_rho_model_nd_array,
    (d_x, d_z),
    wavelet_nd_array,
    d_t,
    src_locations_nd_array,
    rec_locations_nd_array,
    gpus=[0,1,2,4])
  data = elastic_2d.forward(vp_vs_rho_model_nd_array)
"""
import numpy as np
from math import ceil
import abc
from typing import Tuple, List

import genericIO
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
  """Abstract elastic wave equation solver class.

  Contains all methods and variables common to and elastic 2D and 3D
  wave equation solvers. See wave_equation.WaveEquation for __init__
  """
  _BLOCK_SIZE = 16
  _N_MODEL_PARAMETERS = 3

  def __init__(self,
               model: np.ndarray,
               model_sampling: Tuple[float, ...],
               wavelet: np.ndarray,
               d_t: float,
               src_locations: np.ndarray,
               rec_locations: np.ndarray,
               gpus: List[int],
               model_padding: Tuple[int, ...] = None,
               model_origins: Tuple[int, ...] = None,
               recording_components: List[str] = None,
               subsampling: int = None,
               free_surface: bool = False) -> None:
    """Constructor for elastic wave equation solvers.

    Args:
      model (np.ndarray): earth model without padding (2D or 3D). For the
        acoustic wave equation, this must be pressure wave velocity in (m/s). If
        elastic, the first axis must be the three elastic parameters (pressure
        wave velocity (m/s), shear wave velocity (m/s), and density (kg/m^3)).
          - a 3D elastic model will have shape (3, n_y, n_x, n_z)
          - a 2D acoustic model will have shape (n_x, n_z)
      model_sampling (Tuple[float, ...]): spatial sampling of provided earth
        model axes. Must be same length as number of dimensions of model.
          - (d_y, d_x, d_z) with 3D models
          - (d_x, d_z) with 2D models
      wavelet (np.ndarray): : source signature of wave equation. Also defines
        the recording duration of the seismic data space.
        - In 2D and 3D acoustic case, wavelet will have shape (n_t)
        - In 2D, elastic case, wavelet will have shape (5, n_t) describing the
          five source components (Fx, Fz, sxx, szz, sxz)
        - In 3D, elastic case, wavelet will have shape (9, n_t) describing the
          nine source components (Fy, Fx, Fz, syy, sxx, szz, syx, syz, sxz)
      d_t (float): temporal sampling rate (s)
      src_locations (np.ndarray): coordinates of each seismic source to
        propagate. 
          - In 2D has shape (n_src, 2) where two values on fast axis describe
            the x and y positions, respectively.
          - In 3D has shape (n_src, 3) where the fast axis describes the y, x,
            and z positions, respectively.
      rec_locations (np.ndarray): receiver locations of each src. The number
        of receivers must be the same for each shot, but the position of the
        receivers can change.
          - In the 2D case, can either have shape (n_rec, 2) if the receiver
            positions are constant or (n_src,n_rec,2) if the receiver
            positions change with each shot.
          - In the 3D case, can either have shape (n_rec, 3) if the receiver
            positions are constant or (n_src,n_rec,3) if the receiver
            positions change with each shot.
      gpus (List[int]): the gpu devices to use for wave propagation.
      model_padding (Tuple[int, ...], optional): Thickness of absorbing boundary,
        defined in number of samples, to be added to each axis of the earth
        model. Just like the model_sampling arguement, must be same length as
        the number of dimensions of model. If None, the padding on each axis
        is 50 samples in 2D and 30 samples in 3D. Defaults to None.
      model_origins (Tuple[int, ...], optional): Origin of each axis of the earth
        model. Just like the model_sampling arguement, must be same length as
        the number of dimensions of model. If None, the origins are assumed to
        be 0.0. Defaults to None.
      recording_components (List[str, ...]), optional): A list of the wavefield
       components to record at receiver locations. Options are any number of
       those listed below. If None, all veclocity and stress tensor components
       are recorded. Default is None.
          In 2D:
          - 'vx' - particle velocity in the x direction (default)
          - 'vz' - particle velocity in the z direction (default)
          - 'sxx' - xx component of the stress tensor (default)
          - 'szz' - zz component of the stress tensor (default)
          - 'sxz' - xz component of the stress tensor (default)
          - 'p' - pressure := 0.5 * (sxx + szz)
          In 3D:
          - 'vx' - particle velocity in the x direction (default)
          - 'vy' - particle velocity in the y direction (default)
          - 'vz' - particle velocity in the z direction (default)
          - 'sxx' - xx component of the stress tensor (default)
          - 'syy' - yy component of the stress tensor (default)
          - 'szz' - zz component of the stress tensor (default)
          - 'sxz' - xz component of the stress tensor (default)
          - 'sxy' - xy component of the stress tensor (default)
          - 'syx' - yz component of the stress tensor (default)
          - 'p' - pressure := 0.33333 * (sxx + syy + szz)
      subsampling (Tuple[int, ...], optional): Exact subsampling multiple of d_t to
        use during finite difference wave propagation. If None, this is
        calculated automatically by the wave equation solver to obey the
        Courant condition. Defaults to None.
          - subsampling = 3 implies that the finite difference code will take
            time steps at intervals of d_t / 3.
      free_surface (bool, optional): whether of not to use a free surface
        condition. Defaults to False.

    Raises:
        NotImplementedError: if free surface condition is requested but is not
          available.
    """
    if not recording_components:
      recording_components = self._DEFAULT_RECORDING_COMPONENTS
    self.recording_components = recording_components

    super().__init__(model, model_sampling, wavelet, d_t, src_locations,
                     rec_locations, gpus, model_padding, model_origins,
                     subsampling, free_surface)

  def _setup_wavefield_sampling_operator(self, recording_components, data_sep):
    # _ElasticDatComp_nD expects a string of comma seperated values
    recording_components = ",".join(recording_components)
    #make sampling opeartor
    return self._wavefield_sampling_class(recording_components, data_sep)

  def _get_model_shape(self, model: np.ndarray) -> Tuple[int, ...]:
    """Helper function to get shape of model space.
    
    Disregard elastic parameter axis.

    Args:
        model (np.ndarray): model space

    Returns:
        Tuple[int, ...]: shape of model space
    """
    return model.shape[1:]

  def _pad_model(self, model: np.ndarray, padding: Tuple[Tuple[int, int], ...],
                 fat: int) -> np.ndarray:
    """Helper to pad earth models before wave prop.
    
    Args:
        model (np.ndarray):  earth model without padding (2D or 3D). For the
        acoustic wave equation, this must be pressure wave velocity in (m/s). If
        elastic, the first axis must be the three elastic parameters (pressure
        wave velocity (m/s), shear wave velocity (m/s), and density (kg/m^3)).
          - a 3D elastic model will have shape (3, n_y, n_x, n_z)
          - a 2D elastic model will have shape (3, n_x, n_z)
        padding (Tuple[Tuple[int, int], ...]): the padding (minus, plus)
          tuples that was applied to each axis.
        fat (int): the additional boundary to be added around the padding. Used
          to make laplacian computation more easily self adjoint.

    Returns:
        np.ndarray: padded elastic earth model
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

  def _get_velocities(self, model: np.ndarray) -> np.ndarray:
    """Helper function to return only the velocity components of a model.
    
    Returns only Vp and Vs

    Args:
        model (np.ndarray): earth model

    Returns:
        np.ndarray: velocity components of model
    """
    return model[:2]

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
    # setup model sampling operator
    self._param_convert_nl_op = self._nl_elastic_param_conv_class(model_sep, 1)
    tmp_model = model_sep.clone()
    self._param_convert_nl_op.forward(0, model_sep, tmp_model)

    # make and set gpu operator
    self._wave_nl_op = self._setup_nl_wave_op(data_sep, tmp_model, sep_par,
                                              src_devices, rec_devices,
                                              wavelet_nl_sep)

    # make wavefield sampling to gpu operator
    self._wavefield_sampling_nl_op = self._setup_wavefield_sampling_operator(
        self.recording_components, data_sep)

    # append operators
    _nl_op = Operator.ChainOperator(self._param_convert_nl_op, self._wave_nl_op)
    _nl_op = Operator.ChainOperator(_nl_op, self._wavefield_sampling_nl_op)

    return _nl_op

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
    # setup model sampling operator
    self._param_convert_jac_op = self._jac_elastic_param_conv_class(
        model_sep, model_sep, 1)
    tmp_model = model_sep.clone()
    self._param_convert_nl_op.forward(0, model_sep, tmp_model)

    # make and set gpu operator
    self._wave_jac_op = self._setup_jac_wave_op(data_sep, tmp_model, sep_par,
                                                src_devices, rec_devices,
                                                wavelet_lin_sep)

    # make wavefield sampling to gpu operator
    self._wavefield_sampling_jac_op = self._setup_wavefield_sampling_operator(
        self.recording_components, data_sep)

    # append operators
    _jac_op = Operator.ChainOperator(self._param_convert_jac_op,
                                     self._wave_jac_op)
    _jac_op = Operator.ChainOperator(_jac_op, self._wavefield_sampling_jac_op)

    return _jac_op

  def _combine_nl_and_jac_operators(self) -> Operator.NonLinearOperator:
    """Combine the nonlinear and jacobian operators into one nonlinear operator

    Returns:
        Operator.NonlinearOperator: a nonlinear operator that combines the
          nonlienar and jacobian operators
    """
    # make nonlinear parameter conversion op
    _param_convert_comb_op = Operator.NonLinearOperator(
        self._param_convert_nl_op, self._param_convert_jac_op,
        self._param_convert_jac_op.setBackground)

    # make and set gpu operator
    _wave_comb_op = Operator.NonLinearOperator(self._wave_nl_op,
                                               self._wave_jac_op,
                                               self._wave_jac_op.set_background)

    # append wavefield sampling to gpu operator
    _wavefield_sampling_comb_op = Operator.NonLinearOperator(
        self._wavefield_sampling_nl_op, self._wavefield_sampling_jac_op)

    # append operators
    _comb_op = Operator.CombNonlinearOp(_param_convert_comb_op, _wave_comb_op)
    _comb_op = Operator.CombNonlinearOp(_comb_op, _wavefield_sampling_comb_op)

    return _comb_op

  def _get_free_surface_pad_minus(self) -> int:
    """Abstract helper function to get the amount of padding to add to the free surface

    Raises:
        NotImplementedError: if not implemented by child class

    Returns:
        int: number of cells to pad
    """
    return self._FAT


class ElasticIsotropic2D(ElasticIsotropic):
  """2D elastic, isotropic wave equation solver class.

  Velocity/stress formulation with a staggered grid. Expects model space to be
  paramterized by Vp, Vs, and Rho. Free surface condition is avialable.
  
  Typical usage example:
    from pyseis.wave_equations import elastic_isotropic
    
    elastic_2d = elastic_isotropic.ElasticIsotropic2D(
      vp_vs_rho_model_nd_array,
      (d_x, d_z),
      wavelet_nd_array,
      d_t,
      src_locations_nd_array,
      rec_locations_nd_array,
      gpus=[0,1,2,4],
      recording_components=['vx','vz'])
    data = elastic_2d.forward(vp_vs_rho_model_nd_array)
  """
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

  def _make_sep_wavelets(self, wavelet: np.ndarray, d_t: float) -> Tuple:
    """Helper function to make SepVector wavelets needed for wave prop
    
    The elastic 2D prop uses a 4d SepVector for nonlinear prop and a list
    containing a single, 3d SepVector for linear prop. THIS IS FOR BACKWARDS
    COMPATIBILITY WITH PYBIND11 C++ CODE.

    Args:
      wavelet (np.ndarray): : source signature of wave equation. Also defines
        the recording duration of the seismic data space.
        - In 2D, elastic case, wavelet will have shape (5, n_t) describing the
          five source components (Fx, Fz, sxx, szz, sxz)
        - In 3D, elastic case, wavelet will have shape (9, n_t) describing the
          nine source components (Fy, Fx, Fz, syy, sxx, szz, syx, syz, sxz)
      d_t (float): temporal sampling rate (s)

    Returns:
        Tuple: 4d SepVector.floatVector (wavelet for 3d nonlinear elastic prop)
          and list with a single 4d SepVector.floatVector (wavelet linear wave
          prop).
    """
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

  def _setup_src_devices(self, src_locations: np.ndarray, n_t: float) -> List:
    """Helper function to setup source devices needed for wave prop.

    Args:
        src_locations (np.ndarray): location of each source device. Should have
          shape (n_src,n_dim) where n_dim is the number of dimensions.
        n_t (float): number of time steps in the wavelet/data space

    Returns:
        List: list of pybind11 device classes
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
                          staggered_grid_hyper.getCpp(), int(n_t), 'linear', 4,
                          0, 0, 0))

    self.fd_param['n_src'] = len(src_locations)
    return src_devices_staggered_grids

  def _setup_rec_devices(self, rec_locations: np.ndarray, n_t: float) -> List:
    """Helper function to setup receiver devices needed for wave prop.

    Args:
        rec_locations (np.ndarray): location of each source device. Should have
          shape (n_rec, n_dim) or (n_src, n_rec, n_dim) where n_dim is the
          number of dimensions. The latter allows for different receiver
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
                          staggered_grid_hyper.getCpp(), int(n_t), 'linear', 4,
                          0, 0, 0))

    self.fd_param['n_rec'] = n_rec
    # self.rec_devices = rec_devices_staggered_grids
    return rec_devices_staggered_grids

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

  def _truncate_model(self, model):
    x_pad = self.fd_param['x_pad_minus'] + self._FAT
    x_pad_plus = self.fd_param['x_pad_plus'] + self._FAT
    z_pad = self.fd_param['z_pad_minus'] + self._FAT
    z_pad_plus = self.fd_param['z_pad_plus'] + self._FAT

    return model[..., x_pad:-x_pad_plus:, z_pad:-z_pad_plus:]


class ElasticIsotropic3D(ElasticIsotropic):
  """3D elastic, isotropic wave equation solver class.

  Velocity/stress formulation with a staggered grid. Expects model space to be
  paramterized by Vp, Vs, and Rho. Free surface condition is NOT avialable. 
  
  Typical usage example (see examples directory for more):
    from pyseis.wave_equations import elastic_isotropic
    
    elastic_3d = elastic_isotropic.ElasticIsotropic3D(
      vp_vs_rho_model_nd_array,
      (d_y, d_x, d_z),
      wavelet_nd_array,
      d_t,
      src_locations_nd_array,
      rec_locations_nd_array,
      gpus=[0,1,2,4],
      recording_components=['vy','vx','vz'])
    data = elastic_3d.forward(vp_vs_rho_model_nd_array)
  """
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

  def _make_sep_wavelets(self, wavelet: np.ndarray, d_t: float) -> Tuple:
    """Helper function to make SepVector wavelets needed for wave prop
    
    The elastic 3D prop uses a 4d SepVector for nonlinear prop and a list
    containing a single, 3d SepVector for linear prop. THIS IS FOR BACKWARDS
    COMPATIBILITY WITH PYBIND11 C++ CODE.

    Args:
      wavelet (np.ndarray): : source signature of wave equation. Also defines
        the recording duration of the seismic data space.
        - In 2D, elastic case, wavelet will have shape (5, n_t) describing the
          five source components (Fx, Fz, sxx, szz, sxz)
        - In 3D, elastic case, wavelet will have shape (9, n_t) describing the
          nine source components (Fy, Fx, Fz, syy, sxx, szz, syx, syz, sxz)
      d_t (float): temporal sampling rate (s)

    Returns:
        Tuple: 4d SepVector.floatVector (wavelet for 3d nonlinear elastic prop)
          and list with a single 4d SepVector.floatVector (wavelet linear wave
          prop).
    """
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
                          int(n_t), sep_par.param, 0, 0, 0, 0, 'linear', 1))

    self.fd_param['n_src'] = len(src_locations)
    return src_devices_staggered_grids

  def _setup_rec_devices(self, rec_locations: np.ndarray, n_t: float) -> List:
    """Helper function to setup receiver devices needed for wave prop.

    Args:
        rec_locations (np.ndarray): location of each source device. Should have
          shape (n_rec, n_dim) or (n_src, n_rec, n_dim) where n_dim is the
          number of dimensions. The latter allows for different receiver
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
                          int(n_t), sep_par.param, 0, 0, 0, 0, 'linear', 1))

    self.fd_param['n_rec'] = n_rec
    # self.rec_devices = rec_devices_staggered_grids
    return rec_devices_staggered_grids

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

  def _truncate_model(self, model):
    y_pad = self.fd_param['y_pad'] + self._FAT
    x_pad = self.fd_param['x_pad_minus'] + self._FAT
    x_pad_plus = self.fd_param['x_pad_plus'] + self._FAT
    z_pad = self.fd_param['z_pad_minus'] + self._FAT
    z_pad_plus = self.fd_param['z_pad_plus'] + self._FAT
    return model[..., y_pad:-y_pad:, x_pad:-x_pad_plus:, z_pad:-z_pad_plus:]


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