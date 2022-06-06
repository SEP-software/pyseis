"""Defines the 2D and 3D ElasticIsotropic wave equation classes.

The ElasticIsotropic2D and ElasticIsotropic3D inherit from the abstract
ElasticIsotropic class. ElasticIsotropic2D and ElasticIsotropic3D can model
the elastic, isotropic wave equation in two and three dimensions, respectively,
using a staggered-grid implementation. With a pressure-wave velocity model,
source wavelet, source positions, and receiver positions, a user can forward
model the wave wave equation and sample at receiver locations. Pybind11 is used
to wrap C++ code which then uses CUDA kernels to execute finite difference
operations. Current implementation parallelizes shots over gpus. 

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
    data = Elastic_2d.fwd(vp_model_half_space)
"""
import numpy as np
from math import ceil
import abc
import Hypercube
import SepVector
import pyOperator as Operator
from wave_equations import wave_equation
# 2d pybind modules
from pyElastic_iso_float_nl import spaceInterpGpu as device_gpu_2d
from pyElastic_iso_float_nl import nonlinearPropElasticShotsGpu, ostream_redirect
from dataCompModule import ElasticDatComp as _ElasticDatComp2D
# 3d pybind modules
from pyElastic_iso_float_nl_3D import spaceInterpGpu_3D as device_gpu_3d
from pyElastic_iso_float_nl_3D import nonlinearPropElasticShotsGpu_3D
from dataCompModule_3D import ElasticDatComp_3D as _ElasticDatComp_3D


class ElasticIsotropic(wave_equation.WaveEquation):
  _BLOCK_SIZE = 16
  _FAT = 4
  _N_MODEL_PARAMETERS = 3

  def __init__(self):
    super().__init__()

  def _make(self,
            model,
            wavelet,
            d_t,
            src_locations,
            rec_locations,
            gpus,
            recording_components,
            lame_model=False,
            subsampling=None):
    """Make an elastic, isotropic wave-equation operator.

    Operator can be used to forward model the elastic, isotropic wave equation
    using a velocity-stress formulation, 2nd order time derivative stencil, and
    10th order laplacian stencil with a staggered grid implementation. 

    Args:
        model (np array): un-padded earth model that matches dimensions of wave
          equation operator (2D or 3D). Can be parameterized parameterized by
          wave velocity [vp,vs,rho] or lame parameters [rho,lame,mu]. If using
          lame parameterization, make sure the lame_model argument is set to
          True.
            - In 2D case, model will have (3, n_x, n_z) shape.
            - In 3D case, model will have (3, n_y, n_x, n_z) shape.
        model_sampling (tuple): spatial sampling of provided earth model.
        model_padding (tuple): desired amount of padding to use at propagation
        time to add on each axis of the earth model during wave prop.
        wavelet (np array): source signature of wave equation. 
          - In 2D case, wavelet will have shape (5, n_t) describing the five
            source components (Fx, Fz, sxx, szz, sxz)
          - In 3D case, wavelet will have shape (9, n_t) describing the nine
            source components (Fy, Fx, Fz, syy, sxx, szz, syx, syz, sxz)
        n_t (int): number of time step to propagate
        d_t (float): temporal sampling rate (s)
        src_locations (np array): coordinates of each seismic source to
          propagate. 
            - In 2D has shape (n_src, 2) where two values on fast axis describe
              the x and y positions, respectively.
            - In 3D has shape (n_src, 3) where the fast axis describes the y, x, and z
              positions, respectively.  
        rec_locations (np array): receiver locations of each src. The number of
          receivers must be the same for each shot, but the position of the
          receivers can change. 
            - In the 2D case, can either have shape (n_rec, 2) if the receiver
              positions are constant or (n_src,n_rec,2) if the receiver
              positions change with each shot.
            - In the 3D case, can either have shape (n_rec, 3) if the receiver
              positions are constant or (n_src,n_rec,3) if the receiver
              positions change with each shot. 
        gpus (list): the gpu devices to use for wave propagation.
        model_origins (tuple, optional): Origin of each axis of the earth
          model. If None, the origins are assumed to be 0.0.
        lame_model (bool, optional): Whether the provided model is parameterized
          by wave velocity [vp,vs,rho] or lame parameters [rho,lame,mu].
          Defaults to False.
    """
    # pads model, makes self.model_sep, and updates self.fd_param
    self._setup_model(model, lame_model)

    # make and set wavelet
    self._setup_wavelet(wavelet, d_t)

    # make and set source devices
    self._setup_src_devices(src_locations, self.fd_param['n_t'])

    # make and set rec devices
    self._setup_rec_devices(rec_locations, self.fd_param['n_t'])

    # make and set data space
    self._setup_data(self.fd_param['n_t'], d_t)

    # calculate and find subsampling
    self._setup_subsampling(model, d_t, self.model_sampling, lame_model)
    if subsampling is not None:
      if subsampling < self.fd_param['sub']:
        raise RuntimeError(
            f"User specified subsampling={subsampling} that will does not satisfy Courant condition. subsampling must be >={self.fd_param['sub']}"
        )
      self.fd_param['sub'] = subsampling
    # set gpus list
    self.fd_param['gpus'] = str(gpus)[1:-1]

    #set ginsu
    self.fd_param['ginsu'] = 0

    # make and set sep par
    self._setup_sep_par(self.fd_param)

    # make and set gpu operator
    self._setup_wave_prop_operator(self.data_sep, self.model_sep,
                                   self.sep_param, self.src_devices,
                                   self.rec_devices, self.wavelet_sep)

    # append wavefield sampling to gpu operator
    self._setup_wavefield_sampling_operator(recording_components, self.data_sep)

  def _setup_subsampling(self, model, d_t, model_sampling, lame_model=False):
    sub = self._calc_subsampling(model, d_t, model_sampling, lame_model)
    if 'sub' in self.fd_param:
      if sub > self.fd_param['sub']:
        raise RuntimeError(
            'Newly set model requires greater subsampling than what wave equation operator was initialized with. This is currently not allowed.'
        )
    self.fd_param['sub'] = sub

  def _calc_subsampling(self, model, d_t, model_sampling, lame_model=False):
    """Find time downsampling needed during propagation to remain stable.

    Args:
        model (nd nparray): model or model that will be propagated in. Should
          not include padding.
        d_t (float): initial sampling rate
        d_x (float): sampling rate of axis -2 of model
        d_z (float): sampling rate of axis -1 of model

    Returns:
        int: amount that input  d_t is to be downsampled in order for nl prop to remain stable
    """
    if lame_model:
      model = convert_to_vel(model)
    max_vel = np.amax(model[:2])
    d_t_sub = ceil(max_vel * d_t /
                   (min(model_sampling) * wave_equation.COURANT_LIMIT))

    return d_t_sub

  @abc.abstractmethod
  def _setup_wavefield_sampling_operator(self, recording_components, data_sep):
    pass


class ElasticIsotropic2D(ElasticIsotropic):
  _N_WFLD_COMPONENTS = 5

  def __init__(self,
               model,
               model_sampling,
               wavelet,
               d_t,
               src_locations,
               rec_locations,
               gpus,
               model_padding=(50, 50),
               model_origins=(0.0, 0.0),
               lame_model=False,
               recording_components=[
                   'vx',
                   'vz',
                   'sxx',
                   'szz',
                   'sxz',
               ],
               subsampling=None):
    super().__init__()
    self.wave_prop_cpp_op_class = _Ela2dWavePropCppOp
    self.required_sep_params = [
        'nx', 'dx', 'nz', 'dz', 'xPadMinus', 'xPadPlus', 'zPadMinus',
        'zPadPlus', 'mod_par', 'dts', 'nts', 'fMax', 'sub', 'nExp', 'iGpu',
        'blockSize', 'fat'
    ]

    self.model_sampling = model_sampling
    self.model_padding = model_padding
    self.model_origins = model_origins
    self._make(model, wavelet, d_t, src_locations, rec_locations, gpus,
               recording_components, lame_model, subsampling)

  def _setup_model(self, model, lame_model=False):
    if not lame_model:
      model = convert_to_lame(model)
    self.fd_param['mod_par'] = 1
    model, x_pad, x_pad_plus, new_o_x, z_pad, z_pad_plus, new_o_z = self._pad_model(
        model, self.model_sampling, self.model_padding, self.model_origins)
    self.model = model
    self.model_sep = SepVector.getSepVector(
        Hypercube.hypercube(axes=[
            Hypercube.axis(
                n=model.shape[2], o=new_o_z, d=self.model_sampling[1]),
            Hypercube.axis(
                n=model.shape[1], o=new_o_x, d=self.model_sampling[0]),
            Hypercube.axis(n=model.shape[0], o=0.0, d=1.0)
        ]))
    self.model_sep.getNdArray()[:] = self.model
    self.fd_param['n_x'] = model.shape[1]
    self.fd_param['n_z'] = model.shape[2]
    self.fd_param['o_x'] = new_o_x
    self.fd_param['o_z'] = new_o_z
    self.fd_param['d_x'] = self.model_sampling[0]
    self.fd_param['d_z'] = self.model_sampling[1]
    self.fd_param['x_pad_minus'] = x_pad
    self.fd_param['x_pad_plus'] = x_pad_plus
    self.fd_param['z_pad_minus'] = z_pad
    self.fd_param['z_pad_plus'] = z_pad_plus

  def _pad_model(self,
                 model,
                 model_sampling,
                 model_padding,
                 model_origins=None):
    """Pad 2d model.

    Finds the correct padding on either end of the axis so both directions are
    divisible by _BLOCK_SIZE for optimal gpu computation.

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
    n_x = model.shape[1]
    d_x = model_sampling[0]
    x_pad = model_padding[0]
    n_z = model.shape[2]
    d_z = model_sampling[1]
    z_pad = model_padding[1]
    if model_origins is None:
      model_origins = (0.0, 0.0)

    # Compute size of z_pad_plus
    n_z_total = z_pad * 2 + n_z
    ratio_z = n_z_total / self._BLOCK_SIZE
    nb_blockz = ceil(ratio_z)
    z_pad_plus = nb_blockz * self._BLOCK_SIZE - n_z - z_pad

    # Compute sixe of x_pad_plus
    n_x_total = x_pad * 2 + n_x
    ratio_x = n_x_total / self._BLOCK_SIZE
    nb_blockx = ceil(ratio_x)
    x_pad_plus = nb_blockx * self._BLOCK_SIZE - n_x - x_pad

    # pad
    model = np.pad(np.pad(model,
                          ((0, 0), (x_pad, x_pad_plus), (z_pad, z_pad_plus)),
                          mode='edge'),
                   ((0, 0), (self._FAT, self._FAT), (self._FAT, self._FAT)),
                   mode='constant')

    # Compute new origins
    new_o_x = model_origins[0] - (self._FAT + x_pad) * d_x
    new_o_z = model_origins[1] - (self._FAT + z_pad) * d_z

    return model, x_pad, x_pad_plus, new_o_x, z_pad, z_pad_plus, new_o_z

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
    self.src_devices = src_devices_staggered_grids

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
    self.rec_devices = rec_devices_staggered_grids

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

    self.data_sep = data_sep

  def _make_sep_wavelet(self, wavelet, d_t):
    n_t = wavelet.shape[-1]
    wavelet_sep = SepVector.getSepVector(
        Hypercube.hypercube(axes=[
            Hypercube.axis(n=n_t, o=0.0, d=d_t),
            Hypercube.axis(n=1),
            Hypercube.axis(n=self._N_WFLD_COMPONENTS),
            Hypercube.axis(n=1)
        ]))
    wavelet_sep.getNdArray()[0, :, 0, :] = wavelet

    return wavelet_sep

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

  def _setup_wavefield_sampling_operator(self, recording_components, data_sep):
    # _ElasticDatComp2D expects a string of comma seperated values
    recording_components = ",".join(recording_components)
    #make sampling opeartor
    wavefield_sampling_operator = _ElasticDatComp2D(recording_components,
                                                    data_sep)
    self.data_sep = wavefield_sampling_operator.range.clone()

    self.wave_prop_cpp_op = Operator.ChainOperator(self.wave_prop_cpp_op,
                                                   wavefield_sampling_operator)


class ElasticIsotropic3D(ElasticIsotropic):
  _N_WFLD_COMPONENTS = 9

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
               ],
               subsampling=None):
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
    self._make(model, wavelet, d_t, src_locations, rec_locations, gpus,
               recording_components, lame_model, subsampling)

  def _setup_model(self, model, lame_model=False):
    if not lame_model:
      model = convert_to_lame(model)

    self.fd_param['mod_par'] = 1
    model, y_pad, y_pad_plus, new_o_y, x_pad, x_pad_plus, new_o_x, z_pad, z_pad_plus, new_o_z = self._pad_model(
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

  def _pad_model(self,
                 model,
                 model_sampling,
                 model_padding,
                 model_origins=None):
    """Pad 3d model.

    Finds the correct padding on either end of the axis so both directions are
    divisible by _BLOCK_SIZE for optimal gpu computation.

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
    ratio_z = n_z_total / self._BLOCK_SIZE
    nb_blockz = ceil(ratio_z)
    z_pad_plus = nb_blockz * self._BLOCK_SIZE - n_z - z_pad

    # Compute sixe of x_pad_plus
    n_x_total = x_pad * 2 + n_x
    ratio_x = n_x_total / self._BLOCK_SIZE
    nb_blockx = ceil(ratio_x)
    x_pad_plus = nb_blockx * self._BLOCK_SIZE - n_x - x_pad

    # compute y axis padding
    y_pad_plus = y_pad

    # pad
    model = np.pad(np.pad(model, ((0, 0), (y_pad, y_pad_plus),
                                  (x_pad, x_pad_plus), (z_pad, z_pad_plus)),
                          mode='edge'),
                   ((0, 0), (self._FAT, self._FAT), (self._FAT, self._FAT),
                    (self._FAT, self._FAT)),
                   mode='constant')

    # Compute new origins
    new_o_y = model_origins[0] - (self._FAT + y_pad) * d_y
    new_o_x = model_origins[1] - (self._FAT + x_pad) * d_x
    new_o_z = model_origins[2] - (self._FAT + z_pad) * d_z

    return model, y_pad, y_pad_plus, new_o_y, x_pad, x_pad_plus, new_o_x, z_pad, z_pad_plus, new_o_z

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
    self.src_devices = src_devices_staggered_grids

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
    self.rec_devices = rec_devices_staggered_grids

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

    self.data_sep = data_sep

  def _make_sep_wavelet(self, wavelet, d_t):
    n_t = wavelet.shape[-1]
    wavelet_sep = SepVector.getSepVector(
        Hypercube.hypercube(axes=[
            Hypercube.axis(n=n_t, o=0.0, d=d_t),
            Hypercube.axis(n=1),
            Hypercube.axis(n=self._N_WFLD_COMPONENTS),
            Hypercube.axis(n=1)
        ]))
    wavelet_sep.getNdArray().flat[:] = wavelet

    return wavelet_sep

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

  def _setup_wavefield_sampling_operator(self, recording_components, data_sep):
    # _ElasticDatComp_3D expects a string of comma seperated values
    recording_components = ",".join(recording_components)
    #make sampling opeartor
    wavefield_sampling_operator = _ElasticDatComp_3D(recording_components,
                                                     data_sep)
    self.data_sep = wavefield_sampling_operator.range.clone()

    self.wave_prop_cpp_op = Operator.ChainOperator(self.wave_prop_cpp_op,
                                                   wavefield_sampling_operator)


class _Ela2dWavePropCppOp(wave_equation._WavePropCppOp):
  """Wrapper encapsulating PYBIND11 module for the wave propagator"""

  wave_prop_module = nonlinearPropElasticShotsGpu

  def set_background(self, model_sep):
    with ostream_redirect():
      self.wave_prop_operator.setBackground(model_sep.getCpp())


class _Ela3dWavePropCppOp(wave_equation._WavePropCppOp):
  """Wrapper encapsulating PYBIND11 module for the wave propagator"""

  wave_prop_module = nonlinearPropElasticShotsGpu_3D

  def set_background(self, model_sep):
    with ostream_redirect():
      self.wave_prop_operator.setBackground(model_sep.getCpp())


def convert_to_lame(model):
  """convert elastic model to rho, lame, and mu

  Args:
      model (nd numpy array): (n_params, ...)
  """

  converted_model = np.zeros_like(model)

  #VpVsRho to RhoLameMu (m/s|m/s|kg/m3 -> kg/m3|Pa|Pa)
  converted_model[0] += model[2]  #rho
  converted_model[1] += model[2] * (
      model[0] * model[0] - 2.0 * model[1] * model[1])  #lame
  converted_model[2] += model[2] * model[1] * model[1]  #mu

  return converted_model


def convert_to_vel(model):
  converted_model = np.zeros_like(model)
  #RhoLameMu to VpVsRho (kg/m3|Pa|Pa -> m/s|m/s|kg/m3)
  converted_model[0] += np.sqrt(np.divide((model[1] + 2 * model[2]),
                                          model[0]))  #vp
  converted_model[1] += np.sqrt(np.divide(model[2], model[0]))  #vs
  converted_model[2] += model[0]  #rho

  return converted_model