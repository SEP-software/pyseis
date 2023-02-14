"""Defines the abstract WaveEquation class.

A WaveEquation object is able to forward time march a finite difference wave
equation in two or three dimensions and sample the resulting wavefield(s) at
receiver locations. AcousticIsotropic and ElasticIsotropic child classes
implement different wave equations but share common methods that are defined in
WaveEquation. All use an absorbing boundary to handle model edge effects during
wave propagation.
"""
import numpy as np
import abc
import math
from typing import List, Tuple

import genericIO
import pyOperator as Operator
import SepVector
from pyElastic_iso_float_nl_3D import ostream_redirect
from pyseis.wavelets import Wavelet

SEP_PARAM_CYPHER = {
    'ny': 'n_y',
    'dy': 'd_y',
    'nx': 'n_x',
    'dx': 'd_x',
    'nz': 'n_z',
    'dz': 'd_z',
    'yPad': 'y_pad',
    'xPadMinus': 'x_pad_minus',
    'xPadPlus': 'x_pad_plus',
    'zPadMinus': 'z_pad_minus',
    'zPadPlus': 'z_pad_plus',
    'mod_par': 'mod_par',
    'dts': 'd_t',
    'nts': 'n_t',
    'fMax': 'f_max',
    'sub': 'sub',
    'nShot': 'n_src',
    'nExp': 'n_src',
    'iGpu': 'gpus',
    'block_size': '_BLOCK_SIZE',
    'blockSize': '_BLOCK_SIZE',
    'fat': '_FAT',
    'ginsu': 'ginsu',
    'freeSurface': 'free_surface',
    'surfaceCondition': 'free_surface'
}
COURANT_LIMIT = 0.45
GRID_CELLS_PER_WAVELENGTH = 3.0


class WaveEquation(abc.ABC):
  """Abstract wave equation solver class.

  Contains all methods and variables common to elastic and acoustic, 2D and 3D
  wave equation solvers.
  """

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
               subsampling: int = None,
               free_surface: bool = False) -> None:
    """Base constructor for a wave equation solvers.

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
    if free_surface and not self._FREE_SURFACE_AVAIL:
      raise NotImplementedError(
          'free surface condition has not been implemented yet!')

    self.model_sep = None
    self.data_sep = None
    self.fd_param = {
        '_BLOCK_SIZE': self._BLOCK_SIZE,
        '_FAT': self._FAT,
        'mod_par': 1,
        'free_surface': int(free_surface)
    }
    self._nl_operator = None
    self._jac_operator = None
    self._fwi_operator = None

    self._make(model, wavelet, d_t, src_locations, rec_locations, gpus,
               model_padding, model_origins, model_sampling, subsampling,
               free_surface)

  def _make(self,
            model: np.ndarray,
            wavelet: np.ndarray,
            d_t: float,
            src_locations: np.ndarray,
            rec_locations: np.ndarray,
            gpus: Tuple[int],
            model_padding: Tuple[int] = None,
            model_origins: Tuple[int] = None,
            model_sampling: Tuple[float] = None,
            subsampling: int = None,
            free_surface: bool = False):
    """Helper function to make WaveEquation classes

    Args:
      model (np.ndarray): earth model without padding (2D or 3D). For the
        acoustic wave equation, this must be pressure wave velocity in (m/s). If
        elastic, the first axis must be the three elastic parameters (pressure
        wave velocity (m/s), shear wave velocity (m/s), and density (kg/m^3)).
          - a 3D elastic model will have shape (3, n_y, n_x, n_z)
          - a 2D acoustic model will have shape (n_x, n_z)
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
      model_sampling (Tuple[float, ...]): spatial sampling of provided earth
        model axes. Must be same length as number of dimensions of model.
          - (d_y, d_x, d_z) with 3D models
          - (d_x, d_z) with 2D models
      subsampling (Tuple[int, ...], optional): Exact subsampling multiple of d_t to
        use during finite difference wave propagation. If None, this is
        calculated automatically by the wave equation solver to obey the
        Courant condition. Defaults to None.
          - subsampling = 3 implies that the finite difference code will take
            time steps at intervals of d_t / 3.
      free_surface (bool, optional): whether of not to use a free surface
        condition. Defaults to False.
    """
    # default model padding
    if not model_padding:
      model_padding = (self._DEFAULT_PADDING,) * len(model_sampling)

    # default origins are at 0.0
    if not model_origins:
      model_origins = (0.0,) * len(model_sampling)

    # pads model, makes self.model_sep, and updates self.fd_param
    self.model_sep, self.model_padding = self._setup_model(
        model, model_padding, model_sampling, model_origins, free_surface)
    self.model_sampling = model_sampling

    # make and set wavelet
    self.wavelet_nl_sep, self.wavelet_lin_sep = self._setup_wavelet(
        wavelet, d_t)

    # check that that the starting model will not cause dispersion
    if not self.check_dispersion(self._get_velocities(model),
                                 self.model_sampling, self.fd_param['f_max']):
      min_vel = self.find_min_vel(self.model_sampling, self.fd_param['f_max'])
      raise RuntimeError(
          f"The provided model will cause dispersion because the minumum velocity value is too low. Given the current max frequency in the source wavelet, {self.fd_param['f_max']}Hz, and max spatial sampling, {max(self.model_sampling)}m, the minimum allowed velocity is {min_vel} m/s"
      )

    # make and set source devices
    self.src_devices = self._setup_src_devices(src_locations,
                                               self.fd_param['n_t'])

    # make and set rec devices
    self.rec_devices = self._setup_rec_devices(rec_locations,
                                               self.fd_param['n_t'])

    # make and set data space
    self.data_sep = self._setup_data(self.fd_param['n_t'], d_t)

    # calculate and find subsampling. pass only the velocity componenets
    velocities = self._get_velocities(model)
    self._setup_subsampling(velocities, d_t, model_sampling, subsampling)

    # set gpus list
    self.fd_param['gpus'] = str(gpus)[1:-1]

    #set ginsu to zero for now
    self.fd_param['ginsu'] = 0

    # make and set sep par
    self.sep_param = self._setup_sep_par(self.fd_param)

    # make nonlinear operator
    # self._operator = self._setup_operators(self.data_sep, self.model_sep,
    #                                        self.sep_param, self.src_devices,
    #                                        self.rec_devices,
    #                                        self.wavelet_nl_sep,
    #                                        self.wavelet_lin_sep)

    self._nl_operator = self._setup_nonlinear_operator(
        self.data_sep, self.model_sep, self.sep_param, self.src_devices,
        self.rec_devices, self.wavelet_nl_sep)

    self.data_sep = self._nl_operator.range.clone()

  def forward(self, model: np.ndarray) -> np.ndarray:
    """Run the nonlinear, forward wave equation.
    
    d = f(m)

    Args:
        model (np.ndarray): earth model in which to forward propagate.

    Returns:
        np.ndarray: seismic wavefield sampled at receivers
    """
    self._set_model(model)
    self._nl_operator.forward(0, self.model_sep, self.data_sep)
    return np.copy(self.data_sep.getNdArray())

  def jacobian(self,
               lin_model: np.ndarray,
               background_model: np.ndarray = None) -> np.ndarray:
    """Run the forward Jacobian of the wave equation
    
    d = Fm

    Args:
        lin_model (np.ndarray): linear model aka image space on which to scatter
          the source wavefield.
        background_model (np.ndarray, optional): The earth model around which to
          linearize the wave equation. When None, uses earth model from
          nonlinear forward. Defaults to None.

    Returns:
        np.ndarray: linear data recorded at receiver locations.
    """
    if self._jac_operator is None:
      self._jac_operator = self._setup_jacobian_operator(
          self._setup_data(self.fd_param['n_t'], self.fd_param['d_t']),
          self.model_sep, self.sep_param, self.src_devices, self.rec_devices,
          self.wavelet_lin_sep)
    self._set_lin_model(lin_model)
    if background_model is not None:
      self._set_background_model(background_model)

    self._jac_operator.forward(0, self.lin_model_sep, self.data_sep)
    return np.copy(self.data_sep.getNdArray())

  def jacobian_adjoint(self,
                       lin_data: np.ndarray,
                       background_model: np.ndarray = None) -> np.ndarray:
    """Run the adjoint Jacobian of the wave equation.
    
    m = F'd

    Args:
        lin_data (np.ndarray): linear data to create receiver wavefield.
        background_model (np.ndarray, optional): The earth model around which to
          linearize the wave equation. When None, uses earth model from
          nonlinear forward. Defaults to None.

    Returns:
        np.ndarray: linear model, zero lag cross correlation of receiver and
          source wavefields. 
    """
    if self._jac_operator is None:
      self._jac_operator = self._setup_jacobian_operator(
          self._setup_data(self.fd_param['n_t'], self.fd_param['d_t']),
          self.model_sep, self.sep_param, self.src_devices, self.rec_devices,
          self.wavelet_lin_sep)

    self._set_data(lin_data)
    if background_model is not None:
      self._set_background_model(background_model)

    self._jac_operator.adjoint(0, self.lin_model_sep, self.data_sep)
    return np.copy(self._truncate_model(self.lin_model_sep.getNdArray()))

  def dot_product_test(self,
                       verb: bool = False,
                       tolerance: float = 0.00001) -> bool:
    """Test the adjointness of the jacobian operator.

    Args:
        verb (bool, optional): Whether to priont intermediate results. Defaults to False.
        tolerance (float, optional): tolerance to pass test. Defaults to 0.00001.

    Returns:
        bool: whether test passes or not
    """
    if self._jac_operator is None:
      self._jac_operator = self._setup_jacobian_operator(
          self._setup_data(self.fd_param['n_t'], self.fd_param['d_t']),
          self.model_sep, self.sep_param, self.src_devices, self.rec_devices,
          self.wavelet_lin_sep)

    with ostream_redirect():
      return self._jac_operator.dotTest(verb, tolerance)

  def _setup_model(
      self,
      model: np.ndarray,
      model_padding: Tuple[int, ...],
      model_sampling: Tuple[float, ...],
      model_origins: Tuple[float, ...] = None,
      free_surface: bool = False
  ) -> Tuple[SepVector.floatVector, Tuple[Tuple[int, int], ...]]:
    """Helper function to initially setup of model space.
    
    Finds model padding that fits gpu block sizes. Pads model and sets padding
    parameters.

    Args:
      model (np.ndarray): earth model without padding (2D or 3D). For the
        acoustic wave equation, this must be pressure wave velocity in (m/s). If
        elastic, the first axis must be the three elastic parameters (pressure
        wave velocity (m/s), shear wave velocity (m/s), and density (kg/m^3)).
          - a 3D elastic model will have shape (3, n_y, n_x, n_z)
          - a 2D acoustic model will have shape (n_x, n_z)
      model_padding (Tuple[int, ...], optional): Thickness of absorbing boundary,
        defined in number of samples, to be added to each axis of the earth
        model. Just like the model_sampling arguement, must be same length as
        the number of dimensions of model. If None, the padding on each axis
        is 50 samples in 2D and 30 samples in 3D. Defaults to None.
      model_sampling (Tuple[float, ...]): spatial sampling of provided earth
        model axes. Must be same length as number of dimensions of model.
          - (d_y, d_x, d_z) with 3D models
          - (d_x, d_z) with 2D models
      model_origins (Tuple[int, ...], optional): Origin of each axis of the earth
        model. Just like the model_sampling arguement, must be same length as
        the number of dimensions of model. If None, the origins are assumed to
        be 0.0. Defaults to None.
      free_surface (bool, optional): whether of not to use a free surface
        condition. Defaults to False.
    Returns:
        Tuple[SepVector.floatVector, Tuple[Tuple[int,int], ...]]: padded model
          space in SepVector format and the padding parameters for each axis.
    """
    model_shape = self._get_model_shape(model)
    padding, padded_model_shape, padded_model_origins = self._calc_pad_params(
        model_shape, model_padding, model_sampling, self._FAT, model_origins,
        free_surface)
    self._set_padding_params(padding, padded_model_shape, padded_model_origins,
                             model_sampling)
    model = self._pad_model(model, padding, self._FAT)

    # make SepVector model
    model_sep = self._make_sep_vector_model_space(padded_model_shape,
                                                  padded_model_origins,
                                                  model_sampling)
    model_sep.getNdArray()[:] = model

    return model_sep, padding

  @abc.abstractmethod
  def _get_model_shape(self, model: np.ndarray) -> Tuple[int, ...]:
    """Abstract helper function to get shape of model space.
    
    Abstract because elastic model needs to disregard elastic parameter axis.

    Args:
        model (np.ndarray): model space

    Raises:
        NotImplementedError: if not implemented by child class

    Returns:
        Tuple[int, ...]: shape of model space
    """
    raise NotImplementedError(
        '_get_model_shape not overwritten by WaveEquation child class')

  def _calc_pad_params(
      self,
      model_shape: Tuple[int, ...],
      model_padding: Tuple[int, ...],
      model_sampling: Tuple[float, ...],
      fat: int,
      model_origins: Tuple[float, ...] = None,
      free_surface: bool = False) -> Tuple[Tuple[int, int], ...]:
    """Helper function to find correct padding of each spatial axis.

    Returns the padding, new shape, and new origins of the model based on the
    given parameters.

    Args:
        model_shape (Tuple[int, ...]): The spatial shape of the unpadded model.
          In the elastic case this should not include the elastic parameter axis.
        model_padding (Tuple[int, ...], optional): Thickness of absorbing boundary,
          defined in number of samples, to be added to each axis of the earth
          model. Just like the model_sampling arguement, must be same length as
          the number of dimensions of model. If None, the padding on each axis
          is 50 samples in 2D and 30 samples in 3D. Defaults to None.
        model_sampling (Tuple[float, ...]): spatial sampling of provided earth
          model axes. Must be same length as number of dimensions of model.
            - (d_y, d_x, d_z) with 3D models
            - (d_x, d_z) with 2D models
        fat (int): the additional boundary to be added around the padding. Used
          to make laplacian computation more easily self adjoint.
        model_origins (Tuple[float, ...], optional): Origin of each axis of the
          model. Just like the model_sampling arguement, must be same length as
          the number of dimensions of model. If None, the origins are assumed to
          be 0.0. Defaults to None.
        free_surface (bool, optional): whether of not to use a free surface
          condition. Defaults to False.

    Raises:
        ValueError: if model_shape, model_padding, model_origins, or
          model_sampling are not the same length

    Returns:
        Tuple[Tuple[int, int], ...]: the padding parameters for each axis
    """
    # check that dimensions are consistent
    if len(model_shape) != len(model_padding):
      raise ValueError(
          f'Shape of model_padding has {len(model_padding)} dimension but model has {len(model_shape)}'
      )
    if len(model_shape) != len(model_sampling):
      raise ValueError(
          f'Shape of model_sampling has {len(model_sampling)} dimension but model has {len(model_shape)}'
      )
    if model_origins and len(model_shape) != len(model_origins):
      raise ValueError(
          f'Shape of model_origins has {len(model_origins)} dimension but model has {len(model_shape)}'
      )
    elif not model_origins:
      model_origins = (0.0,) * len(model_shape)

    padding = []

    # y axis if it exists
    if len(model_shape) == 3:
      padding.append((model_padding[0], model_padding[0]))

    # x axis
    padding.append(
        self._calc_pad_minus_plus(model_shape[-2], model_padding[-2], fat))

    # z axis
    padding.append(
        self._calc_pad_minus_plus(model_shape[-1],
                                  model_padding[-1],
                                  fat,
                                  free_surface=free_surface))

    # new origins
    new_origins = [
        old_origin - (fat + pad_minus) * sampling for old_origin, (
            pad_minus,
            _), sampling in zip(model_origins, padding, model_sampling)
    ]

    # new shape
    new_shape = [
        fat + pad_minus + shape + pad_plus + fat
        for shape, (pad_minus, pad_plus) in zip(model_shape, padding)
    ]

    return tuple(padding), tuple(new_shape), tuple(new_origins)

  def _calc_pad_minus_plus(self,
                           axis_size: int,
                           axis_padding: int,
                           fat: int,
                           free_surface: bool = False) -> Tuple[int, int]:
    """Helper function to find the padding for a single axis
    
    Returns the pad minus and pad plus values for the given axis size and 
    so an axis size is divisible by gpu block size after padding. In 3D case,
    the y axis does not need to be divisible by block size.
   
    Args:
      axis_size (int): Size of the axis
      axis_padding (int): Padding of the axis
      fat (int): the additional boundary to be added around the padding. Used
        to make laplacian computation more easily self adjoint.
      free_surface (bool, optional): whether of not to use a free surface
          condition. Defaults to False.
          
    Returns:
        Tuple[int, int]: Tuple of pad minus and pad plus values for the axis
    """
    if free_surface:
      pad_minus = self._get_free_surface_pad_minus()
      axis_total_samples = pad_minus + axis_size + axis_padding
    else:
      axis_total_samples = axis_padding * 2 + axis_size
      pad_minus = axis_padding

    n_gpu_blocks = math.ceil(axis_total_samples / self._BLOCK_SIZE)
    pad_plus = n_gpu_blocks * self._BLOCK_SIZE - axis_size - pad_minus

    return (pad_minus, pad_plus)

  @abc.abstractmethod
  def _get_free_surface_pad_minus(self) -> int:
    """Abstract helper function to get the amount of padding to add to the free surface

    Raises:
        NotImplementedError: if not implemented by child class

    Returns:
        int: number of cells to pad
    """
    raise NotImplementedError(
        '_get_free_surface_pad_minus not overwritten by WaveEquation child class'
    )

  def _set_padding_params(self, padding: Tuple[Tuple[int, int], ...],
                          padded_model_shape: Tuple[int, ...],
                          padded_model_origins: Tuple[float, ...],
                          padded_model_sampling: Tuple[float, ...]) -> None:
    """Helper function to set the padding parameters in the fd_param dictionary.
    
    The fd_param dictionary is used to create a genericIO io object which is
    required to instantiate the gpu, pybind11 wave prop classes.

    Args:
        padding (Tuple[Tuple[int, int], ...]): the padding (minus, plus)
          tuples that was applied to each axis.
        padded_model_shape (Tuple[int, ...]): the shape of the model after
          padding. If elastic, this shape excludes the elastic parameter axis.
        padded_model_origins (Tuple[float, ...]): the new model origins after
          padding.
        padded_model_sampling (Tuple[float, ...]): the sampling rate of each
          axis
    """
    if len(padded_model_shape) == 3:
      self.fd_param['n_y'] = padded_model_shape[0]
      self.fd_param['o_y'] = padded_model_origins[0]
      self.fd_param['d_y'] = padded_model_sampling[0]
      self.fd_param['y_pad'] = padding[0][0]

    self.fd_param['n_x'] = padded_model_shape[-2]
    self.fd_param['o_x'] = padded_model_origins[-2]
    self.fd_param['d_x'] = padded_model_sampling[-2]
    self.fd_param['x_pad_minus'] = padding[-2][0]
    self.fd_param['x_pad_plus'] = padding[-2][1]

    self.fd_param['n_z'] = padded_model_shape[-1]
    self.fd_param['o_z'] = padded_model_origins[-1]
    self.fd_param['d_z'] = padded_model_sampling[-1]
    self.fd_param['z_pad_minus'] = padding[-1][0]
    self.fd_param['z_pad_plus'] = padding[-1][1]

  @abc.abstractmethod
  def _pad_model(self, model: np.ndarray, padding: Tuple[Tuple[int, int], ...],
                 fat: int) -> np.ndarray:
    """Abstract helper to pad earth models before wave prop.
    
    Abstract because elastic models have extra axis which does not need padding

    Args:
        model (np.ndarray):  earth model without padding (2D or 3D). For the
        acoustic wave equation, this must be pressure wave velocity in (m/s). If
        elastic, the first axis must be the three elastic parameters (pressure
        wave velocity (m/s), shear wave velocity (m/s), and density (kg/m^3)).
          - a 3D elastic model will have shape (3, n_y, n_x, n_z)
          - a 2D acoustic model will have shape (n_x, n_z)
        padding (Tuple[Tuple[int, int], ...]): the padding (minus, plus)
          tuples that was applied to each axis.
        fat (int): the additional boundary to be added around the padding. Used
          to make laplacian computation more easily self adjoint.

    Raises:
        NotImplementedError: if not implemented by child class

    Returns:
        np.ndarray: padded earth model
    """
    raise NotImplementedError(
        '_pad_model not overwritten by WaveEquation child class')

  @abc.abstractmethod
  def _make_sep_vector_model_space(
      self, shape: Tuple[int, ...], origins: Tuple[float, ...],
      sampling: Tuple[float, ...]) -> SepVector.floatVector:
    """Abstract helper to make empty SepVector moddel space

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
    raise NotImplementedError(
        '_pad_model not overwritten by WaveEquation child class')

  def _setup_wavelet(
      self, wavelet: np.ndarray,
      d_t: float) -> Tuple[SepVector.floatVector, SepVector.floatVector]:
    """Helper function to set create the necessary wavelets for wave equations
    
    pybind11 operators expect SepVectors. Create them from provided numpy format

    Args:
        wavelet (np.ndarray): source signature of wave equation. Also defines
        the recording duration of the seismic data space.
        - In 2D and 3D acoustic case, wavelet will have shape (n_t)
        - In 2D, elastic case, wavelet will have shape (5, n_t) describing the
          five source components (Fx, Fz, sxx, szz, sxz)
        - In 3D, elastic case, wavelet will have shape (9, n_t) describing the
          nine source components (Fy, Fx, Fz, syy, sxx, szz, syx, syz, sxz)
        d_t (float): temporal sampling rate (s)

    Returns:
        Tuple[SepVector.floatVector, SepVector.floatVector]: the wavelets for
          nonlinear and linear wave prop in SepVector format
    """
    self.fd_param['d_t'] = d_t
    self.fd_param['n_t'] = wavelet.shape[-1]
    self.fd_param['f_max'] = Wavelet.calc_max_freq(wavelet, d_t)
    return self._make_sep_wavelets(wavelet, d_t)

  @abc.abstractmethod
  def _make_sep_wavelets(self, wavelet: np.ndarray, d_t: float) -> Tuple:
    """Abstract helper function to make SepVector wavelets needed for wave prop
    
    Abstract because the shape and type of elastic, acoustic, 2D, and 3D
      wavelets all vary.

    Args:
      wavelet (np.ndarray): : source signature of wave equation. Also defines
        the recording duration of the seismic data space.
        - In 2D and 3D acoustic case, wavelet will have shape (n_t)
        - In 2D, elastic case, wavelet will have shape (5, n_t) describing the
          five source components (Fx, Fz, sxx, szz, sxz)
        - In 3D, elastic case, wavelet will have shape (9, n_t) describing the
          nine source components (Fy, Fx, Fz, syy, sxx, szz, syx, syz, sxz)
      d_t (float): temporal sampling rate (s)

    Raises:
        NotImplementedError: if not implemented by child class

    Returns:
        Tuple: the wavelets for nonlinear and linear wave prop in SepVector
          format
    """
    raise NotImplementedError(
        '_calc_min_subsampling not overwritten by WaveEquation child class')

  @abc.abstractmethod
  def _setup_src_devices(self, src_locations: np.ndarray, n_t: float) -> List:
    """Abstract helper function to setup source devices needed for wave prop
    
    Abstract because elastic and acoustic use different pybind11 device classes.

    Args:
        src_locations (np.ndarray): location of each source device. Should have
          shape (n_src,n_dim) where n_dim is the number of dimensions.
        n_t (float): number of time steps in the wavelet/data space

    Raises:
        NotImplementedError: if not implemented by child class

    Returns:
        List: list of pybind11 device classes
    """
    raise NotImplementedError(
        '_setup_src_devices not overwritten by WaveEquation child class')

  @abc.abstractmethod
  def _setup_rec_devices(self, rec_locations: np.ndarray, n_t: float) -> List:
    """Abstract helper function to setup receiver devices needed for wave prop
    
    Abstract because elastic and acoustic use different pybind11 device classes.

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
    raise NotImplementedError(
        '_setup_rec_devices not overwritten by WaveEquation child class')

  def _setup_sep_par(self, fd_param: dict) -> genericIO.io:
    """Helper function to settup io object required by pybind11 gpu wave prop classes.
    
    Checks that all the required finite difference parameters are set.

    Args:
        fd_param (dict): A dictionary of finite difference parameters to create
        the io object.

    Raises:
        RuntimeError: if a required parameter is not a key within the passed
          fd_param dict

    Returns:
        genericIO.io: io object
    """
    sep_param_dict = {}
    for required_sep_param in self.required_sep_params:
      fd_param_key = SEP_PARAM_CYPHER[required_sep_param]
      if fd_param_key not in fd_param:
        raise RuntimeError(f'{fd_param_key} was not set.')
      sep_param_dict[required_sep_param] = fd_param[fd_param_key]

    return self._make_sep_par(sep_param_dict)

  @abc.abstractmethod
  def _setup_data(self,
                  n_t: int,
                  d_t: float,
                  data: np.ndarray = None) -> SepVector.floatVector:
    """Abstract helper to setup the SepVector data space.
    
    Abstract beceuse the dimensioanlty of the SepVector changes with elastic
    parameter axis.

    Args:
        n_t (int): number of time samples in the data space
        d_t (float): time sampling in the data space (s)
        data (np.ndarray, optional): Data to fill SepVector. If None, filled
          with zeros. Defaults to None.

    Raises:
        NotImplementedError: if not implemented by child class

    Returns:
        SepVector.floatVector: data space
    """
    raise NotImplementedError(
        '_setup_data not overwritten by WaveEquation child class')

  @abc.abstractmethod
  def _get_velocities(self, model: np.ndarray) -> np.ndarray:
    """Abstract helper function to return only the velocity components of a model.
    
    Abstract beceuse the elastic models need to disregard the denisty parameter.

    Args:
        model (np.ndarray): earth model

    Raises:
        NotImplementedError: if not implemented by child class

    Returns:
        np.ndarray: velocity components of model
    """
    raise NotImplementedError(
        '_get_velocities not overwritten by WaveEquation child class')

  def _make_sep_par(self, param_dict: dict) -> genericIO.io:
    """Helper function to set turn a dictionary of kwargs into a genericIO par 
    object needed for wave prop

    Args:
        dict (dictionary): a dictionary of args passed to nonlinearPropShotsGpu
          constructor

    Returns:
        genericIO.io: io object
    """
    kwargs_str = {
        key: str(value)[1:-1] if isinstance(value, list) else str(value)
        for key, value in param_dict.items()
    }

    return genericIO.io(params=kwargs_str)

  def _setup_nl_wave_op(
      self, data_sep: SepVector.floatVector, model_sep: SepVector.floatVector,
      sep_par: genericIO.io, src_devices: List, rec_devices: List,
      wavelet_nl_sep: SepVector.floatVector) -> Operator.Operator:
    """Helper function to create the nonlinear wave prop operator.

    Args:
        data_sep (SepVector.floatVector): data space in SepVector format.
        model_sep (SepVector.floatVector): model space in SepVector format.
        sep_par (genericIO.io): io object
        src_devices (List): source devices
        rec_devices (List): receiver devices
        wavelet_nl_sep (SepVector.floatVector): nonlinear wavelet

    Returns:
        Operator.NonLinearOperator: nonlinear operator
    """
    return _NonlinearWaveCppOp(model_sep, data_sep, sep_par, src_devices,
                               rec_devices, wavelet_nl_sep,
                               self._nl_wave_pybind_class)

  def _setup_jac_wave_op(
      self, data_sep: SepVector.floatVector, model_sep: SepVector.floatVector,
      sep_par: genericIO.io, src_devices: List, rec_devices: List,
      wavelet_lin_sep: SepVector.floatVector) -> Operator.Operator:
    """Helper function to create the jacobian wave prop operator.

    Args:
        data_sep (SepVector.floatVector): data space in SepVector format.
        model_sep (SepVector.floatVector): model space in SepVector format.
        sep_par (genericIO.io): io object
        src_devices (List): source devices
        rec_devices (List): receiver devices
        wavelet_lin_sep (SepVector.floatVector): linear wavelet
    """
    self.lin_model_sep = self.model_sep.clone()
    return _JacobianWaveCppOp(self.lin_model_sep, data_sep, model_sep, sep_par,
                              src_devices, rec_devices, wavelet_lin_sep,
                              self._jac_wave_pybind_class)

  def _setup_fwi_op(self):
    """creates an operator that can be used with FWI.

    Returns:
        Operator.NonlinearOperator: a nonliner operator that has the nonlinear
          and jacobian operators combined
    """
    if self._jac_operator is None:
      self._jac_operator = self._setup_jacobian_operator(
          self._setup_data(self.fd_param['n_t'], self.fd_param['d_t']),
          self.model_sep, self.sep_param, self.src_devices, self.rec_devices,
          self.wavelet_lin_sep)

    return self._combine_nl_and_jac_operators()

  @abc.abstractmethod
  def _setup_nonlinear_operator(
      self, data_sep: SepVector.floatVector, model_sep: SepVector.floatVector,
      sep_par: genericIO.io, src_devices: List, rec_devices: List,
      wavelet_nl_sep: SepVector.floatVector) -> Operator.Operator:
    """Helper function to set up the nonlinear operators
    
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
    raise NotImplementedError(
        '_setup_nonlinear_operator not overwritten by WaveEquation child class')

  @abc.abstractmethod
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
    raise NotImplementedError(
        '_setup_nonlinear_operator not overwritten by WaveEquation child class')

  @abc.abstractmethod
  def _combine_nl_and_jac_operators(self) -> Operator.NonLinearOperator:
    """Combine the nonlinear and jacobian operators into one nonlinear operator

    Returns:
        Operator.NonlinearOperator: a nonlinear operator that combines the
          nonlienar and jacobian operators
    """
    raise NotImplementedError(
        '_setup_nonlinear_operator not overwritten by WaveEquation child class')

  def _set_lin_model(self, lin_model: np.ndarray) -> None:
    """Helper function to set the SepVector linear model space.
    
    Takes unpadded model and input. Pads then sets model space SepVector.

    Args:
        lin_model (np.ndarray): unpadded linear model space.
    """
    # pad model
    lin_model = self._pad_model(lin_model, self.model_padding, self._FAT)

    # set sep model
    self.lin_model_sep.getNdArray()[:] = lin_model

  def _set_background_model(self, background_model: np.ndarray) -> None:
    """Helper function to set the SepVector background for the jacobian operator. 

    Args:
        background_model (np.ndarray): unpadded background model space.
    """
    background_model = self._pad_model(background_model, self.model_padding,
                                       self._FAT)
    background_model_sep = self.model_sep.clone()
    background_model_sep.getNdArray()[:] = background_model
    self._jac_operator.set_background(background_model_sep)

  def _set_data(self, data: np.ndarray) -> None:
    """Helper function to set the SepVector data space

    Args:
        data (np.ndarray): np array data space
    """
    self.data_sep.getNdArray()[:] = data

  def _set_model(self, model: np.ndarray) -> None:
    """Helper function to set the SepVector model space.
    
    Takes unpadded model and input. Pads then sets model space SepVector.

    Args:
        model (np.ndarray): unpadded model space.
    """
    # pad model
    model = self._pad_model(model, self.model_padding, self._FAT)

    # set sep model
    self.model_sep.getNdArray()[:] = model

  def _get_data(self) -> np.ndarray:
    """Helper function to get the data in numpy format

    Returns:
        np.ndarray: data
    """
    return self.data_sep.getNdArray()

  def _get_model(self, padded: bool = False) -> np.ndarray:
    """Helper function to get the model space in numpy format.
    
    Returns the unpadded model by default. If padded = True will return the
    model with padding still included.

    Args:
        padded (bool, optional): Whether to return the model with padding.
          Defaults to False.

    Returns:
        np.ndarray: model space
    """
    model = self.model_sep.getNdArray()
    if not padded:
      model = self._truncate_model(model)

    return model

  def _setup_subsampling(self,
                         vel_models: np.ndarray,
                         d_t: float,
                         model_sampling: Tuple[float, ...],
                         subsampling: int = None):
    """Helper function to initially setup the subsampling rate for wave prop.
    
    For wave prop to remain stable, we need to take time steps that obey the 
    Courant condition. This is a function of the spatial sampling of the model
    and the minumum velocity (vp and vs velocity).

    Args:
        vel_models (np.ndarray): the unpadded velocity model(s). Vp and Vs must
        d_t (float): d_t (float): temporal sampling rate (s) if the wavelet
          and data.
        model_sampling (Tuple[float, ...]): spatial sampling of provided earth
          model axes. Must be same length as number of dimensions of model.
            - (d_y, d_x, d_z) with 3D models
            - (d_x, d_z) with 2D models
        subsampling (int, optional): User can specify the desired subsampling.
          Must be greater than subsampling needed to obey Courant. If None, the
          subsampling value is calcualted. Defaults to None.

    Raises:
        RuntimeError: if subsampling is already set and is less than what is 
          required to remain stable.
        RuntimeError: if user provided subsampling is less than what is 
          required to remain stable.
    """
    # caclulate subsampling minimum
    min_sub = self._calc_min_subsampling(vel_models, d_t, model_sampling)
    # if subsampling was already used to init a pybind operator, check that the newly caclulated subsampling is lesser
    if 'sub' in self.fd_param:
      if min_sub > self.fd_param['sub']:
        raise RuntimeError(
            'Newly set model requires greater subsampling than what wave equation operator was initialized with. This is currently not allowed.'
        )
    # user specified subsampling must be greater than calculated subsampling
    if subsampling is not None:
      if subsampling < min_sub:
        raise RuntimeError(
            f"User specified subsampling={subsampling} that does not satisfy Courant condition. subsampling must be >={min_sub}"
        )
    else:
      subsampling = min_sub
    self.fd_param['sub'] = subsampling

  def _calc_min_subsampling(self, vel_models: np.ndarray, d_t: float,
                            model_sampling: Tuple[float, ...]) -> int:
    """Find minimum downsampling needed during propagation to remain stable.
    
    For wave prop to remain stable, we need to take time steps that obey the 
    Courant condition. This is a function of the spatial sampling of the model
    and the minumum velocity (vp and vs velocity).

    Args:
        models (nd nparray): model or models that will be propagated in. Should
          not include padding
        d_t (float): initial sampling rate

    Returns:
        int: amount that input  d_t is to be downsampled in order for nl prop to remain stable
    """
    max_vel = np.amax(vel_models)
    d_t_sub = math.ceil(max_vel * d_t / (min(model_sampling) * COURANT_LIMIT))
    return d_t_sub

  def check_dispersion(self, velocities, sampling, f_max):
    # find minumum velocity greater than 0
    min_vel = velocities[velocities != 0].min()

    # check dispersion
    max_spatial_sampling = max(sampling)
    dispersion = min_vel / f_max / max_spatial_sampling

    if dispersion < GRID_CELLS_PER_WAVELENGTH:
      return False
    else:
      return True

  def find_min_vel(self, sampling, f_max):
    max_spatial_sampling = max(sampling)
    return GRID_CELLS_PER_WAVELENGTH * max_spatial_sampling * f_max


class _NonlinearWaveCppOp(abc.ABC, Operator.Operator):
  """Wrapper encapsulating PYBIND11 module for the nonlinear wave propagator"""

  def __init__(self, model_sep, data_sep, sep_par, src_devices, rec_devices,
               wavelet_nl_sep, _nl_wave_pybind_class):
    self.setDomainRange(model_sep, data_sep)
    if not isinstance(src_devices[0], list):
      src_devices = [src_devices]
    if not isinstance(rec_devices[0], list):
      rec_devices = [rec_devices]
    self.wave_prop_operator = _nl_wave_pybind_class(model_sep.getCpp(),
                                                    sep_par.param, *src_devices,
                                                    *rec_devices)
    self.wavelet_nl_sep = wavelet_nl_sep

  def forward(self, add, model_sep, data_sep):
    #Setting elastic model parameters
    self.set_background(model_sep)
    with ostream_redirect():
      self.wave_prop_operator.forward(add, self.wavelet_nl_sep.getCpp(),
                                      data_sep.getCpp())

  def set_background(self, model_sep):
    with ostream_redirect():
      self.wave_prop_operator.setBackground(model_sep.getCpp())


class _JacobianWaveCppOp(abc.ABC, Operator.Operator):
  """Wrapper encapsulating PYBIND11 module for the Jocobian aka Born wave propagator"""

  def __init__(self, lin_model_sep, data_sep, velocity_sep, sep_par,
               src_devices, rec_devices, wavelet_lin_sep,
               _jac_wave_pybind_class):

    self.setDomainRange(lin_model_sep, data_sep)
    if not isinstance(src_devices[0], list):
      src_devices = [src_devices]
    if not isinstance(rec_devices[0], list):
      rec_devices = [rec_devices]

    self.wave_prop_operator = _jac_wave_pybind_class(velocity_sep.getCpp(),
                                                     sep_par.param,
                                                     *src_devices,
                                                     wavelet_lin_sep,
                                                     *rec_devices)

  def forward(self, add, model_sep, data_sep):
    with ostream_redirect():
      self.wave_prop_operator.forward(add, model_sep.getCpp(),
                                      data_sep.getCpp())

  def adjoint(self, add, model_sep, data_sep):
    with ostream_redirect():
      self.wave_prop_operator.adjoint(add, model_sep.getCpp(),
                                      data_sep.getCpp())

  def set_background(self, model_sep):
    with ostream_redirect():
      self.wave_prop_operator.setBackground(model_sep.getCpp())
