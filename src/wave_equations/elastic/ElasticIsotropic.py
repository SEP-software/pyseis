import numpy as np
from math import ceil
from wave_equations.WaveEquation import WaveEquation, COURANT_LIMIT


class ElasticIsotropic(WaveEquation):
  block_size = 16
  fat = 4

  def __init__(self):
    super().__init__()

  def make(self,
           model,
           wavelet,
           d_t,
           src_locations,
           rec_locations,
           gpus,
           recording_components,
           lame_model=False):
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
    self.setup_model(model, lame_model)

    # make and set wavelet
    self.setup_wavelet(wavelet, d_t)

    # make and set source devices
    self.setup_src_devices(src_locations, self.fd_param['n_t'])

    # make and set rec devices
    self.setup_rec_devices(rec_locations, self.fd_param['n_t'])

    # make and set data space
    self.setup_data(self.fd_param['n_t'], d_t)

    # calculate and find subsampling
    self.setup_subsampling(model, d_t, self.model_sampling, lame_model)

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

    # append wavefield sampling to gpu operator
    self.setup_wavefield_sampling_operator(recording_components, self.data_sep)

  def setup_subsampling(self, model, d_t, model_sampling, lame_model=False):
    sub = self.calc_subsampling(model, d_t, model_sampling, lame_model)
    if 'sub' in self.fd_param:
      if sub > self.fd_param['sub']:
        raise RuntimeError(
            'Newly set model requires greater subsampling than what wave equation operator was initialized with. This is currently not allowed.'
        )
    self.fd_param['sub'] = sub

  def calc_subsampling(self, model, d_t, model_sampling, lame_model=False):
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
    d_t_sub = ceil(max_vel * d_t / (min(model_sampling) * COURANT_LIMIT))

    # return ceil(d_t / d_t_sub) + 1
    return d_t_sub


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