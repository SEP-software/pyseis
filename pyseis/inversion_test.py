from mock import patch
import pytest
import numpy as np
from scipy import ndimage
from pyseis.wave_equations import acoustic_isotropic
from pyseis.wavelets.acoustic import Acoustic2D
from pyseis import inversion

N_X = 100
D_X = 10.0
N_Z = 50
D_Z = 5.0
model_sampling = (D_X, D_Z)
N_X_PAD = 25
N_Z_PAD = 20
VP_1 = 1500
VP_2 = 2500

N_T = 1000
D_T = 0.01

DELAY = 2.0
DOM_FREQ = 5.0
CONFIG_RICKER = {'delay': DELAY, 'dom_freq': DOM_FREQ}

N_SRCS = 8
Z_SHOT = 10.0
N_REC = 100
D_X_REC = 10.0
Z_REC = 5.0
#expected values
SUB = 3
I_GPUS = [0, 1, 2, 3]


@pytest.fixture
def ricker_wavelet():
  ricker = Acoustic2D.AcousticIsotropicRicker2D(N_T, D_T, DOM_FREQ, DELAY)
  return ricker.get_arr()


@pytest.fixture
def src_locations():
  x_locs = np.linspace(0.0, N_X * D_X, num=N_SRCS)
  z_locs = np.ones(N_SRCS) * Z_SHOT
  return np.array([x_locs, z_locs]).T


@pytest.fixture
def rec_locations():
  rec_x_locs = np.arange(N_REC) * D_X_REC
  rec_z_locs = np.ones(N_REC) * Z_REC
  return np.array([rec_x_locs, rec_z_locs]).T


@pytest.fixture
def vp_model_half_space():
  vp_model = np.zeros((N_X, N_Z))
  vp_model[:, :N_Z // 2] = VP_1
  vp_model[:, N_Z // 2:] = VP_2
  return vp_model


@pytest.fixture
def vp_model_full_space():
  return VP_1 * np.ones((N_X, N_Z))


@pytest.fixture
def acoustic_2d_weq_solver(ricker_wavelet, src_locations, rec_locations,
                           vp_model_half_space):
  return acoustic_isotropic.AcousticIsotropic2D(model=vp_model_half_space,
                                                model_sampling=(D_X, D_Z),
                                                model_padding=(N_X_PAD,
                                                               N_Z_PAD),
                                                wavelet=ricker_wavelet,
                                                d_t=D_T,
                                                src_locations=src_locations,
                                                rec_locations=rec_locations,
                                                gpus=I_GPUS)


@pytest.mark.gpu
def test_init_fwi(acoustic_2d_weq_solver, vp_model_full_space, tmp_path):
  # Arrange
  obs_data = np.zeros_like(acoustic_2d_weq_solver.data_sep.getNdArray())

  # Run
  fwi_prob = inversion.Fwi(wave_eq_solver=acoustic_2d_weq_solver,
                           obs_data=obs_data,
                           starting_model=vp_model_full_space,
                           num_iter=3,
                           work_dir=tmp_path)


@pytest.mark.gpu
def test_init_fwi_dispersion_fails(acoustic_2d_weq_solver, vp_model_full_space,
                                   tmp_path):
  # Arrange
  obs_data = np.zeros_like(acoustic_2d_weq_solver.data_sep.getNdArray())

  vp_model_will_fail = np.ones_like(vp_model_full_space)

  # Run
  with pytest.raises(RuntimeError):
    fwi_prob = inversion.Fwi(wave_eq_solver=acoustic_2d_weq_solver,
                             obs_data=obs_data,
                             starting_model=vp_model_will_fail,
                             num_iter=3,
                             work_dir=tmp_path)


@pytest.mark.gpu
def test_run_fwi(acoustic_2d_weq_solver, vp_model_half_space,
                 vp_model_full_space, tmp_path):
  # Arrange
  obs_data = acoustic_2d_weq_solver.forward(vp_model_half_space)
  fwi_prob = inversion.Fwi(wave_eq_solver=acoustic_2d_weq_solver,
                           obs_data=obs_data,
                           starting_model=vp_model_full_space,
                           num_iter=2,
                           solver_type='nlcg',
                           iterations_per_save=1,
                           work_dir=tmp_path)

  # Run
  history = fwi_prob.run()

  # Assert
  all_keys = ['inv_mod', 'gradient', 'model', 'residual', 'obj']
  for key in all_keys:
    assert history[key] is not None

  assert np.any(history['gradient'])


@pytest.mark.gpu
def test_fwi_with_grad_mask(acoustic_2d_weq_solver, vp_model_half_space,
                            vp_model_full_space, tmp_path):
  # Arrange
  gradient_mask = np.zeros_like(vp_model_full_space)
  obs_data = acoustic_2d_weq_solver.forward(vp_model_half_space)
  fwi_prob = inversion.Fwi(wave_eq_solver=acoustic_2d_weq_solver,
                           obs_data=obs_data,
                           starting_model=vp_model_full_space,
                           num_iter=2,
                           solver_type='nlcg',
                           iterations_per_save=1,
                           work_dir=tmp_path,
                           gradient_mask=gradient_mask)
  # Run
  history = fwi_prob.run()

  # FWI with a gradient mask of zeros will not change the model
  assert np.allclose(vp_model_full_space, history['inv_mod'])


@pytest.mark.gpu
def test_init_lsrtm(acoustic_2d_weq_solver, vp_model_full_space, tmp_path):
  # Arrange
  obs_data = np.zeros_like(acoustic_2d_weq_solver.data_sep.getNdArray())

  # Run
  lsrtm_prob = inversion.Lsrtm(wave_eq_solver=acoustic_2d_weq_solver,
                               linear_data=obs_data,
                               migration_model=vp_model_full_space,
                               num_iter=3,
                               work_dir=tmp_path)


@pytest.mark.gpu
def test_run_lsrtm(acoustic_2d_weq_solver, vp_model_half_space, tmp_path):
  # Arrange
  # make linear model
  lin_model = np.gradient(vp_model_half_space, axis=-1)
  # make migration velocity
  mig_vel = ndimage.gaussian_filter(vp_model_half_space, sigma=10)
  # make linear data
  lin_data = acoustic_2d_weq_solver.jacobian(lin_model, mig_vel)

  # Run
  lsrtm_prob = inversion.Lsrtm(wave_eq_solver=acoustic_2d_weq_solver,
                               linear_data=lin_data,
                               migration_model=mig_vel,
                               num_iter=3,
                               work_dir=tmp_path)
  # Run
  history = lsrtm_prob.run()
