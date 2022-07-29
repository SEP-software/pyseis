import pytest
import numpy as np
from pysynth.data_gen import survey

N_X = 100
D_X = 10.0
N_Z = 50
D_Z = 5.0
N_X_PAD = 25
N_Z_PAD = 20
VP_1 = 1500
VP_2 = 2500

N_T = 1000
D_T = 0.01

DELAY = 2.0
DOM_FREQ = 5.0
CONFIG_RICKER = {'delay': DELAY, 'dom_freq': DOM_FREQ}

N_SRC_FULL = 21
O_X_SRC_FULL = 0.0
D_X_SRC_FULL = 50.0
Z_SRC_FULL = 15.0
N_REC_FULL = 201
O_X_REC_FULL = 0.0
D_X_REC_FULL = 5.0
Z_REC_FULL = 11.0

I_GPUS = [0, 1, 2, 3]

CONFIG_FULL_GEOMETRY = {
    'type': 'Full2D',
    'args': {
        'n_src': N_SRC_FULL,
        'o_x_src': O_X_SRC_FULL,
        'd_x_src': D_X_SRC_FULL,
        'z_src': Z_SRC_FULL,
        'n_rec': N_REC_FULL,
        'o_x_rec': O_X_REC_FULL,
        'd_x_rec': D_X_REC_FULL,
        'z_rec': Z_REC_FULL,
    }
}

DELAY = 2.0
DOM_FREQ = 5.0
CONFIG_RICKER = {
    'type': 'AcousticIsotropicRicker2D',
    'args': {
        'n_t': N_T,
        'd_t': D_T,
        'delay': DELAY,
        'dom_freq': DOM_FREQ
    }
}

CONFIG_ACOUSTIC_WAVE = {
    'type': 'AcousticIsotropic2D',
    'args': {
        'model_sampling': [D_X, D_Z],
        'd_t': D_T,
        'gpus': I_GPUS
    }
}

CONFIG = {
    'geometry': CONFIG_FULL_GEOMETRY,
    'wavelet': CONFIG_RICKER,
    'wave_equation': CONFIG_ACOUSTIC_WAVE
}


@pytest.fixture
def vp_model():
  vp_model = np.zeros((N_X, N_Z))
  vp_model[:, :N_Z // 2] = VP_1
  vp_model[:, N_Z // 2:] = VP_2
  return vp_model


@pytest.fixture
def vp_model2():
  vp_model = np.zeros((N_X, N_Z))
  vp_model[:, :N_Z // 4] = VP_1
  vp_model[:, N_Z // 4:] = VP_2
  return vp_model


def test_make_aquisition():
  geometry = survey.Survey._make_geometry(None, CONFIG_FULL_GEOMETRY)
  # check src and rec positions
  src_positions = geometry.get_src_positions()
  assert src_positions.shape == (N_SRC_FULL, 2)
  assert src_positions[0, 0] == O_X_SRC_FULL
  assert (src_positions[1, 0] -
          src_positions[0, 0]) == pytest.approx(D_X_SRC_FULL)
  assert np.all(src_positions[:, 1] == Z_SRC_FULL)
  rec_positions = geometry.get_rec_positions()
  assert rec_positions.shape == (N_REC_FULL, 2)
  assert rec_positions[0, 0] == O_X_REC_FULL
  assert (rec_positions[1, 0] -
          rec_positions[0, 0]) == pytest.approx(D_X_REC_FULL)
  assert np.all(rec_positions[:, 1] == Z_REC_FULL)


def test_make_wavelet():
  wavelet = survey.Survey._make_wavelet(None, CONFIG_RICKER)
  # check src and rec positions
  assert wavelet.get_arr().shape == (N_T,)


@pytest.mark.gpu
def test_make_wave_equation(vp_model):
  geometry = survey.Survey._make_geometry(None, CONFIG_FULL_GEOMETRY)
  wavelet = survey.Survey._make_wavelet(None, CONFIG_RICKER)

  CONFIG_ACOUSTIC_WAVE['args']['model'] = vp_model
  CONFIG_ACOUSTIC_WAVE['args']['wavelet'] = wavelet.get_arr()
  CONFIG_ACOUSTIC_WAVE['args']['d_t'] = wavelet.d_t
  CONFIG_ACOUSTIC_WAVE['args']['src_locations'] = geometry.get_src_positions()
  CONFIG_ACOUSTIC_WAVE['args']['rec_locations'] = geometry.get_rec_positions()
  wave_equation = survey.Survey._make_wave_equation(None, CONFIG_ACOUSTIC_WAVE)


@pytest.mark.gpu
def test_init(vp_model):
  test_survey = survey.Survey(vp_model, CONFIG)


@pytest.mark.gpu
def test_fwd(vp_model):
  test_survey = survey.Survey(vp_model, CONFIG)
  data = test_survey.fwd(vp_model)

  assert data.shape == (N_SRC_FULL, N_REC_FULL, N_T)
  assert np.amax(data) > 0.0


@pytest.mark.gpu
def test_fwd_diff_model_make_diff_data(vp_model, vp_model2):
  test_survey = survey.Survey(vp_model, CONFIG)
  data1 = test_survey.fwd(vp_model)
  data2 = test_survey.fwd(vp_model2)
  max_data = max(np.abs(np.amax(data1)), np.abs(np.amax(data2)))
  print(max_data)

  assert not np.allclose(vp_model, vp_model2)
  assert not np.allclose(data1, data2)
