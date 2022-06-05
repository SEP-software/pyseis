import pytest
import numpy as np
from wave_equations.acoustic.AcousticIsotropic import AcousticIsotropic

N_X = 100
D_X = 10.0
N_Z = 50
D_Z = 5.0
model_sampling = (D_X, D_Z)
VP_1 = 1500
VP_2 = 2500

D_T = 0.001

#expected values
SUB = 2


@pytest.fixture
def vp_model_half_space():
  vp_model = np.zeros((N_X, N_Z))
  vp_model[:, :N_Z // 2] = VP_1
  vp_model[:, N_Z // 2:] = VP_2
  return vp_model


###############################################
#### AcousticIsotropic parent class tests #####
###############################################
def test_AcousticIsotropic_is_abstract():
  "AcousticIsotropic class should be purely abstract so will fail if initiailized"
  with pytest.raises(TypeError):
    acoustic_wave_equation = AcousticIsotropic()


def test_AcousticIsotropic_setup_subsampling(vp_model_half_space):
  #make dummy child class so we can test concrete methods of abstract class
  AcousticIsotropic.__abstractmethods__ = set()
  acoustic_wave_equation = AcousticIsotropic()

  acoustic_wave_equation.setup_subsampling(vp_model_half_space, D_T,
                                           model_sampling)

  assert SUB == acoustic_wave_equation.fd_param['sub']


def test_AcousticIsotropic_calc_subsampling(vp_model_half_space):
  #make dummy child class so we can test concrete methods of abstract class
  AcousticIsotropic.__abstractmethods__ = set()
  acoustic_wave_equation = AcousticIsotropic()

  sub = acoustic_wave_equation.calc_subsampling(vp_model_half_space, D_T,
                                                model_sampling)

  assert sub == SUB
