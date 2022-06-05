import pytest
import numpy as np
from wave_equations.elastic.ElasticIsotropic import ElasticIsotropic, convert_to_lame, convert_to_vel

N_X = 100
D_X = 10.0
N_Z = 50
D_Z = 5.0

V_P = 2500
V_S = 1500
RHO = 1000

D_T = 0.001

#expected values
SUB = 2


@pytest.fixture
def vp_vs_rho_model_2d():
  vp_2d = V_P * np.ones((N_X, N_Z))
  vs_2d = V_S * np.ones((N_X, N_Z))
  rho_2d = RHO * np.ones((N_X, N_Z))

  return np.array((vp_2d, vs_2d, rho_2d))


def test_calc_subsampling_lame_model(vp_vs_rho_model_2d):
  #make dummy child class so we can test concrete methods of abstract class
  ElasticIsotropic.__abstractmethods__ = set()
  elastic_wave_equation = ElasticIsotropic()
  rho_lame_mu_model_2d = convert_to_lame(vp_vs_rho_model_2d)

  sub = elastic_wave_equation.calc_subsampling(rho_lame_mu_model_2d,
                                               D_T, (D_X, D_Z),
                                               lame_model=True)

  assert sub == SUB


def test_calc_subsampling_vel_model(vp_vs_rho_model_2d):
  #make dummy child class so we can test concrete methods of abstract class
  ElasticIsotropic.__abstractmethods__ = set()
  elastic_wave_equation = ElasticIsotropic()

  sub = elastic_wave_equation.calc_subsampling(vp_vs_rho_model_2d,
                                               D_T, (D_X, D_Z),
                                               lame_model=False)

  assert sub == SUB


def test_convert_elastic_params(vp_vs_rho_model_2d):
  rho_lame_mu_model_2d = convert_to_lame(vp_vs_rho_model_2d)

  assert rho_lame_mu_model_2d.shape == vp_vs_rho_model_2d.shape
  assert np.allclose(rho_lame_mu_model_2d[0], vp_vs_rho_model_2d[2])

  recovered_vp_vs_rho_model_2d = convert_to_vel(rho_lame_mu_model_2d)
  assert np.allclose(recovered_vp_vs_rho_model_2d, vp_vs_rho_model_2d)
