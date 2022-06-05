from mock import patch
import pytest
import numpy as np
from wave_equations.elastic.ElasticIsotropic2D import ElasticIsotropic2D
from wave_equations.elastic.ElasticIsotropic import convert_to_lame, convert_to_vel
from wavelets.elastic import Elastic2D

N_X = 100
D_X = 10.0
N_Z = 50
D_Z = 5.0
N_X_PAD = 25
N_Z_PAD = 20
V_P = 2500
V_S = 1500
RHO = 1000

N_T = 1000
D_T = 0.01

DELAY = 2.0
DOM_FREQ = 5.0
# CONFIG_RICKER = {'delay': DELAY, 'dom_freq': DOM_FREQ}

N_SRCS = 8
Z_SHOT = 10.0
N_REC = 10
D_X_REC = 11.0
Z_REC = 5.0

I_GPUS = [0, 1, 2, 3]

#expected values
N_WFLD_COMPONENTS = 5


@pytest.fixture
def ricker_wavelet():
  ricker = Elastic2D.ElasticIsotropicRicker2D(N_T, D_T, DOM_FREQ, DELAY)
  # ricker.make(CONFIG_RICKER)
  return ricker.get_arr()


@pytest.fixture
def src_locations():
  x_locs = np.linspace(0.0, N_X * D_X, num=N_SRCS)
  z_locs = np.ones(N_SRCS) * Z_SHOT
  return np.array([x_locs, z_locs]).T


@pytest.fixture
def fixed_rec_locations():
  rec_x_locs = np.arange(N_REC) * D_X_REC
  rec_z_locs = np.ones(N_REC) * Z_REC
  return np.array([rec_x_locs, rec_z_locs]).T


@pytest.fixture
def variable_rec_locations(src_locations):
  relative_rec_x_locs = np.arange(N_REC) * D_X_REC
  rec_z_locs = np.ones(N_REC) * Z_REC

  rec_locs = []
  for src_location in src_locations:
    rec_x_locs = src_location[0] + relative_rec_x_locs
    rec_locs.append(np.array([rec_x_locs, rec_z_locs]).T)
  return np.array(rec_locs)


@pytest.fixture
def vp_vs_rho_model_2d():
  vp_2d = V_P * np.ones((N_X, N_Z))
  vs_2d = V_S * np.ones((N_X, N_Z))
  rho_2d = RHO * np.ones((N_X, N_Z))

  return np.array((vp_2d, vs_2d, rho_2d))


# mock the make function so we can test all of the individual function calls within make below
def mock_make(self,
              model,
              wavelet,
              d_t,
              src_locations,
              rec_locations,
              gpus,
              recording_components,
              lame_model=False):
  return None


# ###############################################
# #### ElasticIsotropic2D tests ################
# ###############################################
@pytest.mark.gpu
def test_fwd(ricker_wavelet, fixed_rec_locations, src_locations,
             vp_vs_rho_model_2d):
  # Arrange
  elastic_2d = ElasticIsotropic2D(model=vp_vs_rho_model_2d,
                                  model_sampling=(D_X, D_Z),
                                  model_padding=(N_X_PAD, N_Z_PAD),
                                  wavelet=ricker_wavelet,
                                  d_t=D_T,
                                  src_locations=src_locations,
                                  rec_locations=fixed_rec_locations,
                                  gpus=I_GPUS)
  # Act
  data = elastic_2d.fwd(vp_vs_rho_model_2d)

  # Assert
  assert data.shape == (N_SRCS, N_WFLD_COMPONENTS, N_REC, N_T)
  assert not np.all((data == 0))


@pytest.mark.gpu
def test_init(ricker_wavelet, fixed_rec_locations, src_locations,
              vp_vs_rho_model_2d):
  # Arrange
  elastic_2d = ElasticIsotropic2D(model=vp_vs_rho_model_2d,
                                  model_sampling=(D_X, D_Z),
                                  model_padding=(N_X_PAD, N_Z_PAD),
                                  wavelet=ricker_wavelet,
                                  d_t=D_T,
                                  src_locations=src_locations,
                                  rec_locations=fixed_rec_locations,
                                  gpus=I_GPUS)


def test_set_wavelet(ricker_wavelet):
  # Arrange
  with patch.object(ElasticIsotropic2D, "make", mock_make):
    elastic_2d = ElasticIsotropic2D(None, None, None, None, None, None, None)

    # Act
    elastic_2d.set_wavelet(ricker_wavelet, D_T)


def test_set_data(variable_rec_locations, src_locations, vp_vs_rho_model_2d):
  # Arrange
  with patch.object(ElasticIsotropic2D, "make", mock_make):
    elastic_2d = ElasticIsotropic2D(None, (D_X, D_Z), None, None, None, None,
                                    None)
    elastic_2d.set_model(vp_vs_rho_model_2d)
    elastic_2d.set_src_devices(src_locations, N_T)
    elastic_2d.set_rec_devices(variable_rec_locations, N_T)

    # Act
    elastic_2d.set_data(N_T, D_T)

    # Assert
    assert elastic_2d.get_data_sep().getNdArray().shape == (N_SRCS,
                                                            N_WFLD_COMPONENTS,
                                                            N_REC, N_T)


def test_set_data_fails_if_no_src_or_rec(variable_rec_locations, src_locations,
                                         vp_vs_rho_model_2d):
  # Arrange
  with patch.object(ElasticIsotropic2D, "make", mock_make):
    elastic_2d = ElasticIsotropic2D(None, (D_X, D_Z), None, None, None, None,
                                    None)
    elastic_2d.set_model(vp_vs_rho_model_2d)
    with pytest.raises(RuntimeError):
      elastic_2d.set_data(N_T, D_T)

    # Act
    elastic_2d.set_src_devices(src_locations, N_T)

    # Assert
    with pytest.raises(RuntimeError):
      elastic_2d.set_data(N_T, D_T)


def test_set_rec_devices_variable_receivers(variable_rec_locations,
                                            src_locations, vp_vs_rho_model_2d):
  # Arrange
  with patch.object(ElasticIsotropic2D, "make", mock_make):
    elastic_2d = ElasticIsotropic2D(None, (D_X, D_Z), None, None, None, None,
                                    None)
    elastic_2d.set_model(vp_vs_rho_model_2d)
    elastic_2d.set_src_devices(src_locations, N_T)

    # Act
    elastic_2d.set_rec_devices(variable_rec_locations, N_T)

    # Assert
    assert len(elastic_2d.get_rec_devices()) == 4
    for rec_device_grid in elastic_2d.get_rec_devices():
      assert len(rec_device_grid) == N_SRCS


def test_set_rec_devices_variable_receivers_nshot_mismatch(
    variable_rec_locations, src_locations, vp_vs_rho_model_2d):
  # Arrange
  with patch.object(ElasticIsotropic2D, "make", mock_make):
    elastic_2d = ElasticIsotropic2D(None, (D_X, D_Z), None, None, None, None,
                                    None)
    elastic_2d.set_model(vp_vs_rho_model_2d)
    elastic_2d.set_src_devices(src_locations, N_T)

    # Act and Assert
    with pytest.raises(RuntimeError):
      elastic_2d.set_rec_devices(variable_rec_locations[1:], N_T)


def test_set_rec_devices_fixed_receivers(fixed_rec_locations, src_locations,
                                         vp_vs_rho_model_2d):
  # Arrange
  with patch.object(ElasticIsotropic2D, "make", mock_make):
    elastic_2d = ElasticIsotropic2D(None, (D_X, D_Z), None, None, None, None,
                                    None)
    elastic_2d.set_model(vp_vs_rho_model_2d)
    elastic_2d.set_src_devices(src_locations, N_T)

    # Act
    elastic_2d.set_rec_devices(fixed_rec_locations, N_T)

    # Assert
    assert len(elastic_2d.get_rec_devices()) == 4
    for rec_device_grid in elastic_2d.get_rec_devices():
      assert len(rec_device_grid) == N_SRCS


def test_set_rec_devices_fails_if_no_src_locations(fixed_rec_locations,
                                                   vp_vs_rho_model_2d):
  # Arrange
  with patch.object(ElasticIsotropic2D, "make", mock_make):
    elastic_2d = ElasticIsotropic2D(None, (D_X, D_Z), None, None, None, None,
                                    None)
    elastic_2d.set_model(vp_vs_rho_model_2d)
    # Act and Assert
    with pytest.raises(RuntimeError):
      elastic_2d.set_rec_devices(fixed_rec_locations, N_T)


def test_set_rec_devices_fails_if_no_model(fixed_rec_locations):
  # Arrange
  with patch.object(ElasticIsotropic2D, "make", mock_make):
    elastic_2d = ElasticIsotropic2D(None, (D_X, D_Z), None, None, None, None,
                                    None)
    # Act and Assert
    with pytest.raises(RuntimeError):
      elastic_2d.set_rec_devices(fixed_rec_locations, N_T)


def test_set_src_devices(src_locations, vp_vs_rho_model_2d):
  # Arrange
  with patch.object(ElasticIsotropic2D, "make", mock_make):
    elastic_2d = ElasticIsotropic2D(None, (D_X, D_Z), None, None, None, None,
                                    None)
    elastic_2d.set_model(vp_vs_rho_model_2d)

    # Act
    elastic_2d.set_src_devices(src_locations, N_T)

    # Assert
    assert len(elastic_2d.get_src_devices()) == 4
    for src_device_grid in elastic_2d.get_src_devices():
      assert len(src_device_grid) == N_SRCS


def test_set_src_devices_fails_if_no_model(src_locations):
  # Arrange
  with patch.object(ElasticIsotropic2D, "make", mock_make):
    elastic_2d = ElasticIsotropic2D(None, (D_X, D_Z), None, None, None, None,
                                    None)
    # Act and Assert
    with pytest.raises(RuntimeError):
      elastic_2d.set_src_devices(src_locations, N_T)


def test_set_model_default_padding(vp_vs_rho_model_2d):
  # Arrange
  with patch.object(ElasticIsotropic2D, "make", mock_make):
    elastic_2d = ElasticIsotropic2D(None, (D_X, D_Z), None, None, None, None,
                                    None)

    # Act
    elastic_2d.set_model(vp_vs_rho_model_2d)

    # Assert
    default_padding = 50
    assert (elastic_2d.fd_param['n_x'] -
            2 * elastic_2d.fat) % elastic_2d.block_size == 0
    assert (elastic_2d.fd_param['n_z'] -
            2 * elastic_2d.fat) % elastic_2d.block_size == 0
    assert elastic_2d.fd_param['d_x'] == D_X
    assert elastic_2d.fd_param['d_z'] == D_Z
    assert elastic_2d.fd_param['x_pad_minus'] == default_padding
    assert elastic_2d.fd_param['x_pad_plus'] == elastic_2d.fd_param['n_x'] - (
        N_X + default_padding + 2 * elastic_2d.fat)
    assert elastic_2d.fd_param['z_pad_minus'] == default_padding
    assert elastic_2d.fd_param['z_pad_plus'] == elastic_2d.fd_param['n_z'] - (
        N_Z + default_padding + 2 * elastic_2d.fat)

    #check model gets set
    start_x = elastic_2d.fat + default_padding
    end_x = start_x + N_X
    start_z = elastic_2d.fat + default_padding
    end_z = start_z + N_Z

    # check that model was converted to lame paramters
    lame_model_2d = convert_to_lame(vp_vs_rho_model_2d)
    assert np.allclose(
        elastic_2d.get_model_sep().getNdArray()[:, start_x:end_x,
                                                start_z:end_z], lame_model_2d)


def test_set_model(vp_vs_rho_model_2d):
  # Arrange
  with patch.object(ElasticIsotropic2D, "make", mock_make):
    elastic_2d = ElasticIsotropic2D(None, (D_X, D_Z),
                                    None,
                                    None,
                                    None,
                                    None,
                                    None,
                                    model_padding=(N_X_PAD, N_Z_PAD))

    # Act
    elastic_2d.set_model(vp_vs_rho_model_2d)

    # Assert
    assert (elastic_2d.fd_param['n_x'] -
            2 * elastic_2d.fat) % elastic_2d.block_size == 0
    assert (elastic_2d.fd_param['n_z'] -
            2 * elastic_2d.fat) % elastic_2d.block_size == 0
    assert elastic_2d.fd_param['d_x'] == D_X
    assert elastic_2d.fd_param['d_z'] == D_Z
    assert elastic_2d.fd_param['x_pad_minus'] == N_X_PAD
    assert elastic_2d.fd_param['x_pad_plus'] == elastic_2d.fd_param['n_x'] - (
        N_X + N_X_PAD + 2 * elastic_2d.fat)
    assert elastic_2d.fd_param['z_pad_minus'] == N_Z_PAD
    assert elastic_2d.fd_param['z_pad_plus'] == elastic_2d.fd_param['n_z'] - (
        N_Z + N_Z_PAD + 2 * elastic_2d.fat)

    #check model gets set
    start_x = elastic_2d.fat + N_X_PAD
    end_x = start_x + N_X
    start_z = elastic_2d.fat + N_Z_PAD
    end_z = start_z + N_Z

    # check that model was converted to lame paramters
    lame_model_2d = convert_to_lame(vp_vs_rho_model_2d)
    assert np.allclose(
        elastic_2d.get_model_sep().getNdArray()[:, start_x:end_x,
                                                start_z:end_z], lame_model_2d)


# @patch.object(ElasticIsotropic2D, 'make')
def test_pad_model(vp_vs_rho_model_2d):
  # Arrange
  with patch.object(ElasticIsotropic2D, "make", mock_make):
    elastic_2d = ElasticIsotropic2D(None, None, None, None, None, None, None)

    # Act
    model, x_pad, x_pad_plus, new_o_x, z_pad, z_pad_plus, new_o_z = elastic_2d.pad_model(
        vp_vs_rho_model_2d, (D_X, D_Z), (N_X_PAD, N_Z_PAD))

    # Assert
    assert (model.shape[1] - 2 * elastic_2d.fat) % elastic_2d.block_size == 0
    assert (model.shape[2] - 2 * elastic_2d.fat) % elastic_2d.block_size == 0
    assert x_pad == N_X_PAD
    assert x_pad_plus == model.shape[1] - (N_X + N_X_PAD + 2 * elastic_2d.fat)
    assert z_pad == N_Z_PAD
    assert z_pad_plus == model.shape[2] - (N_Z + N_Z_PAD + 2 * elastic_2d.fat)
    start_x = elastic_2d.fat + N_X_PAD
    end_x = start_x + N_X
    start_z = elastic_2d.fat + N_Z_PAD
    end_z = start_z + N_Z
    assert np.allclose(model[:, start_x:end_x, start_z:end_z],
                       vp_vs_rho_model_2d)
