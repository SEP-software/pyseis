from mock import patch
import pytest
import numpy as np
from pyseis.wave_equations import elastic_isotropic, wave_equation
from pyseis.wavelets.elastic import Elastic2D
import SepVector
import pySepVector

N_X = 100
D_X = 10.0
N_Z = 50
D_Z = 5.0
N_X_PAD = 25
N_Z_PAD = 20
V_P1 = 2500
V_P2 = 2750
V_S = 1500
RHO = 1000

N_T = 500
D_T = 0.01

DELAY = 2.0
DOM_FREQ = 5.0
# CONFIG_RICKER = {'delay': DELAY, 'dom_freq': DOM_FREQ}

N_SRCS = 4
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
def trapezoid_wavelet():
  trapezoid = Elastic2D.ElasticIsotropicTrapezoid2D(N_T, D_T, F1, F2, F3, F4,
                                                    DELAY)
  return trapezoid.get_arr()


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
  vp_2d = np.zeros((N_X, N_Z))
  vp_2d[..., :N_Z // 2] = V_P1
  vp_2d[..., N_Z // 2:] = V_P2
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
              model_padding,
              model_origins,
              model_sampling,
              recording_components,
              lame_model=False,
              subsampling=None,
              free_surface=False):
  return None


# ###############################################
# #### ElasticIsotropic2D tests ################
# ###############################################
@pytest.mark.gpu
def test_fwd(ricker_wavelet, fixed_rec_locations, src_locations,
             vp_vs_rho_model_2d):
  # Arrange
  elastic_2d = elastic_isotropic.ElasticIsotropic2D(
      model=vp_vs_rho_model_2d,
      model_sampling=(D_X, D_Z),
      model_padding=(N_X_PAD, N_Z_PAD),
      wavelet=ricker_wavelet,
      d_t=D_T,
      src_locations=src_locations,
      rec_locations=fixed_rec_locations,
      gpus=I_GPUS)
  # Act
  data = elastic_2d.forward(vp_vs_rho_model_2d)

  # Assert
  assert data.shape == (N_SRCS, N_WFLD_COMPONENTS, N_REC, N_T)
  assert not np.all((data == 0))


@pytest.mark.gpu
def test_fwd_vel_recording_components(ricker_wavelet, fixed_rec_locations,
                                      src_locations, vp_vs_rho_model_2d):
  # Arrange
  recording_components = ['vx', 'vz']
  elastic_2d = elastic_isotropic.ElasticIsotropic2D(
      model=vp_vs_rho_model_2d,
      model_sampling=(D_X, D_Z),
      model_padding=(N_X_PAD, N_Z_PAD),
      wavelet=ricker_wavelet,
      d_t=D_T,
      src_locations=src_locations,
      rec_locations=fixed_rec_locations,
      gpus=I_GPUS,
      recording_components=recording_components)
  # Act
  data = elastic_2d.forward(vp_vs_rho_model_2d)

  # Assert
  assert data.shape == (N_SRCS, len(recording_components), N_REC, N_T)
  assert not np.all((data == 0))


@pytest.mark.gpu
def test_fwd_p_recording_components(ricker_wavelet, fixed_rec_locations,
                                    src_locations, vp_vs_rho_model_2d):
  # Arrange
  recording_components = ['p']
  elastic_2d = elastic_isotropic.ElasticIsotropic2D(
      model=vp_vs_rho_model_2d,
      model_sampling=(D_X, D_Z),
      model_padding=(N_X_PAD, N_Z_PAD),
      wavelet=ricker_wavelet,
      d_t=D_T,
      src_locations=src_locations,
      rec_locations=fixed_rec_locations,
      gpus=I_GPUS,
      recording_components=recording_components)
  # Act
  data = elastic_2d.forward(vp_vs_rho_model_2d)

  # Assert
  assert data.shape == (N_SRCS, len(recording_components), N_REC, N_T)
  assert not np.all((data == 0))


@pytest.mark.gpu
def test_init(ricker_wavelet, fixed_rec_locations, src_locations,
              vp_vs_rho_model_2d):
  # Arrange
  elastic_2d = elastic_isotropic.ElasticIsotropic2D(
      model=vp_vs_rho_model_2d,
      model_sampling=(D_X, D_Z),
      model_padding=(N_X_PAD, N_Z_PAD),
      wavelet=ricker_wavelet,
      d_t=D_T,
      src_locations=src_locations,
      rec_locations=fixed_rec_locations,
      gpus=I_GPUS)


@pytest.mark.gpu
def test_init_with_wrong_sub(ricker_wavelet, fixed_rec_locations, src_locations,
                             vp_vs_rho_model_2d):
  # Act
  with pytest.raises(RuntimeError):
    elastic_2d = elastic_isotropic.ElasticIsotropic2D(
        model=vp_vs_rho_model_2d,
        model_sampling=(D_X, D_Z),
        model_padding=(N_X_PAD, N_Z_PAD),
        wavelet=ricker_wavelet,
        d_t=D_T,
        src_locations=src_locations,
        rec_locations=fixed_rec_locations,
        gpus=I_GPUS,
        subsampling=1)


def test_setup_wavelet(ricker_wavelet):
  # Arrange
  with patch.object(elastic_isotropic.ElasticIsotropic2D, "_make", mock_make):
    elastic_2d = elastic_isotropic.ElasticIsotropic2D(None, (D_X, D_Z), None,
                                                      None, None, None, None)

    # Act
    wavelet_nl_sep, wavelet_lin_sep = elastic_2d._setup_wavelet(
        ricker_wavelet, D_T)

    # assert
    assert isinstance(wavelet_nl_sep, SepVector.floatVector)
    assert isinstance(wavelet_lin_sep, list)
    assert isinstance(wavelet_lin_sep[0], pySepVector.float3DReg)


def test_setup_data(variable_rec_locations, src_locations, vp_vs_rho_model_2d):
  # Arrange
  with patch.object(elastic_isotropic.ElasticIsotropic2D, "_make", mock_make):
    elastic_2d = elastic_isotropic.ElasticIsotropic2D(None, (D_X, D_Z), None,
                                                      None, None, None, None)

    elastic_2d.model_sep, elastic_2d.model_padding = elastic_2d._setup_model(
        vp_vs_rho_model_2d,
        model_padding=(N_X_PAD, N_Z_PAD),
        model_sampling=(D_X, D_Z))
    elastic_2d._setup_src_devices(src_locations, N_T)
    elastic_2d._setup_rec_devices(variable_rec_locations, N_T)

    # Act
    data_sep = elastic_2d._setup_data(N_T, D_T)

    # Assert
    assert data_sep.getNdArray().shape == (N_SRCS, N_WFLD_COMPONENTS, N_REC,
                                           N_T)


def test_setup_data_fails_if_no_src_or_rec(variable_rec_locations,
                                           src_locations, vp_vs_rho_model_2d):
  # Arrange
  with patch.object(elastic_isotropic.ElasticIsotropic2D, "_make", mock_make):
    elastic_2d = elastic_isotropic.ElasticIsotropic2D(None, (D_X, D_Z), None,
                                                      None, None, None, None)
    elastic_2d.model_sep, elastic_2d.model_padding = elastic_2d._setup_model(
        vp_vs_rho_model_2d,
        model_padding=(N_X_PAD, N_Z_PAD),
        model_sampling=(D_X, D_Z))
    with pytest.raises(RuntimeError):
      elastic_2d._setup_data(N_T, D_T)

    # Act
    elastic_2d._setup_src_devices(src_locations, N_T)

    # Assert
    with pytest.raises(RuntimeError):
      elastic_2d._setup_data(N_T, D_T)


def test_setup_rec_devices_variable_receivers(variable_rec_locations,
                                              src_locations,
                                              vp_vs_rho_model_2d):
  # Arrange
  with patch.object(elastic_isotropic.ElasticIsotropic2D, "_make", mock_make):
    elastic_2d = elastic_isotropic.ElasticIsotropic2D(None, (D_X, D_Z), None,
                                                      None, None, None, None)
    elastic_2d.model_sep, elastic_2d.model_padding = elastic_2d._setup_model(
        vp_vs_rho_model_2d,
        model_padding=(N_X_PAD, N_Z_PAD),
        model_sampling=(D_X, D_Z))
    elastic_2d._setup_src_devices(src_locations, N_T)

    # Act
    rec_devices = elastic_2d._setup_rec_devices(variable_rec_locations, N_T)

    # Assert
    assert len(rec_devices) == 4
    for rec_device_grid in rec_devices:
      assert len(rec_device_grid) == N_SRCS


def test_setup_rec_devices_variable_receivers_nshot_mismatch(
    variable_rec_locations, src_locations, vp_vs_rho_model_2d):
  # Arrange
  with patch.object(elastic_isotropic.ElasticIsotropic2D, "_make", mock_make):
    elastic_2d = elastic_isotropic.ElasticIsotropic2D(None, (D_X, D_Z), None,
                                                      None, None, None, None)
    elastic_2d.model_sep, elastic_2d.model_padding = elastic_2d._setup_model(
        vp_vs_rho_model_2d,
        model_padding=(N_X_PAD, N_Z_PAD),
        model_sampling=(D_X, D_Z))
    elastic_2d._setup_src_devices(src_locations, N_T)

    # Act and Assert
    with pytest.raises(RuntimeError):
      elastic_2d._setup_rec_devices(variable_rec_locations[1:], N_T)


def test_setup_rec_devices_fixed_receivers(fixed_rec_locations, src_locations,
                                           vp_vs_rho_model_2d):
  # Arrange
  with patch.object(elastic_isotropic.ElasticIsotropic2D, "_make", mock_make):
    elastic_2d = elastic_isotropic.ElasticIsotropic2D(None, (D_X, D_Z), None,
                                                      None, None, None, None)
    elastic_2d.model_sep, elastic_2d.model_padding = elastic_2d._setup_model(
        vp_vs_rho_model_2d,
        model_padding=(N_X_PAD, N_Z_PAD),
        model_sampling=(D_X, D_Z))
    elastic_2d._setup_src_devices(src_locations, N_T)

    # Act
    rec_devices = elastic_2d._setup_rec_devices(fixed_rec_locations, N_T)

    # Assert
    assert len(rec_devices) == 4
    for rec_device_grid in rec_devices:
      assert len(rec_device_grid) == N_SRCS


def test_setup_rec_devices_fails_if_no_src_locations(fixed_rec_locations,
                                                     vp_vs_rho_model_2d):
  # Arrange
  with patch.object(elastic_isotropic.ElasticIsotropic2D, "_make", mock_make):
    elastic_2d = elastic_isotropic.ElasticIsotropic2D(None, (D_X, D_Z), None,
                                                      None, None, None, None)
    elastic_2d.model_sep, elastic_2d.model_padding = elastic_2d._setup_model(
        vp_vs_rho_model_2d,
        model_padding=(N_X_PAD, N_Z_PAD),
        model_sampling=(D_X, D_Z))
    # Act and Assert
    with pytest.raises(RuntimeError):
      elastic_2d._setup_rec_devices(fixed_rec_locations, N_T)


def test_setup_rec_devices_fails_if_no_model(fixed_rec_locations):
  # Arrange
  with patch.object(elastic_isotropic.ElasticIsotropic2D, "_make", mock_make):
    elastic_2d = elastic_isotropic.ElasticIsotropic2D(None, (D_X, D_Z), None,
                                                      None, None, None, None)
    # Act and Assert
    with pytest.raises(RuntimeError):
      elastic_2d._setup_rec_devices(fixed_rec_locations, N_T)


def test_setup_src_devices(src_locations, vp_vs_rho_model_2d):
  # Arrange
  with patch.object(elastic_isotropic.ElasticIsotropic2D, "_make", mock_make):
    elastic_2d = elastic_isotropic.ElasticIsotropic2D(None, (D_X, D_Z), None,
                                                      None, None, None, None)
    elastic_2d.model_sep, elastic_2d.model_padding = elastic_2d._setup_model(
        vp_vs_rho_model_2d,
        model_padding=(N_X_PAD, N_Z_PAD),
        model_sampling=(D_X, D_Z))

    # Act
    src_devices = elastic_2d._setup_src_devices(src_locations, N_T)

    # Assert
    assert len(src_devices) == 4
    for src_device_grid in src_devices:
      assert len(src_device_grid) == N_SRCS


def test_setup_src_devices_fails_if_no_model(src_locations):
  # Arrange
  with patch.object(elastic_isotropic.ElasticIsotropic2D, "_make", mock_make):
    elastic_2d = elastic_isotropic.ElasticIsotropic2D(None, (D_X, D_Z), None,
                                                      None, None, None, None)
    # Act and Assert
    with pytest.raises(RuntimeError):
      elastic_2d._setup_src_devices(src_locations, N_T)


def test_setup_model(vp_vs_rho_model_2d):
  # Arrange
  with patch.object(elastic_isotropic.ElasticIsotropic2D, "_make", mock_make):
    elastic_2d = elastic_isotropic.ElasticIsotropic2D(None, (D_X, D_Z),
                                                      None,
                                                      None,
                                                      None,
                                                      None,
                                                      None,
                                                      model_padding=(N_X_PAD,
                                                                     N_Z_PAD))

    # Act
    elastic_2d.model_sep, elastic_2d.model_padding = elastic_2d._setup_model(
        vp_vs_rho_model_2d,
        model_padding=(N_X_PAD, N_Z_PAD),
        model_sampling=(D_X, D_Z))

    # Assert
    assert (elastic_2d.fd_param['n_x'] -
            2 * elastic_2d._FAT) % elastic_2d._BLOCK_SIZE == 0
    assert (elastic_2d.fd_param['n_z'] -
            2 * elastic_2d._FAT) % elastic_2d._BLOCK_SIZE == 0
    assert elastic_2d.fd_param['d_x'] == D_X
    assert elastic_2d.fd_param['d_z'] == D_Z
    assert elastic_2d.fd_param['x_pad_minus'] == N_X_PAD
    assert elastic_2d.fd_param['x_pad_plus'] == elastic_2d.fd_param['n_x'] - (
        N_X + N_X_PAD + 2 * elastic_2d._FAT)
    assert elastic_2d.fd_param['z_pad_minus'] == N_Z_PAD
    assert elastic_2d.fd_param['z_pad_plus'] == elastic_2d.fd_param['n_z'] - (
        N_Z + N_Z_PAD + 2 * elastic_2d._FAT)

    #check model gets set
    start_x = elastic_2d._FAT + N_X_PAD
    end_x = start_x + N_X
    start_z = elastic_2d._FAT + N_Z_PAD
    end_z = start_z + N_Z

    # check that model was converted to lame paramters
    # lame_model_2d = elastic_isotropic.convert_to_lame(vp_vs_rho_model_2d)
    assert np.allclose(
        elastic_2d.model_sep.getNdArray()[:, start_x:end_x, start_z:end_z],
        vp_vs_rho_model_2d)


# @patch.object(elastic_isotropic.ElasticIsotropic2D, 'make')
def test_pad_model(vp_vs_rho_model_2d):
  # Arrange
  with patch.object(elastic_isotropic.ElasticIsotropic2D, "_make", mock_make):
    elastic_2d = elastic_isotropic.ElasticIsotropic2D(None, (D_X, D_Z), None,
                                                      None, None, None, None)
    padding, new_shape, new_origins = elastic_2d._calc_pad_params(
        (N_X, N_Z), (N_X_PAD, N_Z_PAD), (D_X, D_Z), elastic_2d._FAT)
    # Act
    model = elastic_2d._pad_model(vp_vs_rho_model_2d, padding, elastic_2d._FAT)

    # Assert
    assert (model.shape[1] - 2 * elastic_2d._FAT) % elastic_2d._BLOCK_SIZE == 0
    assert (model.shape[2] - 2 * elastic_2d._FAT) % elastic_2d._BLOCK_SIZE == 0
    assert padding[0][0] == N_X_PAD
    assert padding[0][1] == model.shape[1] - (N_X + N_X_PAD +
                                              2 * elastic_2d._FAT)
    assert padding[1][0] == N_Z_PAD
    assert padding[1][1] == model.shape[2] - (N_Z + N_Z_PAD +
                                              2 * elastic_2d._FAT)
    start_x = elastic_2d._FAT + N_X_PAD
    end_x = start_x + N_X
    start_z = elastic_2d._FAT + N_Z_PAD
    end_z = start_z + N_Z
    assert np.allclose(model[:, start_x:end_x, start_z:end_z],
                       vp_vs_rho_model_2d)


###############################################
#### ElasticIsotropic2D Born tests ###########
###############################################
@pytest.mark.gpu
def test_jacobian_none_on_init(ricker_wavelet, fixed_rec_locations,
                               src_locations, vp_vs_rho_model_2d):
  # Arrange and act
  elastic_2d = elastic_isotropic.ElasticIsotropic2D(
      model=vp_vs_rho_model_2d,
      model_sampling=(D_X, D_Z),
      model_padding=(N_X_PAD, N_Z_PAD),
      wavelet=ricker_wavelet,
      d_t=D_T,
      src_locations=src_locations,
      rec_locations=fixed_rec_locations,
      gpus=I_GPUS)
  # assert
  assert elastic_2d._jac_operator == None


@pytest.mark.gpu
def test_jacobian(ricker_wavelet, fixed_rec_locations, src_locations,
                  vp_vs_rho_model_2d):
  # Arrange
  elastic_2d = elastic_isotropic.ElasticIsotropic2D(
      model=vp_vs_rho_model_2d,
      model_sampling=(D_X, D_Z),
      model_padding=(N_X_PAD, N_Z_PAD),
      wavelet=ricker_wavelet,
      d_t=D_T,
      src_locations=src_locations,
      rec_locations=fixed_rec_locations,
      gpus=I_GPUS)
  #reflectivity model
  lin_model = np.gradient(vp_vs_rho_model_2d, axis=-1)

  # act
  lin_data = elastic_2d.jacobian(lin_model)

  # Assert
  assert lin_data.shape == (N_SRCS, N_WFLD_COMPONENTS, N_REC, N_T)
  assert not np.all((lin_data == 0))
  assert not np.any(np.isnan(lin_data))


@pytest.mark.gpu
def test_jacobian_after_forward(ricker_wavelet, fixed_rec_locations,
                                src_locations, vp_vs_rho_model_2d):
  # Arrange
  elastic_2d = elastic_isotropic.ElasticIsotropic2D(
      model=vp_vs_rho_model_2d,
      model_sampling=(D_X, D_Z),
      model_padding=(N_X_PAD, N_Z_PAD),
      wavelet=ricker_wavelet,
      d_t=D_T,
      src_locations=src_locations,
      rec_locations=fixed_rec_locations,
      gpus=I_GPUS)
  #run forward
  data = elastic_2d.forward(vp_vs_rho_model_2d)
  #reflectivity model
  lin_model = np.gradient(vp_vs_rho_model_2d, axis=-1)

  # act
  lin_data = elastic_2d.jacobian(lin_model)

  # Assert
  assert lin_data.shape == (N_SRCS, N_WFLD_COMPONENTS, N_REC, N_T)
  assert not np.all((lin_data == 0))
  assert not np.any(np.isnan(lin_data))


@pytest.mark.gpu
def test_jacobian_vel_recording_components(ricker_wavelet, fixed_rec_locations,
                                           src_locations, vp_vs_rho_model_2d):
  # Arrange
  recording_components = ['vx', 'vz']
  elastic_2d = elastic_isotropic.ElasticIsotropic2D(
      model=vp_vs_rho_model_2d,
      model_sampling=(D_X, D_Z),
      model_padding=(N_X_PAD, N_Z_PAD),
      wavelet=ricker_wavelet,
      d_t=D_T,
      src_locations=src_locations,
      rec_locations=fixed_rec_locations,
      gpus=I_GPUS,
      recording_components=recording_components)
  #reflectivity model
  lin_model = np.gradient(vp_vs_rho_model_2d, axis=-1)

  # act
  lin_data = elastic_2d.jacobian(lin_model)

  # Assert
  assert lin_data.shape == (N_SRCS, len(recording_components), N_REC, N_T)
  assert not np.all((lin_data == 0))
  assert not np.any(np.isnan(lin_data))


@pytest.mark.gpu
def test_jacobian_pressure_recording_components(ricker_wavelet,
                                                fixed_rec_locations,
                                                src_locations,
                                                vp_vs_rho_model_2d):
  # Arrange
  recording_components = ['p']
  elastic_2d = elastic_isotropic.ElasticIsotropic2D(
      model=vp_vs_rho_model_2d,
      model_sampling=(D_X, D_Z),
      model_padding=(N_X_PAD, N_Z_PAD),
      wavelet=ricker_wavelet,
      d_t=D_T,
      src_locations=src_locations,
      rec_locations=fixed_rec_locations,
      gpus=I_GPUS,
      recording_components=recording_components)
  #reflectivity model
  lin_model = np.gradient(vp_vs_rho_model_2d, axis=-1)

  # act
  lin_data = elastic_2d.jacobian(lin_model)

  # Assert
  assert lin_data.shape == (N_SRCS, len(recording_components), N_REC, N_T)
  assert not np.all((lin_data == 0))
  assert not np.any(np.isnan(lin_data))


@pytest.mark.gpu
def test_jacobian_adjoint(ricker_wavelet, fixed_rec_locations, src_locations,
                          vp_vs_rho_model_2d):
  # Arrange
  elastic_2d = elastic_isotropic.ElasticIsotropic2D(
      model=vp_vs_rho_model_2d,
      model_sampling=(D_X, D_Z),
      model_padding=(N_X_PAD, N_Z_PAD),
      wavelet=ricker_wavelet,
      d_t=D_T,
      src_locations=src_locations,
      rec_locations=fixed_rec_locations,
      gpus=I_GPUS)
  #reflectivity model
  lin_model = np.gradient(vp_vs_rho_model_2d, axis=-1)

  # act
  lin_data = elastic_2d.jacobian(lin_model)
  lin_model = elastic_2d.jacobian_adjoint(lin_data)

  # Assert
  assert lin_model.shape == vp_vs_rho_model_2d.shape
  assert not np.all((lin_model == 0))
  assert not np.any(np.isnan(lin_model))


@pytest.mark.gpu
def test_jacobian_dot_product(ricker_wavelet, fixed_rec_locations,
                              src_locations, vp_vs_rho_model_2d):
  # Arrange
  vp_vs_rho_model_2d_smooth = np.ones_like(vp_vs_rho_model_2d)
  vp_vs_rho_model_2d_smooth[:] = vp_vs_rho_model_2d
  vp_vs_rho_model_2d_smooth[0] = V_P1
  elastic_2d = elastic_isotropic.ElasticIsotropic2D(
      model=vp_vs_rho_model_2d_smooth,
      model_sampling=(D_X, D_Z),
      model_padding=(N_X_PAD, N_Z_PAD),
      wavelet=ricker_wavelet,
      d_t=D_T,
      src_locations=src_locations,
      rec_locations=fixed_rec_locations,
      gpus=I_GPUS)

  #assert
  elastic_2d.dot_product_test(True)


@pytest.mark.gpu
def test_jacobian_dot_product_vel_recording_components(ricker_wavelet,
                                                       fixed_rec_locations,
                                                       src_locations,
                                                       vp_vs_rho_model_2d):
  # Arrange
  recording_components = ['vx', 'vz']
  vp_vs_rho_model_2d_smooth = np.ones_like(vp_vs_rho_model_2d)
  vp_vs_rho_model_2d_smooth[:] = vp_vs_rho_model_2d
  vp_vs_rho_model_2d_smooth[0] = V_P1
  elastic_2d = elastic_isotropic.ElasticIsotropic2D(
      model=vp_vs_rho_model_2d_smooth,
      model_sampling=(D_X, D_Z),
      model_padding=(N_X_PAD, N_Z_PAD),
      wavelet=ricker_wavelet,
      d_t=D_T,
      src_locations=src_locations,
      rec_locations=fixed_rec_locations,
      gpus=I_GPUS,
      recording_components=recording_components)

  #assert
  elastic_2d.dot_product_test(True)


@pytest.mark.gpu
def test_jacobian_dot_product_pressure_recording_components(
    ricker_wavelet, fixed_rec_locations, src_locations, vp_vs_rho_model_2d):
  # Arrange
  recording_components = ['p']
  vp_vs_rho_model_2d_smooth = np.ones_like(vp_vs_rho_model_2d)
  vp_vs_rho_model_2d_smooth[:] = vp_vs_rho_model_2d
  vp_vs_rho_model_2d_smooth[0] = V_P1
  elastic_2d = elastic_isotropic.ElasticIsotropic2D(
      model=vp_vs_rho_model_2d_smooth,
      model_sampling=(D_X, D_Z),
      model_padding=(N_X_PAD, N_Z_PAD),
      wavelet=ricker_wavelet,
      d_t=D_T,
      src_locations=src_locations,
      rec_locations=fixed_rec_locations,
      gpus=I_GPUS,
      recording_components=recording_components)

  #assert
  elastic_2d.dot_product_test(True)


@pytest.mark.gpu
def test_setup_fwi_operators(ricker_wavelet, fixed_rec_locations, src_locations,
                             vp_vs_rho_model_2d):
  # Arrange
  recording_components = ['p']
  vp_vs_rho_model_2d_smooth = np.ones_like(vp_vs_rho_model_2d)
  vp_vs_rho_model_2d_smooth[:] = vp_vs_rho_model_2d
  vp_vs_rho_model_2d_smooth[0] = V_P1
  elastic_2d = elastic_isotropic.ElasticIsotropic2D(
      model=vp_vs_rho_model_2d_smooth,
      model_sampling=(D_X, D_Z),
      model_padding=(N_X_PAD, N_Z_PAD),
      wavelet=ricker_wavelet,
      d_t=D_T,
      src_locations=src_locations,
      rec_locations=fixed_rec_locations,
      gpus=I_GPUS,
      recording_components=recording_components)

  # Act
  fwi_op = elastic_2d._setup_fwi_op()

  # Assert
  assert isinstance(fwi_op.lin_op.args[1].args[0],
                    wave_equation._JacobianWaveCppOp)
  assert isinstance(fwi_op.nl_op.args[1].args[0],
                    wave_equation._NonlinearWaveCppOp)
