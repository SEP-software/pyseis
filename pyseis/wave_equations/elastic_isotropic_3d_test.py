from mock import patch
import pytest
import numpy as np
from pyseis.wave_equations import elastic_isotropic, wave_equation
from pyseis.wavelets.elastic import Elastic3D
import SepVector
import pySepVector
import time
import sys

N_Y = 51
D_Y = 9.0
N_X = 52
D_X = 10.0
N_Z = 53
D_Z = 11.0
N_Y_PAD = 20
N_X_PAD = 21
N_Z_PAD = 22
V_P = 2500
V_S = 1000
RHO = 2000

N_T = 251
D_T = 0.004

DELAY = 2.0
DOM_FREQ = 5.0
F1 = 0.0
F2 = 15
F3 = 15
F4 = 30

N_SRCS = 4
Z_SHOT = 10.0
N_REC = 7
D_X_REC = 2.0
D_Y_REC = 1.0
Z_REC = 15.0

I_GPUS = [0, 1, 2, 3]
# I_GPUS = [3]

#expected values
N_WFLD_COMPONENTS = 9


@pytest.fixture
def ricker_wavelet():
  ricker = Elastic3D.ElasticIsotropicRicker3D(N_T, D_T, DOM_FREQ, DELAY)
  return ricker.get_arr()


@pytest.fixture
def trapezoid_wavelet():
  trapezoid = Elastic3D.ElasticIsotropicTrapezoid3D(N_T, D_T, F1, F2, F3, F4,
                                                    DELAY)
  return trapezoid.get_arr()


@pytest.fixture
def src_locations():
  y_locs = np.linspace(N_Y * D_Y * 0.25,
                       N_Y * D_Y * 0.75,
                       num=N_SRCS,
                       endpoint=False)
  x_locs = np.linspace(N_X * D_X * 0.25,
                       N_X * D_X * 0.75,
                       num=N_SRCS,
                       endpoint=False)
  z_locs = np.ones(N_SRCS) * Z_SHOT

  return np.array([y_locs, x_locs, z_locs]).T


@pytest.fixture
def fixed_rec_locations():
  rec_y_locs = np.arange(N_REC) * D_Y_REC
  rec_x_locs = np.arange(N_REC) * D_X_REC
  xx, yy = np.meshgrid(rec_x_locs, rec_y_locs, indexing='ij')
  rec_z_locs = np.ones(N_REC * N_REC) * Z_REC
  return np.array([yy.flatten(), xx.flatten(), rec_z_locs]).T


@pytest.fixture
def variable_rec_locations(src_locations):
  relative_rec_y_locs = np.arange(N_REC) * D_Y_REC
  relative_rec_x_locs = np.arange(N_REC) * D_X_REC
  xx, yy = np.meshgrid(relative_rec_x_locs, relative_rec_y_locs, indexing='ij')
  rec_z_locs = np.ones(N_REC * N_REC) * Z_REC

  rec_locs = []
  for src_location in src_locations:
    rec_y_locs = src_location[0] + xx.flatten()
    rec_x_locs = src_location[1] + yy.flatten()
    rec_locs.append(np.array([rec_y_locs, rec_x_locs, rec_z_locs]).T)
  return np.array(rec_locs)


@pytest.fixture
def vp_vs_rho_model_3d():
  vp_3d = V_P * np.ones((N_Y, N_X, N_Z))
  vs_3d = V_S * np.ones((N_Y, N_X, N_Z))
  rho_3d = RHO * np.ones((N_Y, N_X, N_Z))

  return np.array((vp_3d, vs_3d, rho_3d))


# mock the make function so we can test all of the individual function calls within make below
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


###############################################
##### elastic_isotropic.ElasticIsotropic3D tests ################
###############################################
@pytest.mark.gpu
def test_fwd(trapezoid_wavelet, fixed_rec_locations, src_locations,
             vp_vs_rho_model_3d):
  # Arrange
  elastic_3d = elastic_isotropic.ElasticIsotropic3D(
      model=vp_vs_rho_model_3d,
      model_sampling=(D_Y, D_X, D_Z),
      model_padding=(N_Y_PAD, N_X_PAD, N_Z_PAD),
      wavelet=trapezoid_wavelet,
      d_t=D_T,
      src_locations=src_locations,
      rec_locations=fixed_rec_locations,
      gpus=I_GPUS)

  # Act
  data = elastic_3d.forward(vp_vs_rho_model_3d)

  # Assert
  assert data.shape == (N_SRCS, N_WFLD_COMPONENTS, N_REC * N_REC, N_T)


@pytest.mark.gpu
def test_fwd_vel_recording_components(trapezoid_wavelet, fixed_rec_locations,
                                      src_locations, vp_vs_rho_model_3d):
  # Arrange
  recording_components = ['vy', 'vx', 'vz']
  elastic_3d = elastic_isotropic.ElasticIsotropic3D(
      model=vp_vs_rho_model_3d,
      model_sampling=(D_Y, D_X, D_Z),
      model_padding=(N_Y_PAD, N_X_PAD, N_Z_PAD),
      wavelet=trapezoid_wavelet,
      d_t=D_T,
      src_locations=src_locations,
      rec_locations=fixed_rec_locations,
      gpus=I_GPUS,
      recording_components=recording_components)

  # Act
  data = elastic_3d.forward(vp_vs_rho_model_3d)

  # Assert
  assert data.shape == (N_SRCS, len(recording_components), N_REC * N_REC, N_T)


@pytest.mark.gpu
def test_fwd_pressure_recording_components(trapezoid_wavelet,
                                           fixed_rec_locations, src_locations,
                                           vp_vs_rho_model_3d):
  # Arrange
  recording_components = ['p']
  elastic_3d = elastic_isotropic.ElasticIsotropic3D(
      model=vp_vs_rho_model_3d,
      model_sampling=(D_Y, D_X, D_Z),
      model_padding=(N_Y_PAD, N_X_PAD, N_Z_PAD),
      wavelet=trapezoid_wavelet,
      d_t=D_T,
      src_locations=src_locations,
      rec_locations=fixed_rec_locations,
      gpus=I_GPUS,
      recording_components=recording_components)

  # Act
  data = elastic_3d.forward(vp_vs_rho_model_3d)

  # Assert
  assert data.shape == (N_SRCS, len(recording_components), N_REC * N_REC, N_T)


@pytest.mark.gpu
def test_init(trapezoid_wavelet, fixed_rec_locations, src_locations,
              vp_vs_rho_model_3d):
  # Act
  elastic_3d = elastic_isotropic.ElasticIsotropic3D(
      model=vp_vs_rho_model_3d,
      model_sampling=(D_Y, D_X, D_Z),
      model_padding=(N_Y_PAD, N_X_PAD, N_Z_PAD),
      wavelet=trapezoid_wavelet,
      d_t=D_T,
      src_locations=src_locations,
      rec_locations=fixed_rec_locations,
      gpus=I_GPUS)


@pytest.mark.gpu
def test_init_with_wrong_sub(trapezoid_wavelet, fixed_rec_locations,
                             src_locations, vp_vs_rho_model_3d):
  # Act
  with pytest.raises(RuntimeError):
    elastic_3d = elastic_isotropic.ElasticIsotropic3D(
        model=vp_vs_rho_model_3d,
        model_sampling=(D_Y, D_X, D_Z),
        model_padding=(N_Y_PAD, N_X_PAD, N_Z_PAD),
        wavelet=trapezoid_wavelet,
        d_t=D_T,
        src_locations=src_locations,
        rec_locations=fixed_rec_locations,
        gpus=I_GPUS,
        subsampling=1)


def test_setup_wavelet(trapezoid_wavelet):
  # Arrange
  with patch.object(elastic_isotropic.ElasticIsotropic3D, "_make", mock_make):
    elastic_3d = elastic_isotropic.ElasticIsotropic3D(None, (D_Y, D_X, D_Z),
                                                      None, None, None, None,
                                                      None)
    # Act
    wavelet_nl_sep, wavelet_lin_sep = elastic_3d._setup_wavelet(
        trapezoid_wavelet, D_T)

    # assert
    assert isinstance(wavelet_nl_sep, SepVector.floatVector)
    assert isinstance(wavelet_lin_sep, list)
    assert isinstance(wavelet_lin_sep[0], pySepVector.float3DReg)


def test_setup_data(variable_rec_locations, src_locations, vp_vs_rho_model_3d):
  # Arrange
  with patch.object(elastic_isotropic.ElasticIsotropic3D, "_make", mock_make):
    elastic_3d = elastic_isotropic.ElasticIsotropic3D(None, (D_Y, D_X, D_Z),
                                                      None, None, None, None,
                                                      None)

    elastic_3d.model_sep, elastic_3d.model_padding = elastic_3d._setup_model(
        vp_vs_rho_model_3d,
        model_padding=(N_Y_PAD, N_X_PAD, N_Z_PAD),
        model_sampling=(D_Y, D_X, D_Z))
    elastic_3d._setup_src_devices(src_locations, N_T)
    elastic_3d._setup_rec_devices(variable_rec_locations, N_T)

    # Act
    data_sep = elastic_3d._setup_data(N_T, D_T)

    # Assert
    assert data_sep.getNdArray().shape == (N_SRCS, N_WFLD_COMPONENTS,
                                           N_REC * N_REC, N_T)


def test_setup_data_fails_if_no_src_or_rec(variable_rec_locations,
                                           src_locations, vp_vs_rho_model_3d):
  # Arrange
  with patch.object(elastic_isotropic.ElasticIsotropic3D, "_make", mock_make):
    elastic_3d = elastic_isotropic.ElasticIsotropic3D(None, (D_Y, D_X, D_Z),
                                                      None, None, None, None,
                                                      None)
    elastic_3d.model_sep, elastic_3d.model_padding = elastic_3d._setup_model(
        vp_vs_rho_model_3d,
        model_padding=(N_Y_PAD, N_X_PAD, N_Z_PAD),
        model_sampling=(D_Y, D_X, D_Z))

    # Act and Assert
    with pytest.raises(RuntimeError):
      elastic_3d._setup_data(N_T, D_T)

    elastic_3d._setup_src_devices(src_locations, N_T)
    # Act and Assert
    with pytest.raises(RuntimeError):
      elastic_3d._setup_data(N_T, D_T)


def test_setup_rec_devices_variable_receivers(variable_rec_locations,
                                              src_locations,
                                              vp_vs_rho_model_3d):
  # Arrange
  with patch.object(elastic_isotropic.ElasticIsotropic3D, "_make", mock_make):
    elastic_3d = elastic_isotropic.ElasticIsotropic3D(None, (D_Y, D_X, D_Z),
                                                      None, None, None, None,
                                                      None)
    elastic_3d.model_sep, elastic_3d.model_padding = elastic_3d._setup_model(
        vp_vs_rho_model_3d,
        model_padding=(N_Y_PAD, N_X_PAD, N_Z_PAD),
        model_sampling=(D_Y, D_X, D_Z))
    elastic_3d._setup_src_devices(src_locations, N_T)
    # Act
    rec_devices = elastic_3d._setup_rec_devices(variable_rec_locations, N_T)

    # Assert
    assert len(rec_devices) == 7
    for rec_device_grid in rec_devices:
      assert len(rec_device_grid) == N_SRCS


def test_setup_rec_devices_variable_receivers_nshot_mismatch(
    variable_rec_locations, src_locations, vp_vs_rho_model_3d):
  # Arrange
  with patch.object(elastic_isotropic.ElasticIsotropic3D, "_make", mock_make):
    elastic_3d = elastic_isotropic.ElasticIsotropic3D(None, (D_Y, D_X, D_Z),
                                                      None, None, None, None,
                                                      None)
    elastic_3d.model_sep, elastic_3d.model_padding = elastic_3d._setup_model(
        vp_vs_rho_model_3d,
        model_padding=(N_Y_PAD, N_X_PAD, N_Z_PAD),
        model_sampling=(D_Y, D_X, D_Z))
    elastic_3d._setup_src_devices(src_locations, N_T)

    # Act and Assert
    with pytest.raises(RuntimeError):
      elastic_3d._setup_rec_devices(variable_rec_locations[1:], N_T)


def test_setup_rec_devices_fixed_receivers(fixed_rec_locations, src_locations,
                                           vp_vs_rho_model_3d):
  # Arrange
  with patch.object(elastic_isotropic.ElasticIsotropic3D, "_make", mock_make):
    elastic_3d = elastic_isotropic.ElasticIsotropic3D(None, (D_Y, D_X, D_Z),
                                                      None, None, None, None,
                                                      None)
    elastic_3d.model_sep, elastic_3d.model_padding = elastic_3d._setup_model(
        vp_vs_rho_model_3d,
        model_padding=(N_Y_PAD, N_X_PAD, N_Z_PAD),
        model_sampling=(D_Y, D_X, D_Z))
    elastic_3d._setup_src_devices(src_locations, N_T)

    # Act
    rec_devices = elastic_3d._setup_rec_devices(fixed_rec_locations, N_T)

    # Assert
    assert len(rec_devices) == 7
    for rec_device_grid in rec_devices:
      assert len(rec_device_grid) == 1
    assert elastic_3d.fd_param['n_src'] == N_SRCS
    assert elastic_3d.fd_param['n_rec'] == N_REC * N_REC


def test_setup_rec_devices_fails_if_no_src_locations(fixed_rec_locations,
                                                     vp_vs_rho_model_3d):
  # Arrange
  with patch.object(elastic_isotropic.ElasticIsotropic3D, "_make", mock_make):
    elastic_3d = elastic_isotropic.ElasticIsotropic3D(None, (D_Y, D_X, D_Z),
                                                      None, None, None, None,
                                                      None)
    elastic_3d.model_sep, elastic_3d.model_padding = elastic_3d._setup_model(
        vp_vs_rho_model_3d,
        model_padding=(N_Y_PAD, N_X_PAD, N_Z_PAD),
        model_sampling=(D_Y, D_X, D_Z))

    # Act and Assert
    with pytest.raises(RuntimeError):
      elastic_3d._setup_rec_devices(fixed_rec_locations, N_T)


def test_setup_rec_devices_fails_if_no_model(fixed_rec_locations):
  # Arrange
  with patch.object(elastic_isotropic.ElasticIsotropic3D, "_make", mock_make):
    elastic_3d = elastic_isotropic.ElasticIsotropic3D(None, (D_Y, D_X, D_Z),
                                                      None, None, None, None,
                                                      None)

    # Act and Assert
    with pytest.raises(RuntimeError):
      elastic_3d._setup_rec_devices(fixed_rec_locations, N_T)


def test_setup_src_devices(src_locations, vp_vs_rho_model_3d):
  # Arrange
  with patch.object(elastic_isotropic.ElasticIsotropic3D, "_make", mock_make):
    elastic_3d = elastic_isotropic.ElasticIsotropic3D(None, (D_Y, D_X, D_Z),
                                                      None, None, None, None,
                                                      None)
    elastic_3d.model_sep, elastic_3d.model_padding = elastic_3d._setup_model(
        vp_vs_rho_model_3d,
        model_padding=(N_Y_PAD, N_X_PAD, N_Z_PAD),
        model_sampling=(D_Y, D_X, D_Z))

    # Act
    src_devices = elastic_3d._setup_src_devices(src_locations, N_T)

    # Assert
    assert len(src_devices) == 7
    for src_device_grid in src_devices:
      assert len(src_device_grid) == N_SRCS


def test_setup_src_devices_fails_if_no_model(src_locations):
  # Arrange
  with patch.object(elastic_isotropic.ElasticIsotropic3D, "_make", mock_make):
    elastic_3d = elastic_isotropic.ElasticIsotropic3D(None, (D_Y, D_X, D_Z),
                                                      None, None, None, None,
                                                      None)
    # Act and Assert
    with pytest.raises(RuntimeError):
      elastic_3d._setup_src_devices(src_locations, N_T)


def test_setup_model(vp_vs_rho_model_3d):
  # Arrange
  with patch.object(elastic_isotropic.ElasticIsotropic3D, "_make", mock_make):
    elastic_3d = elastic_isotropic.ElasticIsotropic3D(None, (D_Y, D_X, D_Z),
                                                      None,
                                                      None,
                                                      None,
                                                      None,
                                                      None,
                                                      model_padding=(N_Y_PAD,
                                                                     N_X_PAD,
                                                                     N_Z_PAD))
    # Act
    elastic_3d.model_sep, elastic_3d.model_padding = elastic_3d._setup_model(
        vp_vs_rho_model_3d,
        model_padding=(N_Y_PAD, N_X_PAD, N_Z_PAD),
        model_sampling=(D_Y, D_X, D_Z))

    # Assert
    assert elastic_3d.fd_param['n_y'] == 2 * N_Y_PAD + 2 * elastic_3d._FAT + N_Y
    assert (elastic_3d.fd_param['n_x'] -
            2 * elastic_3d._FAT) % elastic_3d._BLOCK_SIZE == 0
    assert (elastic_3d.fd_param['n_z'] -
            2 * elastic_3d._FAT) % elastic_3d._BLOCK_SIZE == 0
    assert elastic_3d.fd_param['d_y'] == D_Y
    assert elastic_3d.fd_param['d_x'] == D_X
    assert elastic_3d.fd_param['d_z'] == D_Z
    assert elastic_3d.fd_param['y_pad'] == N_Y_PAD
    assert elastic_3d.fd_param['x_pad_minus'] == N_X_PAD
    assert elastic_3d.fd_param['x_pad_plus'] == elastic_3d.fd_param['n_x'] - (
        N_X + N_X_PAD + 2 * elastic_3d._FAT)
    assert elastic_3d.fd_param['z_pad_minus'] == N_Z_PAD
    assert elastic_3d.fd_param['z_pad_plus'] == elastic_3d.fd_param['n_z'] - (
        N_Z + N_Z_PAD + 2 * elastic_3d._FAT)

    #check model gets set
    start_y = elastic_3d._FAT + N_Y_PAD
    end_y = start_y + N_Y
    start_x = elastic_3d._FAT + N_X_PAD
    end_x = start_x + N_X
    start_z = elastic_3d._FAT + N_Z_PAD
    end_z = start_z + N_Z

    # check that model was converted to lame paramters
    # lame_model_3d = elastic_isotropic.convert_to_lame(vp_vs_rho_model_3d)
    assert np.allclose(
        elastic_3d.model_sep.getNdArray()[:, start_y:end_y, start_x:end_x,
                                          start_z:end_z], vp_vs_rho_model_3d)


def test_pad_model(vp_vs_rho_model_3d):
  # Arrange
  with patch.object(elastic_isotropic.ElasticIsotropic3D, "_make", mock_make):
    elastic_3d = elastic_isotropic.ElasticIsotropic3D(None, (D_Y, D_X, D_Z),
                                                      None, None, None, None,
                                                      None, None, None)

    padding, new_shape, new_origins = elastic_3d._calc_pad_params(
        (N_Y, N_X, N_Z), (N_Y_PAD, N_X_PAD, N_Z_PAD), (D_Y, D_X, D_Z),
        elastic_3d._FAT)
    # Act
    model = elastic_3d._pad_model(vp_vs_rho_model_3d, padding, elastic_3d._FAT)

    # Assert
    assert (model.shape[2] - 2 * elastic_3d._FAT) % elastic_3d._BLOCK_SIZE == 0
    assert (model.shape[3] - 2 * elastic_3d._FAT) % elastic_3d._BLOCK_SIZE == 0
    assert padding[0][0] == N_Y_PAD
    assert padding[0][1] == N_Y_PAD
    assert padding[1][0] == N_X_PAD
    assert padding[1][1] == model.shape[2] - (N_X + N_X_PAD +
                                              2 * elastic_3d._FAT)
    assert padding[2][0] == N_Z_PAD
    assert padding[2][1] == model.shape[3] - (N_Z + N_Z_PAD +
                                              2 * elastic_3d._FAT)

    #check model gets set
    start_y = elastic_3d._FAT + N_Y_PAD
    end_y = start_y + N_Y
    start_x = elastic_3d._FAT + N_X_PAD
    end_x = start_x + N_X
    start_z = elastic_3d._FAT + N_Z_PAD
    end_z = start_z + N_Z
    assert np.allclose(model[:, start_y:end_y, start_x:end_x, start_z:end_z],
                       vp_vs_rho_model_3d)


###############################################
#### ElasticIsotropic3D Born tests ###########
###############################################
@pytest.mark.gpu
def test_init_jacobian(trapezoid_wavelet, fixed_rec_locations, src_locations,
                       vp_vs_rho_model_3d):

  # Arrange and act
  elastic_3d = elastic_isotropic.ElasticIsotropic3D(
      model=vp_vs_rho_model_3d,
      model_sampling=(D_Y, D_X, D_Z),
      model_padding=(N_Y_PAD, N_X_PAD, N_Z_PAD),
      wavelet=trapezoid_wavelet,
      d_t=D_T,
      src_locations=src_locations,
      rec_locations=fixed_rec_locations,
      gpus=I_GPUS)

  # assert
  assert isinstance(elastic_3d._operator.lin_op.args[1].args[0],
                    wave_equation._JacobianWaveCppOp)


@pytest.mark.gpu
def test_jacobian(trapezoid_wavelet, fixed_rec_locations, src_locations,
                  vp_vs_rho_model_3d):

  # Arrange
  vp_vs_rho_model_3d[..., N_Z // 2:] *= 1.5
  elastic_3d = elastic_isotropic.ElasticIsotropic3D(
      model=vp_vs_rho_model_3d,
      model_sampling=(D_Y, D_X, D_Z),
      model_padding=(N_Y_PAD, N_X_PAD, N_Z_PAD),
      wavelet=trapezoid_wavelet,
      d_t=D_T,
      src_locations=src_locations,
      rec_locations=fixed_rec_locations,
      gpus=I_GPUS)

  #reflectivity model
  lin_model = np.gradient(vp_vs_rho_model_3d, axis=-1)

  # act
  lin_data = elastic_3d.jacobian(lin_model)

  # Assert
  assert lin_data.shape == (N_SRCS, N_WFLD_COMPONENTS, N_REC * N_REC, N_T)
  assert not np.all((lin_data == 0))
  assert not np.any(np.isnan(lin_data))


@pytest.mark.gpu
def test_jacobian_vel_recording_components(trapezoid_wavelet,
                                           fixed_rec_locations, src_locations,
                                           vp_vs_rho_model_3d):

  # Arrange
  recording_components = ['vy', 'vx', 'vz']
  vp_vs_rho_model_3d[..., N_Z // 2:] *= 1.5
  elastic_3d = elastic_isotropic.ElasticIsotropic3D(
      model=vp_vs_rho_model_3d,
      model_sampling=(D_Y, D_X, D_Z),
      model_padding=(N_Y_PAD, N_X_PAD, N_Z_PAD),
      wavelet=trapezoid_wavelet,
      d_t=D_T,
      src_locations=src_locations,
      rec_locations=fixed_rec_locations,
      gpus=I_GPUS,
      recording_components=recording_components)

  #reflectivity model
  lin_model = np.gradient(vp_vs_rho_model_3d, axis=-1)

  # act
  lin_data = elastic_3d.jacobian(lin_model)

  # Assert
  assert lin_data.shape == (N_SRCS, len(recording_components), N_REC * N_REC,
                            N_T)
  assert not np.all((lin_data == 0))
  assert not np.any(np.isnan(lin_data))


@pytest.mark.gpu
def test_jacobian_pressure_recording_components(trapezoid_wavelet,
                                                fixed_rec_locations,
                                                src_locations,
                                                vp_vs_rho_model_3d):
  # Arrange
  recording_components = ['p']
  vp_vs_rho_model_3d[..., N_Z // 2:] *= 1.5
  elastic_3d = elastic_isotropic.ElasticIsotropic3D(
      model=vp_vs_rho_model_3d,
      model_sampling=(D_Y, D_X, D_Z),
      model_padding=(N_Y_PAD, N_X_PAD, N_Z_PAD),
      wavelet=trapezoid_wavelet,
      d_t=D_T,
      src_locations=src_locations,
      rec_locations=fixed_rec_locations,
      gpus=I_GPUS,
      recording_components=recording_components)

  #reflectivity model
  lin_model = np.gradient(vp_vs_rho_model_3d, axis=-1)

  # act
  lin_data = elastic_3d.jacobian(lin_model)

  # Assert
  assert lin_data.shape == (N_SRCS, len(recording_components), N_REC * N_REC,
                            N_T)
  assert not np.all((lin_data == 0))
  assert not np.any(np.isnan(lin_data))


@pytest.mark.gpu
def test_jacobian_adjoint(trapezoid_wavelet, fixed_rec_locations, src_locations,
                          vp_vs_rho_model_3d):
  # Arrange
  vp_vs_rho_model_3d[..., N_Z // 2:] *= 1.5
  elastic_3d = elastic_isotropic.ElasticIsotropic3D(
      model=vp_vs_rho_model_3d,
      model_sampling=(D_Y, D_X, D_Z),
      model_padding=(N_Y_PAD, N_X_PAD, N_Z_PAD),
      wavelet=trapezoid_wavelet,
      d_t=D_T,
      src_locations=src_locations,
      rec_locations=fixed_rec_locations,
      gpus=I_GPUS)

  #reflectivity model
  lin_model = np.gradient(vp_vs_rho_model_3d, axis=-1)

  # act
  lin_data = elastic_3d.jacobian(lin_model)
  lin_model = elastic_3d.jacobian_adjoint(lin_data)

  # Assert
  assert lin_model.shape == vp_vs_rho_model_3d.shape
  assert not np.all((lin_model == 0))
  assert not np.any(np.isnan(lin_model))


@pytest.mark.gpu
def test_jacobian_dot_product(trapezoid_wavelet, fixed_rec_locations,
                              src_locations, vp_vs_rho_model_3d):
  # Arrange
  vp_vs_rho_model_3d[..., N_Z // 2:] *= 1.5
  elastic_3d = elastic_isotropic.ElasticIsotropic3D(
      model=vp_vs_rho_model_3d,
      model_sampling=(D_Y, D_X, D_Z),
      model_padding=(N_Y_PAD, N_X_PAD, N_Z_PAD),
      wavelet=trapezoid_wavelet,
      d_t=D_T,
      src_locations=src_locations,
      rec_locations=fixed_rec_locations,
      gpus=I_GPUS)

  #assert
  elastic_3d.dot_product_test(True)


@pytest.mark.gpu
def test_jacobian_dot_product_vel_recording_components(trapezoid_wavelet,
                                                       fixed_rec_locations,
                                                       src_locations,
                                                       vp_vs_rho_model_3d):
  # Arrange
  recording_components = ['vy', 'vx', 'vz']
  vp_vs_rho_model_3d[..., N_Z // 2:] *= 1.5
  elastic_3d = elastic_isotropic.ElasticIsotropic3D(
      model=vp_vs_rho_model_3d,
      model_sampling=(D_Y, D_X, D_Z),
      model_padding=(N_Y_PAD, N_X_PAD, N_Z_PAD),
      wavelet=trapezoid_wavelet,
      d_t=D_T,
      src_locations=src_locations,
      rec_locations=fixed_rec_locations,
      gpus=I_GPUS,
      recording_components=recording_components)

  #assert
  elastic_3d.dot_product_test(True)


@pytest.mark.gpu
def test_jacobian_dot_product_pressure_recording_components(
    trapezoid_wavelet, fixed_rec_locations, src_locations, vp_vs_rho_model_3d):
  # Arrange
  recording_components = ['p']
  vp_vs_rho_model_3d[..., N_Z // 2:] *= 1.5
  elastic_3d = elastic_isotropic.ElasticIsotropic3D(
      model=vp_vs_rho_model_3d,
      model_sampling=(D_Y, D_X, D_Z),
      model_padding=(N_Y_PAD, N_X_PAD, N_Z_PAD),
      wavelet=trapezoid_wavelet,
      d_t=D_T,
      src_locations=src_locations,
      rec_locations=fixed_rec_locations,
      gpus=I_GPUS,
      recording_components=recording_components)

  #assert
  elastic_3d.dot_product_test(True)