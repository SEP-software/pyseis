from mock import patch
import pytest
import numpy as np
from wave_equations import acoustic_isotropic
from wavelets.acoustic import Acoustic2D

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

N_SRCS = 50
Z_SHOT = 10.0
N_REC = 10
D_X_REC = 11.0
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
def vp_model_half_space():
  vp_model = np.zeros((N_X, N_Z))
  vp_model[:, :N_Z // 2] = VP_1
  vp_model[:, N_Z // 2:] = VP_2
  return vp_model


@pytest.fixture
def vp_model_shallow_half_space():
  vp_model = np.zeros((N_X, N_Z))
  vp_model[:, :N_Z // 4] = VP_1
  vp_model[:, N_Z // 4:] = VP_2
  return vp_model


# mock the make function so we can test all of the individual function calls within make below
def mock_make(self, model, wavelet, d_t, src_locations, rec_locations, gpus):
  return None


###############################################
#### AcousticIsotropic2D tests ################
###############################################
@pytest.mark.gpu
def test_fwd_diff_models_makes_diff_data(ricker_wavelet, fixed_rec_locations,
                                         src_locations, vp_model_half_space,
                                         vp_model_shallow_half_space):
  # Arrange
  acoustic_2d = acoustic_isotropic.AcousticIsotropic2D(
      model=vp_model_half_space,
      model_sampling=(D_X, D_Z),
      model_padding=(N_X_PAD, N_Z_PAD),
      wavelet=ricker_wavelet,
      d_t=D_T,
      src_locations=src_locations,
      rec_locations=fixed_rec_locations,
      gpus=I_GPUS)

  # Act
  data1 = acoustic_2d.fwd(vp_model_half_space)
  data2 = acoustic_2d.fwd(vp_model_shallow_half_space)

  # Assert
  assert not np.allclose(data1, data2)


@pytest.mark.gpu
def test_fwd(ricker_wavelet, fixed_rec_locations, src_locations,
             vp_model_half_space):
  # Arrange
  acoustic_2d = acoustic_isotropic.AcousticIsotropic2D(
      model=vp_model_half_space,
      model_sampling=(D_X, D_Z),
      model_padding=(N_X_PAD, N_Z_PAD),
      wavelet=ricker_wavelet,
      d_t=D_T,
      src_locations=src_locations,
      rec_locations=fixed_rec_locations,
      gpus=I_GPUS)

  # Act
  data = acoustic_2d.fwd(vp_model_half_space)

  # Assert
  assert data.shape == (N_SRCS, N_REC, N_T)
  assert not np.all((data == 0))


@pytest.mark.gpu
def test_init(ricker_wavelet, fixed_rec_locations, src_locations,
              vp_model_half_space):
  # Act
  acoustic_2d = acoustic_isotropic.AcousticIsotropic2D(
      model=vp_model_half_space,
      model_sampling=(D_X, D_Z),
      model_padding=(N_X_PAD, N_Z_PAD),
      wavelet=ricker_wavelet,
      d_t=D_T,
      src_locations=src_locations,
      rec_locations=fixed_rec_locations,
      gpus=I_GPUS)


def test_setup_wavelet(ricker_wavelet):
  # Arrange
  with patch.object(acoustic_isotropic.AcousticIsotropic2D, "_make", mock_make):
    acoustic_2d = acoustic_isotropic.AcousticIsotropic2D(
        None, None, None, None, None, None, None)

    # Act
    acoustic_2d._setup_wavelet(ricker_wavelet, D_T)


def test_setup_data(variable_rec_locations, src_locations, vp_model_half_space):
  # Arrange
  with patch.object(acoustic_isotropic.AcousticIsotropic2D, "_make", mock_make):
    acoustic_2d = acoustic_isotropic.AcousticIsotropic2D(
        None, (D_X, D_Z),
        None,
        None,
        None,
        None,
        None,
        model_padding=(N_X_PAD, N_Z_PAD))
    acoustic_2d._setup_model(vp_model_half_space)
    acoustic_2d._setup_src_devices(src_locations, N_T)
    acoustic_2d._setup_rec_devices(variable_rec_locations, N_T)

    # Act
    acoustic_2d._setup_data(N_T, D_T)

    # Assert
    assert acoustic_2d.data_sep.getNdArray().shape == (N_SRCS, N_REC, N_T)


def test_setup_data_fails_if_no_src_or_rec(variable_rec_locations,
                                           src_locations, vp_model_half_space):
  # Arrange
  with patch.object(acoustic_isotropic.AcousticIsotropic2D, "_make", mock_make):
    acoustic_2d = acoustic_isotropic.AcousticIsotropic2D(
        None, (D_X, D_Z),
        None,
        None,
        None,
        None,
        None,
        model_padding=(N_X_PAD, N_Z_PAD))
    acoustic_2d._setup_model(vp_model_half_space)

    # Act and Assert
    with pytest.raises(RuntimeError):
      acoustic_2d._setup_data(N_T, D_T)

    # Act and Assert
    acoustic_2d._setup_src_devices(src_locations, N_T)
    with pytest.raises(RuntimeError):
      acoustic_2d._setup_data(N_T, D_T)

    # Act and Assert
    acoustic_2d._setup_rec_devices(variable_rec_locations, N_T)
    acoustic_2d._setup_data(N_T, D_T)


def test_setup_rec_devices_variable_receivers(variable_rec_locations,
                                              src_locations,
                                              vp_model_half_space):
  # Arrange
  with patch.object(acoustic_isotropic.AcousticIsotropic2D, "_make", mock_make):
    acoustic_2d = acoustic_isotropic.AcousticIsotropic2D(
        None, (D_X, D_Z),
        None,
        None,
        None,
        None,
        None,
        model_padding=(N_X_PAD, N_Z_PAD))
    acoustic_2d._setup_model(vp_model_half_space)
    acoustic_2d._setup_src_devices(src_locations, N_T)

    # Act
    acoustic_2d._setup_rec_devices(variable_rec_locations, N_T)

    # Assert
    assert len(acoustic_2d.rec_devices) == N_SRCS


def test_setup_rec_devices_variable_receivers_nshot_mismatch(
    variable_rec_locations, src_locations, vp_model_half_space):
  # Arrange
  with patch.object(acoustic_isotropic.AcousticIsotropic2D, "_make", mock_make):
    acoustic_2d = acoustic_isotropic.AcousticIsotropic2D(
        None, (D_X, D_Z),
        None,
        None,
        None,
        None,
        None,
        model_padding=(N_X_PAD, N_Z_PAD))
    acoustic_2d._setup_model(vp_model_half_space)
    acoustic_2d._setup_src_devices(src_locations, N_T)

    # Act and Assert
    with pytest.raises(RuntimeError):
      acoustic_2d._setup_rec_devices(variable_rec_locations[1:], N_T)


def test_setup_rec_devices_fixed_receivers(fixed_rec_locations, src_locations,
                                           vp_model_half_space):
  # Arrange
  with patch.object(acoustic_isotropic.AcousticIsotropic2D, "_make", mock_make):
    acoustic_2d = acoustic_isotropic.AcousticIsotropic2D(
        None, (D_X, D_Z),
        None,
        None,
        None,
        None,
        None,
        model_padding=(N_X_PAD, N_Z_PAD))
    acoustic_2d._setup_model(vp_model_half_space)
    acoustic_2d._setup_src_devices(src_locations, N_T)

    # Act
    acoustic_2d._setup_rec_devices(fixed_rec_locations, N_T)

    # Assert
    assert len(acoustic_2d.rec_devices) == N_SRCS


def test_setup_rec_devices_fails_if_no_src_locations(fixed_rec_locations,
                                                     vp_model_half_space):
  # Arrange
  with patch.object(acoustic_isotropic.AcousticIsotropic2D, "_make", mock_make):
    acoustic_2d = acoustic_isotropic.AcousticIsotropic2D(
        None, (D_X, D_Z),
        None,
        None,
        None,
        None,
        None,
        model_padding=(N_X_PAD, N_Z_PAD))
    acoustic_2d._setup_model(vp_model_half_space)

    # Act and Assert
    with pytest.raises(RuntimeError):
      acoustic_2d._setup_rec_devices(fixed_rec_locations, N_T)


def test_setup_rec_devices_fails_if_no_model(fixed_rec_locations):
  # Arrange
  with patch.object(acoustic_isotropic.AcousticIsotropic2D, "_make", mock_make):
    acoustic_2d = acoustic_isotropic.AcousticIsotropic2D(
        None, (D_X, D_Z),
        None,
        None,
        None,
        None,
        None,
        model_padding=(N_X_PAD, N_Z_PAD))
    # Act and Assert
    with pytest.raises(RuntimeError):
      acoustic_2d._setup_rec_devices(fixed_rec_locations, N_T)


def test_setup_src_devices(src_locations, vp_model_half_space):
  # Arrange
  with patch.object(acoustic_isotropic.AcousticIsotropic2D, "_make", mock_make):
    acoustic_2d = acoustic_isotropic.AcousticIsotropic2D(
        None, (D_X, D_Z),
        None,
        None,
        None,
        None,
        None,
        model_padding=(N_X_PAD, N_Z_PAD))
    acoustic_2d._setup_model(vp_model_half_space)

    # Act
    acoustic_2d._setup_src_devices(src_locations, N_T)

    # Assert
    assert len(acoustic_2d.src_devices) == N_SRCS


def test_setup_src_devices_fails_if_no_model(src_locations):
  # Arrange
  with patch.object(acoustic_isotropic.AcousticIsotropic2D, "_make", mock_make):
    acoustic_2d = acoustic_isotropic.AcousticIsotropic2D(
        None, (D_X, D_Z),
        None,
        None,
        None,
        None,
        None,
        model_padding=(N_X_PAD, N_Z_PAD))
    # Act and Assert
    with pytest.raises(RuntimeError):
      acoustic_2d._setup_src_devices(src_locations, N_T)


def test_setup_model(vp_model_half_space):
  # Arrange
  with patch.object(acoustic_isotropic.AcousticIsotropic2D, "_make", mock_make):
    acoustic_2d = acoustic_isotropic.AcousticIsotropic2D(
        None, (D_X, D_Z),
        None,
        None,
        None,
        None,
        None,
        model_padding=(N_X_PAD, N_Z_PAD))

    # Act
    acoustic_2d._setup_model(vp_model_half_space)

    # Assert
    assert (acoustic_2d.fd_param['n_x'] -
            2 * acoustic_2d._FAT) % acoustic_2d._BLOCK_SIZE == 0
    assert (acoustic_2d.fd_param['n_z'] -
            2 * acoustic_2d._FAT) % acoustic_2d._BLOCK_SIZE == 0
    assert acoustic_2d.fd_param['d_x'] == D_X
    assert acoustic_2d.fd_param['d_z'] == D_Z
    assert acoustic_2d.fd_param['x_pad_minus'] == N_X_PAD
    assert acoustic_2d.fd_param['x_pad_plus'] == acoustic_2d.fd_param['n_x'] - (
        N_X + N_X_PAD + 2 * acoustic_2d._FAT)
    assert acoustic_2d.fd_param['z_pad_minus'] == N_Z_PAD
    assert acoustic_2d.fd_param['z_pad_plus'] == acoustic_2d.fd_param['n_z'] - (
        N_Z + N_Z_PAD + 2 * acoustic_2d._FAT)

    #check model gets set
    start_x = acoustic_2d._FAT + N_X_PAD
    end_x = start_x + N_X
    start_z = acoustic_2d._FAT + N_Z_PAD
    end_z = start_z + N_Z
    assert np.allclose(
        acoustic_2d.model_sep.getNdArray()[start_x:end_x, start_z:end_z],
        vp_model_half_space)


def test_pad_model(vp_model_half_space):
  # Arrange
  with patch.object(acoustic_isotropic.AcousticIsotropic2D, "_make", mock_make):
    acoustic_2d = acoustic_isotropic.AcousticIsotropic2D(
        None, None, None, None, None, None, None)

    # Act
    model, x_pad, x_pad_plus, new_o_x, z_pad, z_pad_plus, new_o_z = acoustic_2d._pad_model(
        vp_model_half_space, (D_X, D_Z), (N_X_PAD, N_Z_PAD))

    # Assert
    assert (model.shape[0] -
            2 * acoustic_2d._FAT) % acoustic_2d._BLOCK_SIZE == 0
    assert (model.shape[1] -
            2 * acoustic_2d._FAT) % acoustic_2d._BLOCK_SIZE == 0
    assert x_pad == N_X_PAD
    assert x_pad_plus == model.shape[0] - (N_X + N_X_PAD + 2 * acoustic_2d._FAT)
    assert z_pad == N_Z_PAD
    assert z_pad_plus == model.shape[1] - (N_Z + N_Z_PAD + 2 * acoustic_2d._FAT)

    #check model gets set
    start_x = acoustic_2d._FAT + N_X_PAD
    end_x = start_x + N_X
    start_z = acoustic_2d._FAT + N_Z_PAD
    end_z = start_z + N_Z
    assert np.allclose(model[start_x:end_x, start_z:end_z], vp_model_half_space)
