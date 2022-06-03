import pytest
import numpy as np
from wavelets.elastic import Elastic3D

NT = 10000
DT = 1 / 10e2

DELAY = 2.0
DOM_FREQ = 5.0
CONFIG_RICKER = {'n_t': NT, 'd_t': DT, 'delay': DELAY, 'dom_freq': DOM_FREQ}

F1 = 3
F2 = 5
F3 = 10
F4 = 12
CONFIG_TRAPEZOID = {'delay': DELAY, 'f1': F1, 'f2': F2, 'f3': F3, 'f4': F4}

FREQ_SPIKE = 100.0

# expected number of wavefields
N_WFLDS = 9


@pytest.fixture
def elastic_time_series_with_freq_spike():
  fs = 1 / DT
  amp = 2 * np.sqrt(2)
  noise_power = 0.001 * fs / 2
  time = np.arange(NT) / fs

  ts = amp * np.sin(2 * np.pi * FREQ_SPIKE * time)

  elastic_ts = np.zeros((N_WFLDS,) + (len(ts),))
  elastic_ts[0] = ts
  elastic_ts[1] = ts
  elastic_ts[8] = ts

  return elastic_ts


###############################################
#### ElasticIsotropicRicker tests ####################
###############################################
def test_make_ElasticIsotropicRicker():
  ricker_wavelet = Elastic3D.ElasticIsotropicRicker3D(NT, DT, DOM_FREQ, DELAY)

  # get np arr
  arr = ricker_wavelet.get_arr()

  # check shape of arr
  assert arr.shape == (N_WFLDS, NT)

  # check that arr is not all zeros
  assert not np.all(arr == 0)

  # check that arr components 0,1,2,6,7,8 are all zeros
  assert np.all(arr[0] == 0)
  assert np.all(arr[1] == 0)
  assert np.all(arr[2] == 0)
  assert np.all(arr[6] == 0)
  assert np.all(arr[7] == 0)
  assert np.all(arr[8] == 0)

  # check that arr components 2,3, 4 are NOT all zeros
  assert not np.all(arr[3] == 0)
  assert not np.all(arr[4] == 0)
  assert not np.all(arr[5] == 0)

  # check center of wavelet is aligned with requested time delay
  t = np.arange(NT) * DT
  wavelet_delay = t[np.argmax(arr[3])]
  assert wavelet_delay == pytest.approx(DELAY, DT)


###############################################
#### ElasticIsotropicTrapezoid class #################
###############################################
def test_make_ElasticIsotropicTrapezoid():
  trapezoid_wavelet = Elastic3D.ElasticIsotropicTrapezoid3D(
      NT, DT, F1, F2, F3, F4, DELAY)

  # get np arr
  arr = trapezoid_wavelet.get_arr()

  # check shape of arr
  assert arr.shape == (N_WFLDS, NT)

  # check that arr is not all zeros
  assert not np.all(arr == 0)

  # check that arr components 0,1,2,6,7,8 are all zeros
  assert np.all(arr[0] == 0)
  assert np.all(arr[1] == 0)
  assert np.all(arr[2] == 0)
  assert np.all(arr[6] == 0)
  assert np.all(arr[7] == 0)
  assert np.all(arr[8] == 0)

  # check that arr components 3,4,5 are NOT all zeros
  assert not np.all(arr[3] == 0)
  assert not np.all(arr[4] == 0)
  assert not np.all(arr[5] == 0)

  # check center of wavelet is aligned with requested time delay
  t = np.arange(NT) * DT
  wavelet_delay = t[np.argmax(arr[3])]
  assert wavelet_delay == pytest.approx(DELAY, DT)

  # check that f_max is set and is less than F4
  assert trapezoid_wavelet.f_max <= F4


###############################################
#### base Elastic3D tests ####################
###############################################
def test_Elastic3D_make(elastic_time_series_with_freq_spike):
  wavelet = Elastic3D.Elastic3D(elastic_time_series_with_freq_spike, DT)

  assert np.allclose(wavelet.get_arr(), elastic_time_series_with_freq_spike)
  assert wavelet.f_max == pytest.approx(FREQ_SPIKE, 0.25)


def test_Elastic3D_get_sep(elastic_time_series_with_freq_spike):
  wavelet = Elastic3D.Elastic3D(elastic_time_series_with_freq_spike, DT)
  wavelet_sep = wavelet.get_sep()
  assert wavelet_sep.getNdArray().shape == (1, N_WFLDS, 1, NT)
  assert np.allclose(wavelet_sep.getNdArray()[0, :, 0, :], wavelet.arr)


def test_Elastic3D_init(elastic_time_series_with_freq_spike):
  wavelet = Elastic3D.Elastic3D(elastic_time_series_with_freq_spike, DT)
