import pytest
import numpy as np
from wavelets.acoustic import Acoustic3D

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
CONFIG_TRAPEZOID = {'delay': DELAY, 'f1': F1, 'f2': F2, 'f3': F3, 'f4': F4}

FREQ_SPIKE = 100.0


@pytest.fixture
def time_series_with_freq_spike():
  fs = 1 / DT
  amp = 2 * np.sqrt(2)
  noise_power = 0.001 * fs / 2
  time = np.arange(NT) / fs

  ts = amp * np.sin(2 * np.pi * FREQ_SPIKE * time)

  return ts


###############################################
#### IsotropicRicker tests ####################
###############################################
def test_make_IsotropicRicker():
  ricker_wavelet = Acoustic3D.AcousticIsotropicRicker3D(NT, DT, DOM_FREQ, DELAY)

  # check that arr is not all zeros
  arr = ricker_wavelet.get_arr()
  assert not np.all(arr == 0)

  # check center of wavelet is aligned with requested time delay
  t = np.arange(NT) * DT - DELAY
  wavelet_delay = t[np.argmax(arr)]
  assert wavelet_delay == pytest.approx(DELAY, 1 / DT)


###############################################
#### IsotropicTrapezoid class #################
###############################################
def test_make_IsotropicTrapezoid():
  trapezoid_wavelet = Acoustic3D.AcousticIsotropicTrapezoid3D(
      NT, DT, F1, F2, F3, F4, DELAY)

  # check that arr is not all zeros
  arr = trapezoid_wavelet.get_arr()
  assert not np.all(arr == 0)

  # check center of wavelet is aligned with requested time delay
  t = np.arange(NT) * DT - DELAY
  wavelet_delay = t[np.argmax(arr)]
  assert wavelet_delay == pytest.approx(DELAY, 1 / DT)

  # check that f_max is set and is less than F4
  assert trapezoid_wavelet.f_max <= F4


###############################################
#### base Acoustic3D tests ####################
###############################################
def test_Acoustic3D_make(time_series_with_freq_spike):
  wavelet = Acoustic3D.Acoustic3D(time_series_with_freq_spike, DT)
  assert wavelet.f_max == pytest.approx(FREQ_SPIKE, 0.25)


def test_Acoustic3D_get_sep(time_series_with_freq_spike):
  wavelet = Acoustic3D.Acoustic3D(time_series_with_freq_spike, DT)
  wavelet_sep = wavelet.get_sep()
  assert wavelet_sep.getNdArray().shape == (1, NT)
  assert np.allclose(wavelet_sep.getNdArray(), wavelet.arr)


def test_Acoustic3D_init(time_series_with_freq_spike):
  wavelet = Acoustic3D.Acoustic3D(time_series_with_freq_spike, DT)
