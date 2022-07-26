import pytest
import numpy as np
from wavelets import Wavelet

NT = 10000
OT = 0.0
DT = 1 / 10e2
FREQ_SPIKE = 100.0

# CONFIG = {'n_t': NT, 'd_t': DT}

# DELAY = 2.0
# DOM_FREQ = 5.0
# CONFIG_RICKER = {'n_t': NT, 'd_t': DT, 'delay': DELAY, 'dom_freq': DOM_FREQ}

# F1 = 3
# F2 = 5
# F3 = 10
# F4 = 12
# CONFIG_TRAPEZOID = {
#     'n_t': NT,
#     'd_t': DT,
#     'delay': DELAY,
#     'f1': F1,
#     'f2': F2,
#     'f3': F3,
#     'f4': F4
# }


@pytest.fixture
def time_series_with_freq_spike():
  fs = 1 / DT
  amp = 2 * np.sqrt(2)
  noise_power = 0.001 * fs / 2
  time = np.arange(NT) / fs

  ts = amp * np.sin(2 * np.pi * FREQ_SPIKE * time)

  return ts


def test_linear_spectra(time_series_with_freq_spike):
  # get linear spectra of time series with a freq spike
  freqs, spectra = Wavelet.linear_spectra(time_series_with_freq_spike, 1 / DT)

  # test that freq spike is at expected freq location
  f_spike = freqs[np.argmax(spectra)]
  assert f_spike == pytest.approx(FREQ_SPIKE, 1 / DT)


def test_density_spectra(time_series_with_freq_spike):
  #make dummy child class so we can test concrete methods of abstract class
  # Wavelet.__abstractmethods__ = set()

  # make a dummy wavelet
  # wavelet = Wavelet(NT, DT)

  # get linear spectra of time series with a freq spike
  freqs, spectra = Wavelet.density_spectra(time_series_with_freq_spike, 1 / DT)

  # test that freq spike is at expected freq location
  f_spike = freqs[np.argmax(spectra)]
  assert f_spike == pytest.approx(FREQ_SPIKE, 1 / DT)


def test_spectra(time_series_with_freq_spike):
  # call spectra with all possible magnitude options
  mags = ['amplitude', 'power', 'power_db', 'power_db_norm']
  for mag in mags:
    # get linear spectra of time series with a freq spike
    freqs, spectra = Wavelet.spectra(time_series_with_freq_spike,
                                     1 / DT,
                                     mag=mag)

    # test that freq spike is at expected freq location
    f_spike = freqs[np.argmax(spectra)]
    assert f_spike == pytest.approx(FREQ_SPIKE, 1 / DT)


def test_calc_max_freq(time_series_with_freq_spike):
  # use calc_max_freq to find freq spike
  f_max = Wavelet.calc_max_freq(time_series_with_freq_spike, DT)

  # test that freq spike is at expected freq location
  assert f_max == pytest.approx(FREQ_SPIKE, 1 / DT)


def test_Wavelet_init_fails(time_series_with_freq_spike):
  "Wavelet class should be purely abstract so will fail if initiailized"
  with pytest.raises(TypeError):
    wavelet = Wavelet.Wavelet(time_series_with_freq_spike, DT)


def test_set_arr(time_series_with_freq_spike):
  #make dummy child class so we can test concrete methods of abstract class
  Wavelet.Wavelet.__abstractmethods__ = set()

  # make a dummy wavelet
  wavelet = Wavelet.Wavelet(time_series_with_freq_spike, DT)

  # set arr
  wavelet.set_arr(time_series_with_freq_spike)

  assert np.allclose(wavelet.arr, time_series_with_freq_spike)


def test_get_arr(time_series_with_freq_spike):
  #make dummy child class so we can test concrete methods of abstract class
  Wavelet.Wavelet.__abstractmethods__ = set()

  # make a dummy wavelet
  wavelet = Wavelet.Wavelet(time_series_with_freq_spike, DT)

  # set arr
  wavelet.set_arr(time_series_with_freq_spike)

  # get arr
  arr = wavelet.get_arr()

  assert np.allclose(arr, time_series_with_freq_spike)
