"""This module defines the abstract Wavelet class.

Wavelets contain time series signals, stored in numpy arrays, that can be used
as source functions in finite difference wave propagation.
"""
import numpy as np
from scipy import signal
import abc
import SepVector
import Hypercube

PREF = 2 * 10**-5


def spectra(arr, sampling_rate, average=True, mag='power'):
  """computes an estimate of the power spectrum of provided np array.

  uses scipy.signal.welch. Please reference
  https://www.fon.hum.uva.nl/praat/manual/power_spectral_density.html for
  amplitude vs. power vs power_db calculations.

  Args:
    arr - n dimensional np array - n dimensional sepVector for which to 
    find PSD
    sampling_rate - float - sampling rate (sec/sample). If None then is taken 
      from sep_vector
    average - bool - whether to return the average psd over all traces (True) or
      the psd for each trace (False).
    mag - string - options below
      ["amplitude"] - amplitude spectrum. If input signal is expressed in units
        of Pascal, output is expressed in units of Pa/Hz.
      ["power"] - power spectrum density. If input signal is expressed in units
        of Pascal, output is expressed in units of Pa^2/Hz.
      ["power_db"] - the logarithmic power spectrum density (20 * log10(PSD)).
        If input signal is expressed in units of Pascal, output is expressed in
        units of dB/Hz.
      ["power_db_norm"] - power_db normalized so max value is 0.
  Returns
    freqs - np array - array of sample frequencies.
    spectra - np array - chosen spectrum of input signal
  """
  fs = 1.0 / sampling_rate
  eps = 10**-100

  if mag == 'amplitude':
    freqs, spectra = linear_spectra(arr, fs, average)
  elif mag == 'power':
    freqs, spectra = density_spectra(arr, fs, average)
  elif mag == 'power_db':
    freqs, spectra = density_spectra(arr, fs, average)
    with np.errstate(divide='ignore', invalid='ignore'):
      spectra = 10 * np.log10(spectra / (PREF**2) + eps)
  elif mag == 'power_db_norm':
    freqs, spectra = density_spectra(arr, fs, average)
    with np.errstate(divide='ignore', invalid='ignore'):
      spectra = 10 * np.log10(spectra / (PREF**2) + eps)
    spectra -= np.max(spectra)
  else:
    raise ValueError(
        f"provided mag parameter, {mag}, not implemented. Options are ['amplitude','power,'power_db','power_db_norm']."
    )
  return freqs, spectra


def linear_spectra(arr, fs, average=True):
  freqs, spectra = signal.welch(arr,
                                fs=fs,
                                nperseg=arr.shape[-1],
                                scaling='spectrum')
  if average and len(spectra.shape) > 1:
    spectra = np.mean(spectra, axis=tuple(range(len(spectra.shape) - 1)))
  return freqs, spectra


def density_spectra(arr, fs, average=True):
  freqs, spectra = signal.welch(
      arr,
      fs=fs,
      # nperseg=min(1024, arr.shape[-1]),
      nperseg=arr.shape[-1],
      scaling='density')
  if average and len(spectra.shape) > 1:
    spectra = np.mean(spectra, axis=tuple(range(len(spectra.shape) - 1)))
  return freqs, spectra


def calc_max_freq(arr, sampling_rate, min_db_threshold=-40.0):
  """Find the maximum frequency with dB levels above a threshold.

  Args:
      arr (np array): time series to be analyzed. fast axis should be time 
        axis.
      min_db_threshold (float, optional): dB below max dB value of signal to
        consider having "zero" energy. Defaults to -40.

  Raises:
      RuntimeError: if no frequencies have energy below min_db_threshold

  Returns:
      float: maximum frequency with dB levels above a threshold
  """
  freqs, amps = spectra(arr, sampling_rate, mag="power_db_norm")
  # find max freq index
  max_ind = np.argmax(amps)
  print(freqs[max_ind])
  # window out frequencies and amplitudes before max freq
  freqs = freqs[max_ind:]
  amps = amps[max_ind:]
  print(freqs)
  print(amps)
  # remove freqs below threshold
  freq_greater_than_threshold = freqs[amps > min_db_threshold]
  if len(freq_greater_than_threshold) == len(freqs):
    raise RuntimeError(
        f'No freq are {min_db_threshold} dB below the max dB value.')
  return freq_greater_than_threshold[-1]


def make_ricker_trace(n_t, d_t, dom_freq, delay=0.0):
  """return a 1d np array containing a Ricker wavelet

    Args:
      n_t - int - number of time samples
      d_t - float - time sampling rate (sec)
      dom_freq - float - central freq of ricker
      delay - float - how far to shift center of wavelet
    Returns:
      arr - 1D np array
    """
  t = np.arange(n_t) * d_t - delay
  alpha = (np.pi * dom_freq) * (np.pi * dom_freq)
  arr = (1 - 2.0 * alpha * t * t) * np.exp(-1.0 * alpha * t * t)

  return arr


def make_trapezoid_trace(n_t, d_t, f1, f2, f3, f4, delay=0.0):
  """return a np array trapezoid wavelet 

    wavelet is defined by a trapezoid in the frequency domain
    ^
    |         f2____f3
    |        /        \
    |       /          \
    | ___f1/            \f4___
    --------------------------->
            freq (hz)
    Args:
      n_t - int - number of time samples
      d_t - float - time sampling rate (sec)
      f1 - float - first corner of trapezoid. below which freq content is zero.
      f2 - float - second corner of trapezoid
      f3 - float - third corner of trapezoid
      f4 - float - fourth corner of trapezoid. above which freq content is zero.
      delay - float - how far to shift center of wavelet
    Returns:
      arr - 1D np array 
    """
  if (not (f1 < f2 <= f3 < f4)):
    raise ValueError(
        "**** ERROR: Corner frequencies values must be increasing ****\n")

  # Check if f4 < f_nyquist
  f_nyquist = 1 / (2 * d_t)
  if (f4 > f_nyquist):
    raise ValueError("**** ERROR: f4 > f_nyquist ****\n")

  # define wavelet in fourier domain
  n_f = n_t
  odd = False
  if (n_t % 2 != 0):
    n_f += 1
    odd = True

  arr_fft = np.zeros(n_f, dtype=np.complex64)

  df = 1.0 / ((n_f) * d_t)
  # f = np.arange(n_f // 2) * df
  f = np.arange(n_f) * df

  left_zero = f < f1
  ramp_up = (f >= f1) & (f < f2)
  plateau = (f >= f2) & (f < f3)
  ramp_down = (f >= f3) & (f < f4)
  right_zero = f >= f4

  arr_fft[left_zero] = 0.0
  arr_fft[ramp_up] = np.cos(np.pi / 2.0 * (f2 - f[ramp_up]) /
                            (f2 - f1)) * np.cos(np.pi / 2.0 *
                                                (f2 - f[ramp_up]) / (f2 - f1))
  arr_fft[plateau] = 1.0
  arr_fft[ramp_down] = np.cos(np.pi / 2.0 * (f[ramp_down] - f3) /
                              (f4 - f3)) * np.cos(np.pi / 2.0 *
                                                  (f[ramp_down] - f3) /
                                                  (f4 - f3))
  arr_fft[right_zero] = 0.0
  arr_fft *= np.exp(-1j * 2.0 * np.pi * f * delay)

  # Duplicate, flip spectrum and take the complex conjugate
  arr_fft[n_f // 2 + 1:] = np.flip(arr_fft[1:n_f // 2].conj(), axis=0)

  # Apply inverse FFT
  if (odd):
    arr = np.fft.ifft(arr_fft[:-1]).real  #*2.0/np.sqrt(nts)
  else:
    arr = np.fft.ifft(arr_fft[:]).real  #*2.0/np.sqrt(nts)

  return arr


class Wavelet(abc.ABC):
  """_summary_
  """

  def __init__(
      self,
      arr: np.ndarray,
      d_t: float,
  ):
    """Initialize a wavelet object. 
    
    All wavelets can be described by the number of time samples and the temporal
    sampling rate.

    Args:
       
        d_t (float): spatial sampling rate in seconds.
    """
    self.n_t = arr.shape[-1]
    self.d_t = d_t
    self.arr = arr
    self.f_max = calc_max_freq(self.arr, self.d_t)

  def get_arr(self):
    """Get the numpy array containing a wavelet's time series signal.

    Returns:
        numpy array: time series signal
    """
    return self.arr

  def set_arr(self, arr: np.ndarray):
    self.arr = arr

  @abc.abstractmethod
  def get_sep(self):
    """Get a wavelet's time series signal in the form of an SepVector object.

    Raises:
        NotImplementedError: Inherited class did not implement get_sep
    """
    raise NotImplementedError('get_sep not overwritten by Wavelet child class')
