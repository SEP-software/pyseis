from pyseis.wavelets import Wavelet
import numpy as np
import SepVector
import Hypercube


class Acoustic2D(Wavelet.Wavelet):

  def get_sep(self):
    wavelet_sep = SepVector.getSepVector(
        Hypercube.hypercube(axes=[
            Hypercube.axis(n=self.n_t, o=0.0, d=self.d_t),
            Hypercube.axis(n=1),
            Hypercube.axis(n=1)
        ]))
    wavelet_sep.getNdArray()[:] = self.arr
    return wavelet_sep

  def _make(self, arr):
    pass


class AcousticIsotropicRicker2D(Acoustic2D):

  def __init__(self, n_t: int, d_t: float, dom_freq: float, delay: float):
    """return a 1d np array containing a Ricker wavelet

    Args:
      dom_freq - float - central freq of ricker
      delay - float - how far to shift center of wavelet
    Returns:
      wavelet - np array
    """
    self.dom_freq = dom_freq
    self.delay = delay

    arr = Wavelet.make_ricker_trace(n_t, d_t, self.dom_freq, self.delay)
    super().__init__(arr, d_t)


class AcousticIsotropicTrapezoid2D(Acoustic2D):

  def __init__(self, n_t: int, d_t: float, f1: float, f2: float, f3: float,
               f4: float, delay: float,  a2: float = 1.0, a3: float = 1.0):
    self.f1 = f1
    self.f2 = f2
    self.f3 = f3
    self.f4 = f4
    self.delay = delay
    self.a2=a2
    self.a3=a3

    arr = Wavelet.make_trapezoid_trace(n_t, d_t, self.f1, self.f2, self.f3,
                                       self.f4, self.delay, self.a2, self.a3)

    super().__init__(arr, d_t)
