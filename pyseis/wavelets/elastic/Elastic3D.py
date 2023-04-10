from pyseis.wavelets import Wavelet
import numpy as np
import SepVector
import Hypercube

N_WFLD_COMPONENTS = 9


def make_isotropic_components(trace: np.ndarray) -> np.ndarray:
  arr = np.zeros_like(trace)
  arr = np.repeat(np.expand_dims(arr, axis=0), N_WFLD_COMPONENTS, axis=0)
  arr[3] = trace  #syy
  arr[4] = trace  #sxx
  arr[5] = trace  #szz
  return arr


class Elastic3D(Wavelet.Wavelet):

  def get_sep(self):
    wavelet_sep = SepVector.getSepVector(
        Hypercube.hypercube(axes=[
            Hypercube.axis(n=self.n_t, o=0.0, d=self.d_t),
            Hypercube.axis(n=1),
            Hypercube.axis(n=N_WFLD_COMPONENTS),
            Hypercube.axis(n=1)
        ]))
    wavelet_sep.getNdArray().flat[:] = self.arr
    return wavelet_sep


class ElasticIsotropicRicker3D(Elastic3D):

  def __init__(self, n_t: int, d_t: float, dom_freq: float, delay: float):
    """doc
    """
    self.dom_freq = dom_freq
    self.delay = delay

    trace = Wavelet.make_ricker_trace(n_t, d_t, self.dom_freq, self.delay)
    arr = make_isotropic_components(trace)
    super().__init__(arr, d_t)


class ElasticIsotropicTrapezoid3D(Elastic3D):

  def __init__(self,
               n_t: int,
               d_t: float,
               f1: float,
               f2: float,
               f3: float,
               f4: float,
               delay: float,
               a2: float = 1.0,
               a3: float = 1.0):
    """doc
    """
    self.f1 = f1
    self.f2 = f2
    self.f3 = f3
    self.f4 = f4
    self.delay = delay
    self.a2 = a2
    self.a3 = a3

    trace = Wavelet.make_trapezoid_trace(n_t, d_t, self.f1, self.f2, self.f3,
                                         self.f4, self.delay, self.a2, self.a3)
    arr = make_isotropic_components(trace)
    super().__init__(arr, d_t)
