import abc
import numpy as np


def sample_pressure_2d_elastic(data: np.ndarray) -> np.ndarray:
  return 0.5 * (data[:, 2] + data[:, 3])


class Geometry(abc.ABC):

  def __init__(self):
    self.sampling_op = None
    self.src_positions = None
    self.rec_positions = None

  def get_src_positions(self) -> np.ndarray:
    return self.src_positions

  def get_rec_positions(self) -> np.ndarray:
    return self.rec_positions


class Full2D(Geometry):

  def __init__(self, n_src: int, o_x_src: float, d_x_src: float, z_src: float,
               n_rec: int, o_x_rec: float, d_x_rec: float, z_rec: float):
    src_x = o_x_src + np.arange(n_src) * d_x_src
    src_z = np.ones_like(src_x) * z_src
    self.src_positions = np.array([src_x, src_z]).T

    rec_x = o_x_rec + np.arange(n_rec) * d_x_rec
    rec_z = np.ones_like(rec_x) * z_rec
    self.rec_positions = np.array([rec_x, rec_z]).T


class Streamer2D(Geometry):

  def __init__(self, n_src: int, o_x_src: float, d_x_src: float, z_src: float,
               n_rec: int, o_x_rec: float, d_x_rec: float, z_rec: float):
    src_x = o_x_src + np.arange(n_src) * d_x_src
    src_z = np.ones_like(src_x) * z_src
    self.src_positions = np.array([src_x, src_z]).T

    relative_rec_x = o_x_rec + np.arange(n_rec) * d_x_rec
    rec_x = np.repeat(np.expand_dims(relative_rec_x, axis=0), n_src, axis=0)
    rec_x = (rec_x.T + src_x).T
    rec_z = np.ones_like(rec_x) * z_rec
    self.rec_positions = np.array([rec_x.T, rec_z.T]).T


class Streamer3D(Geometry):

  def __init__(self, n_y_src: int, o_y_src: float, d_y_src: float, n_x_src: int,
               o_x_src: float, d_x_src: float, z_src: float, n_y_rec: int,
               o_y_rec: float, d_y_rec: float, n_x_rec: int, o_x_rec: float,
               d_x_rec: float, z_rec: float):
    src_y = o_y_src + np.arange(n_y_src) * d_y_src
    src_x = o_x_src + np.arange(n_x_src) * d_x_src
    src_xx, src_yy = np.meshgrid(src_x, src_y, indexing='ij')
    src_xx = src_xx.flatten()
    src_yy = src_yy.flatten()
    src_zz = np.ones_like(src_xx) * z_src
    self.src_positions = np.array([src_yy, src_xx, src_zz]).T

    relative_rec_x = o_x_rec + np.arange(n_x_rec) * d_x_rec
    relative_rec_y = o_y_rec + np.arange(n_y_rec) * d_y_rec
    relative_rec_yy, relative_rec_xx = np.meshgrid(relative_rec_y,
                                                   relative_rec_x,
                                                   indexing='ij')
    relative_rec_xx = relative_rec_xx.flatten()
    relative_rec_yy = relative_rec_yy.flatten()
    relative_rec_zz = np.ones_like(relative_rec_xx) * z_rec

    n_src = len(src_xx)
    rec_xx = np.repeat(np.expand_dims(relative_rec_xx, axis=0), n_src, axis=0)
    rec_yy = np.repeat(np.expand_dims(relative_rec_yy, axis=0), n_src, axis=0)
    rec_xx = (rec_xx.T + src_xx).T
    rec_yy = (rec_yy.T + src_yy).T
    rec_zz = np.ones_like(rec_xx) * z_rec
    self.rec_positions = np.array([rec_yy.T, rec_xx.T, rec_zz.T]).T
