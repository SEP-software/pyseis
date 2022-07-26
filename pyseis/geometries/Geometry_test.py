from pyseis.geometries import Geometry
import numpy as np
import pytest

N_SRC_FULL2D = 20
O_X_SRC_FULL2D = 10.0
D_X_SRC_FULL2D = 100.0
Z_SRC_FULL2D = 15.0
N_REC_FULL2D = 200
O_X_REC_FULL2D = 0.0
D_X_REC_FULL2D = 12.5
Z_REC_FULL2D = 11.0

N_SRC_STREAMER2D = 21
O_X_SRC_STREAMER2D = 10.0
D_X_SRC_STREAMER2D = 100.0
Z_SRC_STREAMER2D = 15.0
N_REC_STREAMER2D = 200
O_X_REC_STREAMER2D = 70.0
D_X_REC_STREAMER2D = 12.5
Z_REC_STREAMER2D = 11.0

N_T = 1000
N_WFLD_ELASTIC_2D = 5


@pytest.fixture
def random_elastic_streamer_data_2d():
  return np.random.random(
      (N_SRC_STREAMER2D, N_WFLD_ELASTIC_2D, N_REC_STREAMER2D, N_T))


@pytest.fixture
def random_elastic_full_data_2d():
  return np.random.random((N_SRC_FULL2D, N_WFLD_ELASTIC_2D, N_REC_FULL2D, N_T))


def test_Full2D(random_elastic_full_data_2d):
  full_geometry = Geometry.Full2D(n_src=N_SRC_FULL2D,
                                  o_x_src=O_X_SRC_FULL2D,
                                  d_x_src=D_X_SRC_FULL2D,
                                  z_src=Z_SRC_FULL2D,
                                  n_rec=N_REC_FULL2D,
                                  o_x_rec=O_X_REC_FULL2D,
                                  d_x_rec=D_X_REC_FULL2D,
                                  z_rec=Z_REC_FULL2D)

  src_positions = full_geometry.get_src_positions()
  assert src_positions.shape == (N_SRC_FULL2D, 2)
  assert src_positions[0, 0] == O_X_SRC_FULL2D
  assert (src_positions[1, 0] -
          src_positions[0, 0]) == pytest.approx(D_X_SRC_FULL2D)
  assert np.all(src_positions[:, 1] == Z_SRC_FULL2D)
  rec_positions = full_geometry.get_rec_positions()
  assert rec_positions.shape == (N_REC_FULL2D, 2)
  assert rec_positions[0, 0] == O_X_REC_FULL2D
  assert (rec_positions[1, 0] -
          rec_positions[0, 0]) == pytest.approx(D_X_REC_FULL2D)
  assert np.all(rec_positions[:, 1] == Z_REC_FULL2D)


def test_Streamer2D(random_elastic_streamer_data_2d):
  streamer_geometry = Geometry.Streamer2D(n_src=N_SRC_STREAMER2D,
                                          o_x_src=O_X_SRC_STREAMER2D,
                                          d_x_src=D_X_SRC_STREAMER2D,
                                          z_src=Z_SRC_STREAMER2D,
                                          n_rec=N_REC_STREAMER2D,
                                          o_x_rec=O_X_REC_STREAMER2D,
                                          d_x_rec=D_X_REC_STREAMER2D,
                                          z_rec=Z_REC_STREAMER2D)

  src_positions = streamer_geometry.get_src_positions()
  assert src_positions.shape == (N_SRC_STREAMER2D, 2)
  assert src_positions[0, 0] == O_X_SRC_STREAMER2D
  assert (src_positions[1, 0] -
          src_positions[0, 0]) == pytest.approx(D_X_SRC_STREAMER2D)
  assert np.all(src_positions[:, 1] == Z_SRC_STREAMER2D)

  rec_positions = streamer_geometry.get_rec_positions()
  assert rec_positions.shape == (N_SRC_STREAMER2D, N_REC_STREAMER2D, 2)
  for streamer_x_pos, src_pos in zip(rec_positions, src_positions):
    assert streamer_x_pos[0, 0] == src_pos[0] + O_X_REC_STREAMER2D
    assert (streamer_x_pos[1, 0] -
            streamer_x_pos[0, 0]) == pytest.approx(D_X_REC_STREAMER2D)
  assert np.all(streamer_x_pos[..., 1] == Z_REC_STREAMER2D)
