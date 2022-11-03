from mock import patch
import pytest
import numpy as np
from pyseis.wave_equations import acoustic_isotropic
from pyseis.wavelets.acoustic import Acoustic2D
from pyseis import inversion

# @pytest.mark.gpu
# def test_init_fwi():
#   # Arrange
#   acoustic_2d = acoustic_isotropic.AcousticIsotropic2D(
#       model=vp_model_half_space,
#       model_sampling=(D_X, D_Z),
#       model_padding=(N_X_PAD, N_Z_PAD),
#       wavelet=ricker_wavelet,
#       d_t=D_T,
#       src_locations=src_locations,
#       rec_locations=fixed_rec_locations,
#       gpus=I_GPUS)
#   # init
#   fwi_prob = inversion.Fwi(wave_eq_solver=acoustic_2d
#       data=data,
#       num_iter=3)

# @pytest.mark.gpu
# def test_run_fwi():
#   # Arrange
#   acoustic_2d = acoustic_isotropic.AcousticIsotropic2D(
#       model=vp_model_half_space,
#       model_sampling=(D_X, D_Z),
#       model_padding=(N_X_PAD, N_Z_PAD),
#       wavelet=ricker_wavelet,
#       d_t=D_T,
#       src_locations=src_locations,
#       rec_locations=fixed_rec_locations,
#       gpus=I_GPUS)
#   # init
#   fwi_prob = inversion.Fwi(wave_eq_solver=acoustic_2d
#       data=data,
#       num_iter=3)
#   # run
#   inv_model, hist = fwi_prob.run()