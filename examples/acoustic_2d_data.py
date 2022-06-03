#!/opt/conda/bin/python
import numpy as np

from wave_equations.acoustic.AcousticIsotropic2D import AcousticIsotropic2D

# load model
vp_half_space = np.load('vp_model.npy')
n_x, n_z = vp_half_space.shape
d_x = 10.0
d_z = 15.0
print(f'\nLoaded vp_model.npy with n_x={n_x} and n_z={n_z}')
print(f'Set spatial sampling d_x={d_x} and d_z={d_z}')
# load wavelet

wavelet_arr = np.load('acoustic_ricker.npy')
n_t = len(wavelet_arr)
d_t = 0.01
print(f'\nLoaded acoustic_ricker.npy with n_t={n_t}.')
print(f'Set temporal sampling d_t={d_t}')

# make evenly spaced source and receiver positions
n_src = 20
o_x_src = 0.0
d_x_src = n_x * d_x // n_src
z_src = 10.0
n_rec = n_x
o_x_rec = 0.0
d_x_rec = d_x
z_rec = 10.0
src_x_locations = o_x_src + np.arange(n_src) * d_x_src
src_z_locations = np.ones_like(src_x_locations) * z_src
src_locations = np.array([src_x_locations, src_z_locations]).T
rec_x_locations = o_x_rec + np.arange(n_rec) * d_x_rec
rec_z_locations = np.ones_like(rec_x_locations) * z_rec
rec_locations = np.array([rec_x_locations, rec_z_locations]).T

# make wave equation operator
acoustic_2d = AcousticIsotropic2D(model=vp_half_space,
                                  model_sampling=(d_x, d_z),
                                  model_padding=(100, 100),
                                  wavelet=wavelet_arr,
                                  d_t=d_t,
                                  src_locations=src_locations,
                                  rec_locations=rec_locations,
                                  gpus=[0, 1, 2, 3])

# fwd model
data = acoustic_2d.fwd(vp_half_space)

# save data
np.save('acoustic_data.npy', data)
print(f'\nForward modeled {n_src} shots and saved to acoustic_data.npy')
