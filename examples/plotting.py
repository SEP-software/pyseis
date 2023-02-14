import numpy as np
import matplotlib.pyplot as plt


def plot_model(model,
               sampling=(5, 5),
               origins=(0.0,0.0),
               pclip=99,
               figsize=(15, 5),
               cmap='jet',
               title=None,
               xlabel='x pos (m)',
               zlabel='z pos (m)',
               cbar=False,
               cbar_label='$v_p$ (m/s)',
               src_locations=None,
               rec_locations=None,
               vlims=None):
  fig, ax = plt.subplots(figsize=figsize)

  x_axis = origins[0] + sampling[0] * np.arange(model.shape[0])
  z_axis = origins[1] + sampling[1] * np.arange(model.shape[1])

  clip = np.percentile(model, pclip)
  vmin=0
  vmax=clip
  if vlims:
    vmin = vlims[0]
    vmax = vlims[1]
  pc = plt.pcolormesh(x_axis,
                      z_axis,
                      model.T,
                      shading='nearest',
                      vmin=vmin,
                      vmax=vmax,
                      cmap=cmap)

  plt.gca().invert_yaxis()

  ax.set_xlabel(xlabel)
  ax.set_ylabel(zlabel)

  ax.set_title(title)

  if cbar:
    cbar = plt.colorbar(pc)
    cbar.set_label(cbar_label)

  if rec_locations is not None:
    plt.scatter(rec_locations[:, 0],
                rec_locations[:, 1],
                marker='v',
                s=10,
                c='k',
                label='rec pos')
    _ = plt.xlim([0, sampling[0] * (model.shape[0] - 1)])
    plt.legend()

  if src_locations is not None:
    plt.scatter(src_locations[:, 0],
                src_locations[:, 1],
                marker='*',
                s=10,
                c='r',
                label='src pos')
    _ = plt.xlim([0, sampling[0] * (model.shape[0] - 1)])
    plt.legend()


def plot_wavelet(wavelet, d_t, title='wavelet', figsize=(10, 2.5)):
  wavelet_fft = np.fft.rfft(wavelet)
  freqs = np.fft.rfftfreq(len(wavelet), d_t)
  t = d_t * np.arange(len(wavelet))

  fig, axs = plt.subplots(1, 2, figsize=figsize)

  axs[0].plot(t, wavelet)
  axs[0].set_xlabel('time (s)')
  axs[0].set_title(title)

  axs[1].plot(freqs, np.abs(wavelet_fft))
  axs[1].set_xlabel('freq (1/s)')
  axs[1].set_title(f'{title} FFT')


def plot_data(data,
              d_t,
              src_locations=None,
              rec_locations=None,
              n_shots=6,
              title=None,
              figsize=(3, 5),
              pclip=99,
              clip=None,
              cmap='gray',
              ylabel='time (s)',
              xlabel='rec positions (m)'):
  fig, axs = plt.subplots(1,
                          n_shots,
                          figsize=(figsize[0] * n_shots, figsize[1]))

  skip = data.shape[0] // n_shots
  data = data[::skip]
  if src_locations is not None:
    src_locations = src_locations[::skip]
  if rec_locations is None:
    rec_locations = np.arange(data.shape[1])
    xlabel = 'rec #'
    
  t_axis = d_t * np.arange(data.shape[-1])

  if clip is None:
    clip = np.percentile(np.abs(data), pclip)
  for i, (ax, shot) in enumerate(zip(axs, data)):
    ax.pcolormesh(rec_locations,
                  t_axis,
                  shot.T,
                  shading='nearest',
                  vmin=-clip,
                  vmax=clip,
                  cmap=cmap)
    ax.invert_yaxis()
    if src_locations is None:
      ax.set_title(f'shot# {i*skip}')
    else:
      ax.set_title(f'shot pos %.2f' % src_locations[i])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if title:
        fig.suptitle(title)
