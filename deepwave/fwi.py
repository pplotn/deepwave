import math
 
def extract_subbatch(dataset, src_locs, rec_locs,
                     num_superbatches, num_batches, superbatch_idx,
                     batch_idx):
  num_shots = dataset.num_shots
  superbatch_size = math.ceil(num_shots / num_superbatches)
  batch_size = math.ceil(superbatch_size / num_batches)
  # 00.XX.00.00|00.AA.00.00|00.XX.00.00
  # ------------^ batch_idx * superbatch_size
  #             ---^ superbatch_idx * batch_size
  batch_start = min((batch_idx * superbatch_size +
                     superbatch_idx * batch_size),
                    num_shots)
  batch_end = min(batch_start + batch_size, num_shots)
  batch_slice = slice(batch_start, batch_end)
  batch_data = dataset.get_shots(batch_start, batch_end)
  batch_src_locs = src_locs[batch_slice]
  batch_rec_locs = rec_locs[batch_slice]
  return batch_data, batch_src_locs, batch_rec_locs

def prepare_data(data, cutoff_freq, dt_in, dt_out):
  data = filter_data(data, cutoff_freq, dt_in)
  data = resample_data(data, dt_in, dt_out)
  data = torch.Tensor(data).permute(2, 0, 1) # move time to 1st axis
  return data
    
def filter_data(data, cutoff_freq, dt, order=6):
  nyquist = 1 / dt / 2
  normal_cutoff = cutoff_freq / nyquist
  b, a = scipy.signal.butter(order, normal_cutoff)
  filtered_data = scipy.signal.filtfilt(b, a, data, axis=-1, method='gust')
  return filtered_data

def calc_data_decimation(cutoff_freq, dt, start_time=0.0):
  max_freq = cutoff_freq * 2
  decimation_factor = max(1, int((1 / max_freq / 2) / dt))
  new_dt = decimation_factor * dt
  new_start_time = int(start_time / new_dt) * new_dt # keep sample at t=0
  return new_dt, new_start_time

def calc_model_decimation(model, cutoff_freq, dx):
  max_freq = cutoff_freq * 2
  min_speed = model.min().item()
  min_wavelength = min_speed / max_freq
  # want at least 4 cells per wavelength
  decimation_factor_z = max(1, int((min_wavelength / 4) / dx[0]))
  decimation_factor_y = max(1, int((min_wavelength / 4) / dx[1]))
  new_dx = [decimation_factor_z * dx[0], decimation_factor_y * dx[1]]
  return new_dx

def resample_data(data, dt_in, dt_out, start_time_in=0, start_time_out=0,
                  axis=-1, num_steps_out=None):
  num_steps = data.shape[-1]
  t_in = np.arange(num_steps) * dt_in + start_time_in
  if num_steps_out is None:
    resample_factor = dt_in / dt_out
    num_steps_out = int(num_steps * resample_factor)
  t_out = np.arange(num_steps_out) * dt_out + start_time_out
  resampled_data = scipy.interpolate.interp1d(t_in, data, kind='cubic',
                                              axis=axis,
                                              fill_value='extrapolate',
                                              assume_sorted=True)(t_out)
  return resampled_data

def resample_model(model, dx_in, dx_out, nx_out=None):
  nx = model.shape
  z_in = np.arange(nx[0]) * dx_in[0]
  y_in = np.arange(nx[1]) * dx_in[1]
  if nx_out is None:
    resample_factor_z = dx_in[0] / dx_out[0]
    resample_factor_y = dx_in[1] / dx_out[1]
    nx_out = [int(nx[0] * resample_factor_z), int(nx[1] * resample_factor_y)]
  z_out = np.arange(nx_out[0]) * dx_out[0]
  y_out = np.arange(nx_out[1]) * dx_out[1]
  interp = scipy.interpolate.RectBivariateSpline(z_in, y_in, model)
  resampled_model = interp(z_out, y_out)
  return resampled_model

def invert_freq(dataset, src_amp_init, src_start_time, model_init, cutoff_freq,
                num_epochs, num_superbatches, num_batches, pml_width=10,
                survey_pad=500, lr_model=1e5, lr_src_amp=0.0001):
  
  # Make copies of the initial source amplitude and model for updating
  src_amp = src_amp_init.copy()
  model = model_init.copy()
  
  # Resample source amplitude, and model for specified freq
  dt, new_start_time = calc_data_decimation(cutoff_freq, dataset.dt,
                                            src_start_time)
  src_amp = resample_data(src_amp, dataset.dt, dt,
                          src_start_time, new_start_time)
  dx = calc_model_decimation(model, cutoff_freq, dataset.dx)
  model = resample_model(model, dataset.dx, dx)
  
  # Convert to PyTorch Tensors and send to GPU (if available)
  if torch.cuda.is_available():
    device = torch.device('cuda') 
  else:
    device = torch.device('cpu')
  model_t = torch.Tensor(model[np.newaxis]).to(device)
  model_t.requires_grad_()
  src_amp_t = torch.Tensor(src_amp).to(device)
  src_amp_t.requires_grad_()
  src_locs_t = torch.Tensor(dataset.src_locs).to(device)
  rec_locs_t = torch.Tensor(dataset.rec_locs).to(device)
  dx_t = torch.Tensor(dx)
  pml_width_t = torch.Tensor([0, 1, 1, 1, 0, 0]) * pml_width # free surface

  
  # Set-up inversion
  criterion = torch.nn.MSELoss()
  optimizer = torch.optim.Adam([{'params': [model_t], 'lr': lr_model},
                                {'params': [src_amp_t], 'lr': lr_src_amp}])
  tail = deepwave.utils.Tail()
  prop = deepwave.scalar.Propagator(model_t, dx_t, pml_width=pml_width_t,
                                    survey_pad=survey_pad)

  # Inversion loop
  for epoch in range(num_epochs):
    epoch_loss = 0.0
    for superbatch_idx in range(4):#num_superbatches):
      optimizer.zero_grad()
      for batch_idx in range(num_batches):
        batch_data_true, batch_src_locs, batch_rec_locs = \
          extract_subbatch(dataset, src_locs_t, rec_locs_t,
                           num_superbatches, num_batches, superbatch_idx,
                           batch_idx)
        # Make a copy of the source amplitude for each shot
        batch_src_amps = \
          src_amp_t.reshape(-1, 1, 1).repeat(1, *batch_src_locs.shape[:2])
        batch_data_true = prepare_data(batch_data_true, cutoff_freq, dataset.dt,
                                       dt)
        batch_data_true = batch_data_true.to(device) # send data batch to GPU

        batch_data_pred = prop(batch_src_amps, batch_src_locs, batch_rec_locs,
                               dt)
        loss = criterion(*tail(batch_data_pred, batch_data_true))
        loss.backward()
        epoch_loss += loss.detach().item()
      optimizer.step()
    print('Epoch:', epoch, 'Loss: ', epoch_loss)
    
  # Resample outputs and convert to NumPy
  src_amp = resample_data(src_amp_t.detach().cpu().numpy(), dt, dataset.dt,
                          new_start_time, src_start_time,
                          num_steps_out=len(src_amp_init))
  model = resample_model(model_t[0].detach().cpu().numpy(), dx, dataset.dx,
                         nx_out=model_init.shape)
      
  return src_amp, model
