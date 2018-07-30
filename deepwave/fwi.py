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

    return (torch.as_tensor(batch_data), torch.as_tensor(batch_src_locs),
            torch.as_tensor(batch_rec_locs))


def pool_data(data, num_pool, dt, start_time=0):
    padding = int(start_time / dt) % num_pool  # t=0 starts a new block
    pool = torch.nn.AvgPool1D(num_pool, padding=padding)
    data_pool = pool(data)
    return data_pool, dt * num_pool


def pool_model(model, dt, dx, min_cells_per_wavelength=4):
    ndims = model.dims()
    if ndims == 2:
        poolfunc = torch.nn.AvgPool1D
    elif ndims == 3:
        poolfunc = torch.nn.AvgPool2D
    elif ndims == 4:
        poolfunc = torch.nn.AvgPool3D
    else:
        raise ValueError
    min_vel = model.min().item()
    max_freq = 1 / dt / 2
    min_wavelength = min_vel / max_freq
    max_dx = min_wavelength / min_cells_per_wavelength
    num_pool = torch.ceil(max_dx / torch.as_tensor(dx)).long()
    pool = poolfunc(num_pool)
    model_pool = poolfunc(model.reshape(1, *model.shape)).reshape(model.shape)
    return model_pool, dx * num_pool


# TODO:
# * pool src and data, and use to determine and apply model pool
# * invert increasing time
# * separate into train and validate datasets
# * use validate dataset to determine config changes
# * total variation regularization
# * supershots


def fwi(dataset, src_amp_init, src_start_time, model_init,
        num_pool_data,
        num_epochs, num_superbatches, num_batches, pml_width=10,
        survey_pad=500, lr_model=1e5, lr_src_amp=0.0001,
        invert_source=True, invert_model=True):

    # Check if GPU is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Convert inputs to PyTorch Tensors if they are not already
    src_amp_init = torch.as_tensor(src_amp_init)
    model_init = torch.as_tensor(model_init)

    # Add extra dimension to model if needed
    if model_init.shape[0] != 1:
        model_init = model_init.reshape(1, *model_init.shape)

    # Make copies of the initial source amplitude and model for updating     #
    # and send to GPU (if available)
    src_amp = src_amp_init.data.clone().to(device)
    if invert_source:
        src_amp.requires_grad_()
    model = model_init.data.clone().to(device)
    if invert_model:
        model.requires_grad_()

    # Set-up inversion
    criterion = torch.nn.MSELoss()
    params = []
    if invert_source:
        params.append({'params': [src_amp], 'lr': lr_src_amp})
    if invert_model:
        params.append({'params': [model], 'lr': lr_model})
    optimizer = torch.optim.Adam(params)
    tail = deepwave.utils.Tail()
    pml_width = torch.Tensor([0, 1, 1, 1, 0, 0]) * pml_width  # free surface

    # Inversion loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for superbatch_idx in range(num_superbatches):
            optimizer.zero_grad()
            for batch_idx in range(num_batches):

                batch_data_true, batch_src_locs, batch_rec_locs = \
                    extract_subbatch(dataset, src_locs, rec_locs,
                                     num_superbatches, num_batches,
                                     superbatch_idx, batch_idx)

                # Pool data, source amplitude, and model
                batch_data_true, dt = pool_data(batch_data_true, num_pool_data,
                                                dataset.dt)
                src_amp_pool, _ = pool_data(src_amp.reshape(1, 1, -1),
                                            num_pool_data, dataset.dt,
                                            src_start_time)
                model_pool, dx = pool_model(model, dt, dataset.dx)

                # Move time to first axis of data
                batch_data_true = batch_data_true.permute(2, 0, 1)

                # Send to GPU (if available)
                batch_data_true = batch_data_true.to(device)
                batch_src_locs = batch_src_locs.to(device)
                batch_rec_locs = batch_rec_locs.to(device)

                # Create propagator
                prop = deepwave.scalar.Propagator(model_pool, dx,
                                                  pml_width=pml_width,
                                                  survey_pad=survey_pad)

                # Make a copy of the source amplitude for each shot
                batch_src_amps = \
                    src_amp_pool.reshape(-1, 1, 1)\
                    .repeat(1, *batch_src_locs.shape[:2])

                batch_data_pred = prop(
                    batch_src_amps, batch_src_locs, batch_rec_locs, dt)
                loss = criterion(*tail(batch_data_pred, batch_data_true))
                loss.backward()
                epoch_loss += loss.detach().item()
            optimizer.step()
        print('Epoch:', epoch, 'Loss: ', epoch_loss)

    return src_amp.detach(), model.detach()
