#include "scalar_cpu.h"

#include <stddef.h>
#include <string.h>

#include "scalar.h"

static inline ptrdiff_t location_index(
    const ptrdiff_t *__restrict__ const arr,
    const ptrdiff_t *__restrict__ const shape, const ptrdiff_t index);
static inline TYPE laplacian_1d(const TYPE *__restrict__ const arr,
                                const TYPE *__restrict__ const fd2,
                                const ptrdiff_t si);
static inline TYPE laplacian_2d(const TYPE *__restrict__ const arr,
                                const TYPE *__restrict__ const fd2,
                                const ptrdiff_t si, const ptrdiff_t size_x);
static inline TYPE laplacian_3d(const TYPE *__restrict__ const arr,
                                const TYPE *__restrict__ const fd2,
                                const ptrdiff_t si, const ptrdiff_t size_x,
                                const ptrdiff_t size_xy);
static inline TYPE z_deriv(const TYPE *__restrict__ const arr,
                           const TYPE *__restrict__ const fd1,
                           const ptrdiff_t si, const ptrdiff_t size_xy);
static inline TYPE y_deriv(const TYPE *__restrict__ const arr,
                           const TYPE *__restrict__ const fd1,
                           const ptrdiff_t si, const ptrdiff_t size_x);
static inline TYPE x_deriv(const TYPE *__restrict__ const arr,
                           const TYPE *__restrict__ const fd1,
                           const ptrdiff_t si);

void setup(const TYPE *__restrict__ const fd1,
           const TYPE *__restrict__ const fd2) {}


#if DIM == 2

/*
 * P(t+1/2)------v_x(t)
 * |
 * |
 * |
 * v_z(t)
 */

void propagate(TYPE *__restrict__ const wfn,        /* next wavefield */
               TYPE *__restrict__ const auxn,       /* next auxiliary */
               const TYPE *__restrict__ const wfc,  /* current wavefield */
               const TYPE *__restrict__ const wfp,  /* previous wavefield */
               const TYPE *__restrict__ const auxc, /* current auxiliary */
               const TYPE *__restrict__ const sigma,
               const TYPE *__restrict__ const model,
               const TYPE *__restrict__ const fd1, /* 1st difference coeffs */
               const TYPE *__restrict__ const fd2, /* 2nd difference coeffs */
               const ptrdiff_t *__restrict__ const shape,
               const ptrdiff_t *__restrict__ const pml_width,
               const ptrdiff_t num_shots, const TYPE dt) {
  const ptrdiff_t numel_shot = shape[0] * shape[1];
  const ptrdiff_t size_xy = shape[1];
  TYPE *__restrict__ const phizn = auxn;
  TYPE *__restrict__ const phiyn = auxn + num_shots * numel_shot;
  const TYPE *__restrict__ const phizc = auxc;
  const TYPE *__restrict__ const phiyc = auxc + num_shots * numel_shot;
  const TYPE *__restrict__ const sigmaz = sigma;
  const TYPE *__restrict__ const sigmay = sigma + shape[0];

  /* Combine pz and py */
#pragma omp parallel for default(none) collapse(2)
  for (ptrdiff_t si = 0; si < num_shots * shape[0] * shape[1]; si++) {
    p[si] = pz[si] + py[si];
  }

#pragma omp parallel for default(none) collapse(2)
  for (ptrdiff_t shot = 0; shot < num_shots; shot++) {
    for (ptrdiff_t z = ZPAD; z < shape[0] - ZPAD; z++) {
      for (ptrdiff_t y = YPAD; y < shape[1] - YPAD; y++) {
        const ptrdiff_t i = z * size_xy + y;
        const ptrdiff_t si = shot * numel_shot + i;

        /* Spatial finite differences */
        const TYPE lap = laplacian_2d(wfc, fd2, si, size_xy);
        const TYPE wfc_z = z_deriv(wfc, fd1, si, size_xy);
        const TYPE wfc_y = y_deriv(wfc, fd1, si, 1);
        const TYPE phizc_z = z_deriv(phizc, fd1, si, size_xy);
        const TYPE phiyc_y = y_deriv(phiyc, fd1, si, 1);

	const TYPE p_z = z_deriv(p, fd1, si, size_xy);
	const TYPE p_y = y_deriv(p, fd1, si, 1);

        /* Update vz, vy at t */
	/* TODO: rho is on 1 or 1/2 - DENISE is on 1?
	 * TODO: vel at edges? */
	vz[si] = vz[si] * (1 - dt * sigmaz[z]) - dt / rho[i] * p_z;
	vy[si] = vy[si] * (1 - dt * sigmay[y]) - dt / rho[i] * p_y;

	/* TODO: split into different functions */
		
	/* Add source */
	vz[sz,sx] += dt/rho[sz,sx]*sa[it];
		
	const TYPE vz_z = z_deriv(vz, fd1, si, size_xy);
	const TYPE vy_y = y_deriv(vy, fd1, si, 1);

	/* Update pz, px at t+1/2 */
	/* TODO: p at edges? */
	pz[si] = pz[si] * (1 - dt * sigmaz[z]) - dt * c**2 * rho[i] * vz_z;

      }
    }
  }
}

void imaging_condition(TYPE *__restrict__ const model_grad,
                       const TYPE *__restrict__ const current_wavefield,
                       const TYPE *__restrict__ const saved_wavefield,
                       const TYPE *__restrict__ const saved_wavefield_t,
                       const TYPE *__restrict__ const saved_wavefield_tt,
                       const TYPE *__restrict__ const sigma,
                       const ptrdiff_t *__restrict__ const shape,
                       const ptrdiff_t *__restrict__ const pml_width,
                       const ptrdiff_t num_shots) {
  if (model_grad == NULL) return; /* Not doing model inversion */

  const ptrdiff_t numel_shot = shape[0] * shape[1];
  const ptrdiff_t size_xy = shape[1];
  const TYPE *__restrict__ const sigmaz = sigma;
  const TYPE *__restrict__ const sigmay = sigma + shape[0];

  for (ptrdiff_t shot = 0; shot < num_shots; shot++) {
    for (ptrdiff_t z = ZPAD; z < shape[0] - ZPAD; z++) {
      for (ptrdiff_t y = YPAD; y < shape[1] - YPAD; y++) {
        const ptrdiff_t i = z * size_xy + y;
        const ptrdiff_t si = shot * numel_shot + i;

        model_grad[i] += current_wavefield[si] *
                         (saved_wavefield_tt[si] +
                          (sigmaz[z] + sigmay[y]) * saved_wavefield_t[si] +
                          sigmaz[z] * sigmay[y] * saved_wavefield[si]);
      }
    }
  }
}

static inline ptrdiff_t location_index(
    const ptrdiff_t *__restrict__ const arr,
    const ptrdiff_t *__restrict__ const shape, const ptrdiff_t index) {
  const ptrdiff_t z = arr[index * 2];
  const ptrdiff_t y = arr[index * 2 + 1];

  return z * shape[1] + y;
}

#else
#error "Must specify the dimension to be 2, e.g. -D DIM=2"
#endif /* DIM */

void add_sources(TYPE *__restrict__ const next_wavefield,
                 const TYPE *__restrict__ const model,
                 const TYPE *__restrict__ const source_amplitudes,
                 const ptrdiff_t *__restrict__ const source_locations,
                 const ptrdiff_t *__restrict__ const shape,
                 const ptrdiff_t num_shots,
                 const ptrdiff_t num_sources_per_shot) {
  for (ptrdiff_t shot = 0; shot < num_shots; shot++) {
    for (ptrdiff_t source = 0; source < num_sources_per_shot; source++) {
      const ptrdiff_t s = shot * num_sources_per_shot + source;
      const ptrdiff_t i = location_index(source_locations, shape, s);
      const ptrdiff_t si = shot * shape[0] * shape[1] * shape[2] + i;
      next_wavefield[si] += source_amplitudes[s] * model[i];
    }
  }
}

void add_scattering(TYPE *__restrict__ const next_scattered_wavefield,
                    const TYPE *__restrict__ const next_wavefield,
                    const TYPE *__restrict__ const current_wavefield,
                    const TYPE *__restrict__ const previous_wavefield,
                    const TYPE *__restrict__ const scatter,
                    const ptrdiff_t *__restrict__ const shape,
                    const ptrdiff_t num_shots) {
  const ptrdiff_t numel_shot = shape[0] * shape[1] * shape[2];
  const ptrdiff_t size_x = shape[2];
  const ptrdiff_t size_xy = shape[1] * shape[2];

  for (ptrdiff_t shot = 0; shot < num_shots; shot++) {
    for (ptrdiff_t z = ZPAD; z < shape[0] - ZPAD; z++) {
      for (ptrdiff_t y = YPAD; y < shape[1] - YPAD; y++) {
        for (ptrdiff_t x = XPAD; x < shape[2] - XPAD; x++) {
          const ptrdiff_t i = z * size_xy + y * size_x + x;
          const ptrdiff_t si = shot * numel_shot + i;
          const TYPE current_wavefield_tt =
              (next_wavefield[si] - 2 * current_wavefield[si] +
               previous_wavefield[si]); /* no dt^2 because of cancellation */
          next_scattered_wavefield[si] += current_wavefield_tt * scatter[i];
        }
      }
    }
  }
}

void record_receivers(TYPE *__restrict__ const receiver_amplitudes,
                      const TYPE *__restrict__ const current_wavefield,
                      const ptrdiff_t *__restrict__ const receiver_locations,
                      const ptrdiff_t *__restrict__ const shape,
                      const ptrdiff_t num_shots,
                      const ptrdiff_t num_receivers_per_shot) {
  if (receiver_amplitudes == NULL) return; /* no source inversion */

  for (ptrdiff_t shot = 0; shot < num_shots; shot++) {
    for (ptrdiff_t receiver = 0; receiver < num_receivers_per_shot;
         receiver++) {
      const ptrdiff_t r = shot * num_receivers_per_shot + receiver;
      const ptrdiff_t si = shot * shape[0] * shape[1] * shape[2] +
                           location_index(receiver_locations, shape, r);
      receiver_amplitudes[r] = current_wavefield[si];
    }
  }
}

void save_wavefields(TYPE *__restrict__ const current_saved_wavefield,
                     TYPE *__restrict__ const current_saved_wavefield_t,
                     TYPE *__restrict__ const current_saved_wavefield_tt,
                     const TYPE *__restrict__ const next_wavefield,
                     const TYPE *__restrict__ const current_wavefield,
                     const TYPE *__restrict__ const previous_wavefield,
                     const ptrdiff_t *__restrict__ const shape,
                     const ptrdiff_t num_shots, const TYPE dt,
                     const enum wavefield_save_strategy save_strategy) {
  if (save_strategy == STRATEGY_COPY) {
    const ptrdiff_t numel_shot = shape[0] * shape[1] * shape[2];
    const ptrdiff_t size_x = shape[2];
    const ptrdiff_t size_xy = shape[1] * shape[2];
    memcpy(current_saved_wavefield, current_wavefield,
           num_shots * shape[0] * shape[1] * shape[2] * sizeof(TYPE));

    for (ptrdiff_t shot = 0; shot < num_shots; shot++) {
      for (ptrdiff_t z = ZPAD; z < shape[0] - ZPAD; z++) {
        for (ptrdiff_t y = YPAD; y < shape[1] - YPAD; y++) {
          for (ptrdiff_t x = XPAD; x < shape[2] - XPAD; x++) {
            const ptrdiff_t i = z * size_xy + y * size_x + x;
            const ptrdiff_t si = shot * numel_shot + i;
            current_saved_wavefield_t[si] =
                (next_wavefield[si] - previous_wavefield[si]) / 2 / dt;
            current_saved_wavefield_tt[si] =
                (next_wavefield[si] - 2 * current_wavefield[si] +
                 previous_wavefield[si]) /
                dt / dt;
          }
        }
      }
    }
  }
}

void model_grad_scaling(TYPE *__restrict__ const model_grad,
                        const TYPE *__restrict__ const scaling,
                        const ptrdiff_t *__restrict__ const shape,
                        const ptrdiff_t *__restrict__ const pml_width) {
  if (model_grad == NULL) return; /* Not doing model inversion */

  const ptrdiff_t numel_shot = shape[0] * shape[1] * shape[2];

  for (ptrdiff_t i = 0; i < numel_shot; i++) {
    model_grad[i] *= scaling[i];
  }
}

static inline TYPE laplacian_1d(const TYPE *__restrict__ const arr,
                                const TYPE *__restrict__ const fd2,
                                const ptrdiff_t si) {
  return fd2[0] * arr[si] + fd2[1] * (arr[si + 1] + arr[si - 1]) +
         fd2[2] * (arr[si + 2] + arr[si - 2]);
}

static inline TYPE laplacian_2d(const TYPE *__restrict__ const arr,
                                const TYPE *__restrict__ const fd2,
                                const ptrdiff_t si, const ptrdiff_t size_x) {
  return fd2[0] * arr[si] + fd2[1] * (arr[si + size_x] + arr[si - size_x]) +
         fd2[2] * (arr[si + 2 * size_x] + arr[si - 2 * size_x]) +
         +fd2[3] * (arr[si + 1] + arr[si - 1]) +
         fd2[4] * (arr[si + 2] + arr[si - 2]);
}

static inline TYPE laplacian_3d(const TYPE *__restrict__ const arr,
                                const TYPE *__restrict__ const fd2,
                                const ptrdiff_t si, const ptrdiff_t size_x,
                                const ptrdiff_t size_xy) {
  return fd2[0] * arr[si] + fd2[1] * (arr[si + size_xy] + arr[si - size_xy]) +
         fd2[2] * (arr[si + 2 * size_xy] + arr[si - 2 * size_xy]) +
         +fd2[3] * (arr[si + size_x] + arr[si - size_x]) +
         fd2[4] * (arr[si + 2 * size_x] + arr[si - 2 * size_x]) +
         fd2[5] * (arr[si + 1] + arr[si - 1]) +
         fd2[6] * (arr[si + 2] + arr[si - 2]);
}

static inline TYPE z_deriv(const TYPE *__restrict__ const arr,
                           const TYPE *__restrict__ const fd1,
                           const ptrdiff_t si, const ptrdiff_t size_xy) {
  return fd1[0] * (arr[si + size_xy] - arr[si - size_xy]) +
         fd1[1] * (arr[si + 2 * size_xy] - arr[si - 2 * size_xy]);
}

static inline TYPE y_deriv(const TYPE *__restrict__ const arr,
                           const TYPE *__restrict__ const fd1,
                           const ptrdiff_t si, const ptrdiff_t size_x) {
  return fd1[0] * (arr[si + size_x] - arr[si - size_x]) +
         fd1[1] * (arr[si + 2 * size_x] - arr[si - 2 * size_x]);
}

static inline TYPE x_deriv(const TYPE *__restrict__ const arr,
                           const TYPE *__restrict__ const fd1,
                           const ptrdiff_t si) {
  return fd1[0] * (arr[si + 1] - arr[si - 1]) +
         fd1[1] * (arr[si + 2] - arr[si - 2]);
}
