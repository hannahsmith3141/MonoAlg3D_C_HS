#ifndef MONOALG3D_MODEL_TORORD_DYNCL_PHICAL_IKCA_MIXED_APICOBASAL_INFARCTIONREMOD_RZ1_H
#define MONOALG3D_MODEL_TORORD_DYNCL_PHICAL_IKCA_MIXED_APICOBASAL_INFARCTIONREMOD_RZ1_H

// TOMEK, Jakub and BUENO-OROVIO, Alfonso and RODRIGUEZ, Blanca
// ToR-ORd-dynCl: an update of the ToR-ORd model of human ventricular cardiomyocyte with dynamic intracellular chloride
//  bioRxiv, 2020.

#include "../model_common.h"

#define NEQ 45
#define INITIAL_V (-89.0f)

#ifdef __CUDACC__

#include "../../gpu_utils/gpu_utils.h"

__global__ void kernel_set_model_initial_conditions(real *sv, int num_volumes, real *extra_data);

__global__ void solve_gpu(real cur_time, real dt, real *sv, real* stim_currents,
                          uint32_t *cells_to_solve, uint32_t num_cells_to_solve,
                          int num_steps, real *extra_data);

inline __device__ void RHS_gpu(real *sv, real *rDY, real stim_current, int thread_id, real dt, int layer, int infarct_zone, int infarct_stage, real apicobasal);
inline __device__ void solve_forward_euler_gpu_adpt(real *sv, real stim_curr, real final_time, int thread_id, int layer, int infarct_zone, int infarct_stage, real apicobasal);

#endif

void RHS_cpu(const real *sv, real *rDY, real stim_current, real dt, int layer, int infarct_zone, int infarct_stage, real apicobasal);
inline void solve_forward_euler_cpu_adpt(real *sv, real stim_curr, real final_time, int thread_id, int layer, int infarct_zone, int infarct_stage, real apicobasal);

void solve_model_ode_cpu(real dt, real *sv, real stim_current, int layer, int infarct_zone, int infarct_stage, real apicobasal);

#endif //MONOALG3D_MODEL_TORORD_DYNCL_PHICAL_IKCA_MIXED_APICOBASAL_INFARCTIONREMOD_RZ1_H
