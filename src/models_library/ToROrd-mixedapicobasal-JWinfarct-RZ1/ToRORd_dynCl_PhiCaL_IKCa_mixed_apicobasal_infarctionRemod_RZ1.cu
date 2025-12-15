#include "ToRORd_dynCl_PhiCaL_IKCa_mixed_apicobasal_infarctionRemod_RZ1.h"
#include "../../gpu_utils/gpu_utils.h"
#include <stddef.h>
#include <stdint.h>

__constant__  size_t pitch;
__constant__  real abstol;
__constant__  real reltol;
__constant__  real max_dt;
__constant__  real min_dt;
__constant__  uint8_t use_adpt;

size_t pitch_h;

extern "C" SET_ODE_INITIAL_CONDITIONS_GPU(set_model_initial_conditions_gpu) {

    uint8_t use_adpt_h = (uint8_t)solver->adaptive;

    check_cuda_error(cudaMemcpyToSymbol(use_adpt, &use_adpt_h, sizeof(uint8_t)));
    log_info("Using ToRORd_dynCl_mixed_endo_mid_epi GPU model\n");

    uint32_t num_volumes = solver->original_num_cells;

    if(use_adpt_h) {
        real reltol_h = solver->rel_tol;
        real abstol_h = solver->abs_tol;
        real max_dt_h = solver->max_dt;
        real min_dt_h = solver->min_dt;

        check_cuda_error(cudaMemcpyToSymbol(reltol, &reltol_h, sizeof(real)));
        check_cuda_error(cudaMemcpyToSymbol(abstol, &abstol_h, sizeof(real)));
        check_cuda_error(cudaMemcpyToSymbol(max_dt, &max_dt_h, sizeof(real)));
        check_cuda_error(cudaMemcpyToSymbol(min_dt, &min_dt_h, sizeof(real)));
        log_info("Using Adaptive Euler model to solve the ODEs\n");
    } else {
        log_info("Using Euler model to solve the ODEs\n");
    }

    // Execution configuration
    const int GRID = (num_volumes + BLOCK_SIZE - 1) / BLOCK_SIZE;

    size_t size = num_volumes * sizeof(real);

    if(use_adpt_h)
        check_cuda_error(cudaMallocPitch((void **)&(solver->sv), &pitch_h, size, (size_t)NEQ + 3));
    else
        check_cuda_error(cudaMallocPitch((void **)&(solver->sv), &pitch_h, size, (size_t)NEQ));

    check_cuda_error(cudaMemcpyToSymbol(pitch, &pitch_h, sizeof(size_t)));

    //Get the extra_data array
    real *extra_data = NULL;
    real *extra_data_device = NULL;

    if(solver->ode_extra_data) {
        extra_data = (real*)solver->ode_extra_data;
        check_cuda_error(cudaMalloc((void **)&extra_data_device, solver->extra_data_size));
        check_cuda_error(cudaMemcpy(extra_data_device, extra_data, solver->extra_data_size, cudaMemcpyHostToDevice));
    }
    kernel_set_model_initial_conditions<<<GRID, BLOCK_SIZE>>>(solver->sv, num_volumes, extra_data_device);

    check_cuda_error(cudaPeekAtLastError());
    cudaDeviceSynchronize();
    check_cuda_error(cudaFree(extra_data_device));
    return pitch_h;
}

extern "C" SOLVE_MODEL_ODES(solve_model_odes_gpu) {
    size_t num_cells_to_solve = ode_solver->num_cells_to_solve;
    uint32_t * cells_to_solve = ode_solver->cells_to_solve;
    real *sv = ode_solver->sv;
    real dt = ode_solver->min_dt;
    uint32_t num_steps = ode_solver->num_steps;

    // execution configuration
    const int GRID = ((int)num_cells_to_solve + BLOCK_SIZE - 1) / BLOCK_SIZE;

    size_t stim_currents_size = sizeof(real) * num_cells_to_solve;
    size_t cells_to_solve_size = sizeof(uint32_t) * num_cells_to_solve;

    real *stims_currents_device;
    check_cuda_error(cudaMalloc((void **)&stims_currents_device, stim_currents_size));
    check_cuda_error(cudaMemcpy(stims_currents_device, stim_currents, stim_currents_size, cudaMemcpyHostToDevice));

    // the array cells to solve is passed when we are using and adaptive mesh
    uint32_t *cells_to_solve_device = NULL;
    if(cells_to_solve != NULL) {
        check_cuda_error(cudaMalloc((void **)&cells_to_solve_device, cells_to_solve_size));
        check_cuda_error(cudaMemcpy(cells_to_solve_device, cells_to_solve, cells_to_solve_size, cudaMemcpyHostToDevice));
    }

    //Get the extra_data array
    real *extra_data = NULL;
    real *extra_data_device = NULL;
    if(ode_solver->ode_extra_data) {
        extra_data = (real*)ode_solver->ode_extra_data;
        check_cuda_error(cudaMalloc((void **)&extra_data_device, ode_solver->extra_data_size));
        check_cuda_error(cudaMemcpy(extra_data_device, extra_data, ode_solver->extra_data_size, cudaMemcpyHostToDevice));
    } else {
        log_error_and_exit("You need to specify a mask function when using a mixed model!\n");
    }
    solve_gpu<<<GRID, BLOCK_SIZE>>>(current_t, dt, sv, stims_currents_device, cells_to_solve_device, num_cells_to_solve, num_steps, extra_data_device);

    check_cuda_error(cudaPeekAtLastError());

    check_cuda_error(cudaFree(stims_currents_device));
    if(cells_to_solve_device) check_cuda_error(cudaFree(cells_to_solve_device));
    if(extra_data_device) check_cuda_error(cudaFree(extra_data_device));
}

__global__ void kernel_set_model_initial_conditions(real *sv, int num_volumes, real *extra_data) {
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    int layer = (int) extra_data[thread_id];
    int infarct_zone = (int) extra_data[thread_id + num_volumes];
    int infarct_stage = (int) extra_data[3 * num_volumes];

    if (thread_id < num_volumes) {
        #include "ToRORd_dynCl_PhiCaL_IKCa_mixed_apicobasal_infarctionRemod_SS_RZ1.common.c"

        *((real *)((char *)sv + pitch *  0)+thread_id) = v;
        *((real *)((char *)sv + pitch *  1)+thread_id) = nai;
        *((real *)((char *)sv + pitch *  2)+thread_id) = nass;
        *((real *)((char *)sv + pitch *  3)+thread_id) = ki;
        *((real *)((char *)sv + pitch *  4)+thread_id) = kss;
        *((real *)((char *)sv + pitch *  5)+thread_id) = cai;
        *((real *)((char *)sv + pitch *  6)+thread_id) = cass;
        *((real *)((char *)sv + pitch *  7)+thread_id) = cansr;
        *((real *)((char *)sv + pitch *  8)+thread_id) = cajsr;
        *((real *)((char *)sv + pitch *  9)+thread_id) = m;
        *((real *)((char *)sv + pitch * 10)+thread_id) = hp;
        *((real *)((char *)sv + pitch * 11)+thread_id) = h;
        *((real *)((char *)sv + pitch * 12)+thread_id) = j;
        *((real *)((char *)sv + pitch * 13)+thread_id) = jp;
        *((real *)((char *)sv + pitch * 14)+thread_id) = mL;
        *((real *)((char *)sv + pitch * 15)+thread_id) = hL;
        *((real *)((char *)sv + pitch * 16)+thread_id) = hLp;
        *((real *)((char *)sv + pitch * 17)+thread_id) = a;
        *((real *)((char *)sv + pitch * 18)+thread_id) = iF;
        *((real *)((char *)sv + pitch * 19)+thread_id) = iS;
        *((real *)((char *)sv + pitch * 20)+thread_id) = ap;
        *((real *)((char *)sv + pitch * 21)+thread_id) = iFp;
        *((real *)((char *)sv + pitch * 22)+thread_id) = iSp;
        *((real *)((char *)sv + pitch * 23)+thread_id) = d;
        *((real *)((char *)sv + pitch * 24)+thread_id) = ff;
        *((real *)((char *)sv + pitch * 25)+thread_id) = fs;
        *((real *)((char *)sv + pitch * 26)+thread_id) = fcaf;
        *((real *)((char *)sv + pitch * 27)+thread_id) = fcas;
        *((real *)((char *)sv + pitch * 28)+thread_id) = jca;
        *((real *)((char *)sv + pitch * 29)+thread_id) = nca;
        *((real *)((char *)sv + pitch * 30)+thread_id) = nca_i;
        *((real *)((char *)sv + pitch * 31)+thread_id) = ffp;
        *((real *)((char *)sv + pitch * 32)+thread_id) = fcafp;
        *((real *)((char *)sv + pitch * 33)+thread_id) = xs1;
        *((real *)((char *)sv + pitch * 34)+thread_id) = xs2;
        *((real *)((char *)sv + pitch * 35)+thread_id) = Jrel_np;
        *((real *)((char *)sv + pitch * 36)+thread_id) = CaMKt;
        *((real *)((char *)sv + pitch * 37)+thread_id) = ikr_c0;
        *((real *)((char *)sv + pitch * 38)+thread_id) = ikr_c1;
        *((real *)((char *)sv + pitch * 39)+thread_id) = ikr_c2;
        *((real *)((char *)sv + pitch * 40)+thread_id) = ikr_o;
        *((real *)((char *)sv + pitch * 41)+thread_id) = ikr_i;
        *((real *)((char *)sv + pitch * 42)+thread_id) = Jrel_p;
        *((real *)((char *)sv + pitch * 43)+thread_id) = cli;
        *((real *)((char *)sv + pitch * 44)+thread_id) = clss;

        if(use_adpt) {
            *((real *)((char *)sv + pitch * (NEQ)) + thread_id) = min_dt;   // dt
            *((real *)((char *)sv + pitch * (NEQ+1)) + thread_id) = 0.0;    // time_new
            *((real *)((char *)sv + pitch * (NEQ+2)) + thread_id) = 0.0;    // previous dt
        }
    }
}

// Solving the model for each cell in the tissue matrix ni x nj
__global__ void solve_gpu(real cur_time, real dt, real *sv, real* stim_currents, uint32_t *cells_to_solve,
                            uint32_t num_cells_to_solve, int num_steps, real *extra_data) {
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    int sv_id;
    int layer = (int) extra_data[thread_id];
    int infarct_zone = (int) extra_data[thread_id + num_cells_to_solve];
    real apicobasal = extra_data[thread_id + (2*num_cells_to_solve)];
    int infarct_stage = (int) extra_data[3 * num_cells_to_solve];

    // Each thread solves one cell model
    if(thread_id < num_cells_to_solve) {
        if(cells_to_solve)
            sv_id = cells_to_solve[thread_id];
        else
            sv_id = thread_id;

        if(!use_adpt) {
            real rDY[NEQ];

            for(int n = 0; n < num_steps; ++n) {

                RHS_gpu(sv, rDY, stim_currents[thread_id], sv_id, dt, layer, infarct_zone, infarct_stage, apicobasal);

                // Full Explicit Euler
                //for(int i = 0; i < NEQ; i++) {
                //    *((real *)((char *)sv + pitch * i) + sv_id) = dt * rDY[i] + *((real *)((char *)sv + pitch * i) + sv_id);
                //}

                //Euler with Rush-Larsen
                //for(int i = 0; i < 9; i++)
                //    *((real *)((char *)sv + pitch * i) + sv_id) = dt * rDY[i] + *((real *)((char *)sv + pitch * i) + sv_id);
                //for(int i = 9; i < 29; i++)
                //    *((real *)((char *)sv + pitch * i) + sv_id) = rDY[i];
                //for(int i = 29; i < 31; i++)
                //    *((real *)((char *)sv + pitch * i) + sv_id) = dt * rDY[i] + *((real *)((char *)sv + pitch * i) + sv_id);
                //for(int i = 31; i < 36; i++)
                //    *((real *)((char *)sv + pitch * i) + sv_id) = rDY[i];
                //for(int i = 36; i < 42; i++)
                //    *((real *)((char *)sv + pitch * i) + sv_id) = dt * rDY[i] + *((real *)((char *)sv + pitch * i) + sv_id);
                //for(int i = 42; i < 43; i++)
                //    *((real *)((char *)sv + pitch * i) + sv_id) = rDY[i];
                for(int i = 0; i < NEQ; i++)
                    *((real *)((char *)sv + pitch * i) + sv_id) = rDY[i];
            }
        } else {
            solve_forward_euler_gpu_adpt(sv, stim_currents[thread_id], cur_time + max_dt, sv_id, layer, infarct_zone, infarct_stage, apicobasal);
        }
    }
}

inline __device__ void solve_forward_euler_gpu_adpt(real *sv, real stim_curr, real final_time, int thread_id, int layer, int infarct_zone, int infarct_stage, real apicobasal) {
  #define DT *((real *)((char *)sv + pitch * NEQ) + thread_id)
  #define TIME_NEW *((real *)((char *)sv + pitch * (NEQ + 1)) + thread_id)
  #define PREVIOUS_DT *((real *)((char *)sv + pitch * (NEQ + 2)) + thread_id)

  real rDY[NEQ];

  real _tolerances_[NEQ];
  real _aux_tol = 0.0;
  real dt = DT;
  real time_new = TIME_NEW;
  real previous_dt = PREVIOUS_DT;

  real edos_old_aux_[NEQ];
  real edos_new_euler_[NEQ];
  real _k1__[NEQ];
  real _k2__[NEQ];
  real _k_aux__[NEQ];
  real sv_local[NEQ];

  const real _beta_safety_ = 0.8;

  const real __tiny_ = pow(abstol, 2.0f);

  // dt = ((time_new + dt) > final_time) ? (final_time - time_new) : dt;
  if(time_new + dt > final_time) {
      dt = final_time - time_new;
  }

  //#pragma unroll
  for(int i = 0; i < NEQ; i++) {
      sv_local[i] = *((real *)((char *)sv + pitch * i) + thread_id);
  }

  RHS_gpu(sv_local, rDY, stim_curr, thread_id, dt, layer, infarct_zone, infarct_stage, apicobasal);
  time_new += dt;

  //#pragma unroll
  for(int i = 0; i < NEQ; i++) {
      _k1__[i] = rDY[i];
  }

  int count = 0;

  int count_limit = (final_time - time_new) / min_dt;

  int aux_count_limit = count_limit + 2000000;

  if(aux_count_limit > 0) {
      count_limit = aux_count_limit;
  }

  while(1) {
      for(int i = 0; i < NEQ; i++) {
          // stores the old variables in a vector
          edos_old_aux_[i] = sv_local[i];
          // //computes euler method
          edos_new_euler_[i] = _k1__[i];
          // steps ahead to compute the rk2 method
          sv_local[i] = edos_new_euler_[i];
      }

      time_new += dt;
      RHS_gpu(sv_local, rDY, stim_curr, thread_id, dt, layer, infarct_zone, infarct_stage, apicobasal);
      time_new -= dt; // step back

      real greatestError = 0.0, auxError = 0.0;
      for(int i = 0; i < NEQ; i++) {
          // stores the new evaluation
          _k2__[i] = rDY[i];
          _aux_tol = fabs(edos_new_euler_[i]) * reltol;
          _tolerances_[i] = (abstol > _aux_tol) ? abstol : _aux_tol;
          // finds the greatest error between  the steps
          auxError = fabs(((dt / 2.0) * (_k1__[i] - _k2__[i])) / _tolerances_[i]);
          greatestError = (auxError > greatestError) ? auxError : greatestError;
      }

      /// adapt the time step
      greatestError += __tiny_;
      previous_dt = dt;
      /// adapt the time step
      dt = _beta_safety_ * dt * sqrt(1.0f / greatestError);

      if(time_new + dt > final_time) {
          dt = final_time - time_new;
      }

      // it doesn't accept the solution
      if(count < count_limit && (greatestError >= 1.0f)) {
          // restore the old values to do it again
          for(int i = 0; i < NEQ; i++) {
              sv_local[i] = edos_old_aux_[i];
          }
          count++;
          // throw the results away and compute again
      } else {
          count = 0;

          if(dt < min_dt) {
              dt = min_dt;
          } else if(dt > max_dt && max_dt != 0) {
              dt = max_dt;
          }

          if(time_new + dt > final_time) {
              dt = final_time - time_new;
          }

          // change vectors k1 e k2 , para que k2 seja aproveitado como k1 na proxima iteração
          for(int i = 0; i < NEQ; i++) {
              _k_aux__[i] = _k2__[i];
              _k2__[i] = _k1__[i];
              _k1__[i] = _k_aux__[i];
          }

          // it steps the method ahead, with euler solution
          for(int i = 0; i < NEQ; i++) {
              sv_local[i] = edos_new_euler_[i];
          }

          if(time_new + previous_dt >= final_time) {
              if((fabs(final_time - time_new) < 1.0e-5)) {
                  break;
              } else if(time_new < final_time) {
                  dt = previous_dt = final_time - time_new;
                  time_new += previous_dt;
                  break;
              } else {
                  dt = previous_dt = min_dt;
                  time_new += (final_time - time_new);
                  printf("Error: %d: %lf\n", thread_id, final_time - time_new);
                  break;
              }
          } else {
              time_new += previous_dt;
          }
      }
  }

  //#pragma unroll
  for(int i = 0; i < NEQ; i++) {
      *((real *)((char *)sv + pitch * i) + thread_id) = sv_local[i];
  }

  DT = dt;
  TIME_NEW = time_new;
  PREVIOUS_DT = previous_dt;
}
//     #define DT *((real *)((char *)sv + pitch * (NEQ)) + thread_id)
//     #define TIME_NEW *((real *)((char *)sv + pitch * (NEQ+1)) + thread_id)
//     #define PREVIOUS_DT *((real *)((char *)sv + pitch * (NEQ+2)) + thread_id)
//
//     real rDY[NEQ];
//
//     real _tolerances_[NEQ];
//     real _aux_tol = 0.0;
//     real dt = DT;
//     real time_new = TIME_NEW;
//     real previous_dt = PREVIOUS_DT;
//
//     real edos_old_aux_[NEQ];
//     real edos_new_euler_[NEQ];
//     real _k1__[NEQ];
//     real _k2__[NEQ];
//     real _k_aux__[NEQ];
//     real sv_local[NEQ];
//
//     const real _beta_safety_ = 0.8;
//
//     const real __tiny_ = pow(abstol, 2.0f);
//
//     // dt = ((time_new + dt) > final_time) ? (final_time - time_new) : dt;
//     if(time_new + dt > final_time) {
//         dt = final_time - time_new;
//     }
//
//     //#pragma unroll
//     for(int i = 0; i < NEQ; i++) {
//         sv_local[i] = *((real *)((char *)sv + pitch * i) + thread_id);
//     }
//
//     RHS_gpu(sv_local, rDY, stim_curr, thread_id, dt, layer, infarct_zone, infarct_stage, apicobasal);
//     time_new += dt;
//
//     //#pragma unroll
//     for(int i = 0; i < NEQ; i++) {
//         _k1__[i] = rDY[i];
//     }
//
//     int count = 0;
//     int count_limit = (final_time - time_new) / min_dt;
//     int aux_count_limit = count_limit + 2000000;
//
//     if(aux_count_limit > 0) {
//         count_limit = aux_count_limit;
//     }
//
//     while(1) {
//         //Without RushLarsen
//         //for(int i = 0; i < NEQ; i++) {
//             // stores the old variables in a vector
//         //    edos_old_aux_[i] = sv_local[i];
//             // //computes euler method
//         //    edos_new_euler_[i] = _k1__[i] * dt + edos_old_aux_[i];
//             // steps ahead to compute the rk2 method
//         //    sv_local[i] = edos_new_euler_[i];
//         //}
//         //With RushLarsen
//         for(int i = 0; i < 9; i++) {
//             edos_old_aux_[i] = sv_local[i];
//             edos_new_euler_[i] = _k1__[i] * dt + edos_old_aux_[i];
//             sv_local[i] = edos_new_euler_[i];
//         }
//         for(int i = 9; i < 29; i++) {
//             edos_old_aux_[i] = sv_local[i];
//             edos_new_euler_[i] = _k1__[i];
//             sv_local[i] = edos_new_euler_[i];
//         }
//         for(int i = 29; i < 31; i++) {
//             edos_old_aux_[i] = sv_local[i];
//             edos_new_euler_[i] = _k1__[i] * dt + edos_old_aux_[i];
//             sv_local[i] = edos_new_euler_[i];
//         }
//         for(int i = 31; i < 36; i++) {
//             edos_old_aux_[i] = sv_local[i];
//             edos_new_euler_[i] = _k1__[i];
//             sv_local[i] = edos_new_euler_[i];
//         }
//         for(int i = 36; i < 42; i++) {
//             edos_old_aux_[i] = sv_local[i];
//             edos_new_euler_[i] = _k1__[i] * dt + edos_old_aux_[i];
//             sv_local[i] = edos_new_euler_[i];
//         }
//         for(int i = 42; i < 43; i++) {
//             edos_old_aux_[i] = sv_local[i];
//             edos_new_euler_[i] = _k1__[i];
//             sv_local[i] = edos_new_euler_[i];
//         }
//         for(int i = 43; i < NEQ; i++) {
//             edos_old_aux_[i] = sv_local[i];
//             edos_new_euler_[i] = _k1__[i] * dt + edos_old_aux_[i];
//             sv_local[i] = edos_new_euler_[i];
//         }
//
//         time_new += dt;
//         RHS_gpu(sv_local, rDY, stim_curr, thread_id, dt, layer, infarct_zone, infarct_stage, apicobasal);
//         time_new -= dt; // step back
//
//         real greatestError = 0.0, auxError = 0.0;
//         //#pragma unroll
//         //for(int i = 0; i < NEQ; i++) {
//         //    // stores the new evaluation
//         //    _k2__[i] = rDY[i];
//         //    _aux_tol = fabs(edos_new_euler_[i]) * reltol;
//         //    _tolerances_[i] = (abstol > _aux_tol) ? abstol : _aux_tol;
//         //    // finds the greatest error between  the steps
//         //    auxError = fabs(((dt / 2.0) * (_k1__[i] - _k2__[i])) / _tolerances_[i]);
//         //    greatestError = (auxError > greatestError) ? auxError : greatestError;
//         //}
//         // for(int i = 0; i < 9; i++) {
//         //     _k2__[i] = rDY[i];
//         //     _aux_tol = fabs(edos_new_euler_[i]) * reltol;
//         //     _tolerances_[i] = (abstol > _aux_tol) ? abstol : _aux_tol;
//         //     auxError = fabs(((dt / 2.0) * (_k1__[i] - _k2__[i])) / _tolerances_[i]);
//         //     greatestError = (auxError > greatestError) ? auxError : greatestError;
//         // }
//         // for(int i = 9; i < 29; i++) {
//         //     _k2__[i] = rDY[i];
//         // }
//         // for(int i = 29; i < 31; i++) {
//         //     _k2__[i] = rDY[i];
//         //     _aux_tol = fabs(edos_new_euler_[i]) * reltol;
//         //     _tolerances_[i] = (abstol > _aux_tol) ? abstol : _aux_tol;
//         //     auxError = fabs(((dt / 2.0) * (_k1__[i] - _k2__[i])) / _tolerances_[i]);
//         //     greatestError = (auxError > greatestError) ? auxError : greatestError;
//         // }
//         // for(int i = 31; i < 36; i++) {
//         //     _k2__[i] = rDY[i];
//         // }
//         // for(int i = 36; i < 42; i++) {
//         //     _k2__[i] = rDY[i];
//         //     _aux_tol = fabs(edos_new_euler_[i]) * reltol;
//         //     _tolerances_[i] = (abstol > _aux_tol) ? abstol : _aux_tol;
//         //     auxError = fabs(((dt / 2.0) * (_k1__[i] - _k2__[i])) / _tolerances_[i]);
//         //     greatestError = (auxError > greatestError) ? auxError : greatestError;
//         // }
//         // for(int i = 42; i < 43; i++) {
//         //     _k2__[i] = rDY[i];
//         // }
//         for(int i = 0; i < NEQ; i++) {
//             _k2__[i] = rDY[i];
//             _aux_tol = fabs(edos_new_euler_[i]) * reltol;
//             _tolerances_[i] = (abstol > _aux_tol) ? abstol : _aux_tol;
//             auxError = fabs(((dt / 2.0) * (_k1__[i] - _k2__[i])) / _tolerances_[i]);
//             greatestError = (auxError > greatestError) ? auxError : greatestError;
//         }
//
//         /// adapt the time step
//         greatestError += __tiny_;
//         previous_dt = dt;
//         /// adapt the time step
//         dt = _beta_safety_ * dt * sqrt(1.0f / greatestError);
//
//         if(time_new + dt > final_time) {
//             dt = final_time - time_new;
//         }
//
//         // it doesn't accept the solution
//         if(count < count_limit && (greatestError >= 1.0f)) {
//             // restore the old values to do it again
//             for(int i = 0; i < NEQ; i++) {
//                 sv_local[i] = edos_old_aux_[i];
//             }
//             count++;
//             // throw the results away and compute again
//         } else {
//             count = 0;
//
//             // if(greatestError >=1.0) {
//             //    printf("Thread //d,accepting solution with error > //lf \n", thread_id, greatestError);
//             //}
//
//             // it accepts the solutions
//             // int aux = (dt > max_step && max_step != 0);
//             // dt = (aux) ? max_step : dt;
//
//             if(dt < min_dt) {
//                 dt = min_dt;
//             } else if(dt > max_dt && max_dt != 0) {
//                 dt = max_dt;
//             }
//
//             if(time_new + dt > final_time) {
//                 dt = final_time - time_new;
//             }
//
//             // change vectors k1 e k2 , para que k2 seja aproveitado como k1 na proxima iteração
//             //#pragma unroll
//             for(int i = 0; i < NEQ; i++) {
//                 _k_aux__[i] = _k2__[i];
//                 _k2__[i] = _k1__[i];
//                 _k1__[i] = _k_aux__[i];
//             }
//
//             // it steps the method ahead, with euler solution
//             //#pragma unroll
//             for(int i = 0; i < NEQ; i++) {
//                 sv_local[i] = edos_new_euler_[i];
//             }
//
//             // verifica se o incremento para a próxima iteração ultrapassa o tempo de salvar, q neste caso é o tempo
//             // final
//             if(time_new + previous_dt >= final_time) {
//                 // se são iguais, ja foi calculada a iteração no ultimo passo de tempo e deve-se para o laço
//                 // nao usar igualdade - usar esta conta, pode-se mudar a tolerância
//                 // printf("//d: //lf\n", threadID, fabs(final_time - time_new));
//                 if((fabs(final_time - time_new) < 1.0e-5)) {
//                     break;
//                 } else if(time_new < final_time) {
//                     dt = previous_dt = final_time - time_new;
//                     time_new += previous_dt;
//                     break;
//                 } else {
//                     dt = previous_dt = min_dt;
//                     time_new += (final_time - time_new);
//                     printf("Nao era pra chegar aqui: %d: %lf\n", thread_id, final_time - time_new);
//                     break;
//                 }
//             } else {
//                 time_new += previous_dt;
//             }
//         }
//     }
//
//     //#pragma unroll
//     for(int i = 0; i < NEQ; i++) {
//         *((real *)((char *)sv + pitch * i) + thread_id) = sv_local[i];
//     }
//
//     DT = dt;
//     TIME_NEW = time_new;
//     PREVIOUS_DT = previous_dt;
// }

inline __device__ void RHS_gpu(real *sv, real *rDY, real stim_current, int thread_id, real dt, int layer, int infarct_zone, int infarct_stage, real apicobasal) {
    // State variables
    real v;
    real nai;
    real nass;
    real ki;
    real kss;
    real cai;
    real cass;
    real cansr;
    real cajsr;
    real m;
    real hp;
    real h;
    real j;
    real jp;
    real mL;
    real hL;
    real hLp;
    real a;
    real iF;
    real iS;
    real ap;
    real iFp;
    real iSp;
    real d;
    real ff;
    real fs;
    real fcaf;
    real fcas;
    real jca;
    real nca;
    real nca_i;
    real ffp;
    real fcafp;
    real xs1;
    real xs2;
    real Jrel_np;
    real CaMKt;
    real ikr_c0;
    real ikr_c1;
    real ikr_c2;
    real ikr_o;
    real ikr_i;
    real Jrel_p;
    real cli;
    real clss;

    if (use_adpt) {
        v = sv[0];
        nai = sv[1];
        nass = sv[2];
        ki = sv[3];
        kss = sv[4];
        cai = sv[5];
        cass = sv[6];
        cansr = sv[7];
        cajsr = sv[8];
        m = sv[9];
        hp = sv[10];
        h = sv[11];
        j = sv[12];
        jp = sv[13];
        mL = sv[14];
        hL = sv[15];
        hLp = sv[16];
        a = sv[17];
        iF = sv[18];
        iS = sv[19];
        ap = sv[20];
        iFp = sv[21];
        iSp = sv[22];
        d = sv[23];
        ff = sv[24];
        fs = sv[25];
        fcaf = sv[26];
        fcas = sv[27];
        jca = sv[28];
        nca = sv[29];
        nca_i = sv[30];
        ffp = sv[31];
        fcafp = sv[32];
        xs1 = sv[33];
        xs2 = sv[34];
        Jrel_np = sv[35];
        CaMKt = sv[36];
        ikr_c0 = sv[37];
        ikr_c1 = sv[38];
        ikr_c2 = sv[39];
        ikr_o = sv[40];
        ikr_i = sv[41];
        Jrel_p = sv[42];
        cli = sv[43];
        clss = sv[44];
    } else {
        v = *((real *)((char *)sv + pitch * 0)+thread_id);
        nai = *((real *)((char *)sv + pitch * 1)+thread_id);
        nass = *((real *)((char *)sv + pitch * 2)+thread_id);
        ki = *((real *)((char *)sv + pitch * 3)+thread_id);
        kss = *((real *)((char *)sv + pitch * 4)+thread_id);
        cai = *((real *)((char *)sv + pitch * 5)+thread_id);
        cass = *((real *)((char *)sv + pitch * 6)+thread_id);
        cansr = *((real *)((char *)sv + pitch * 7)+thread_id);
        cajsr = *((real *)((char *)sv + pitch * 8)+thread_id);
        m = *((real *)((char *)sv + pitch * 9)+thread_id);
        hp = *((real *)((char *)sv + pitch * 10)+thread_id);
        h = *((real *)((char *)sv + pitch * 11)+thread_id);
        j = *((real *)((char *)sv + pitch * 12)+thread_id);
        jp = *((real *)((char *)sv + pitch * 13)+thread_id);
        mL = *((real *)((char *)sv + pitch * 14)+thread_id);
        hL = *((real *)((char *)sv + pitch * 15)+thread_id);
        hLp = *((real *)((char *)sv + pitch * 16)+thread_id);
        a = *((real *)((char *)sv + pitch * 17)+thread_id);
        iF = *((real *)((char *)sv + pitch * 18)+thread_id);
        iS = *((real *)((char *)sv + pitch * 19)+thread_id);
        ap = *((real *)((char *)sv + pitch * 20)+thread_id);
        iFp = *((real *)((char *)sv + pitch * 21)+thread_id);
        iSp = *((real *)((char *)sv + pitch * 22)+thread_id);
        d = *((real *)((char *)sv + pitch * 23)+thread_id);
        ff = *((real *)((char *)sv + pitch * 24)+thread_id);
        fs = *((real *)((char *)sv + pitch * 25)+thread_id);
        fcaf = *((real *)((char *)sv + pitch * 26)+thread_id);
        fcas = *((real *)((char *)sv + pitch * 27)+thread_id);
        jca = *((real *)((char *)sv + pitch * 28)+thread_id);
        nca = *((real *)((char *)sv + pitch * 29)+thread_id);
        nca_i = *((real *)((char *)sv + pitch * 30)+thread_id);
        ffp = *((real *)((char *)sv + pitch * 31)+thread_id);
        fcafp = *((real *)((char *)sv + pitch * 32)+thread_id);
        xs1 = *((real *)((char *)sv + pitch * 33)+thread_id);
        xs2 = *((real *)((char *)sv + pitch * 34)+thread_id);
        Jrel_np = *((real *)((char *)sv + pitch * 35)+thread_id);
        CaMKt = *((real *)((char *)sv + pitch * 36)+thread_id);
        ikr_c0 = *((real *)((char *)sv + pitch * 37)+thread_id);
        ikr_c1 = *((real *)((char *)sv + pitch * 38)+thread_id);
        ikr_c2 = *((real *)((char *)sv + pitch * 39)+thread_id);
        ikr_o = *((real *)((char *)sv + pitch * 40)+thread_id);
        ikr_i = *((real *)((char *)sv + pitch * 41)+thread_id);
        Jrel_p = *((real *)((char *)sv + pitch * 42)+thread_id);
        cli = *((real *)((char *)sv + pitch * 43)+thread_id);
        clss = *((real *)((char *)sv + pitch * 44)+thread_id);
    }

    #include "ToRORd_dynCl_PhiCaL_IKCa_mixed_apicobasal_infarctionRemod_RZ1.common.c"
}
