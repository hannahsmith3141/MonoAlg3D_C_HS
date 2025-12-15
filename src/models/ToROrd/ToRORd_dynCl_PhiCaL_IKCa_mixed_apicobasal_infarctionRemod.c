#include "ToRORd_dynCl_PhiCaL_IKCa_mixed_apicobasal_infarctionRemod.h"
#include <stdlib.h>
#include <stdio.h>

real max_step;
real min_step;
real abstol;
real reltol;
bool adpt;
real *ode_dt, *ode_previous_dt, *ode_time_new;

GET_CELL_MODEL_DATA(init_cell_model_data) {
    if(get_initial_v)
        cell_model->initial_v = INITIAL_V;
    if(get_neq)
        cell_model->number_of_ode_equations = NEQ;
}

SET_ODE_INITIAL_CONDITIONS_CPU(set_model_initial_conditions_cpu) {
    log_info("Using ToRORd_dynCl_PhiCaL_IKCa_mixed_apicobasal_infarctionRemod CPU model\n");

    // Get the extra_data array
    real *extra_data = NULL;
    if(solver->ode_extra_data) {
        extra_data = (real*)solver->ode_extra_data;
    } else {
        log_error_and_exit("You need to specify a mask function when using a mixed model!\n");
    }

    uint32_t num_cells = solver->original_num_cells;
    solver->sv = (real*)malloc(NEQ*num_cells*sizeof(real));

    max_step = solver->max_dt;
    min_step = solver->min_dt;
    abstol   = solver->abs_tol;
    reltol   = solver->rel_tol;
    adpt     = solver->adaptive;

    if(adpt) {
        ode_dt = (real*)malloc(num_cells*sizeof(real));

        OMP(parallel for)
        for(int i = 0; i < num_cells; i++)
            ode_dt[i] = solver->min_dt;

        ode_previous_dt = (real*)calloc(num_cells, sizeof(real));
        ode_time_new    = (real*)calloc(num_cells, sizeof(real));
        log_info("Using Adaptive Euler model to solve the ODEs\n");
    } else {
        log_info("Using Euler model to solve the ODEs\n");
    }

    OMP(parallel for)
    for(uint32_t i = 0; i < num_cells; i++) {
        int layer = (int) extra_data[i];
        int infarct_zone = (int) extra_data[i + num_cells];
        int infarct_stage = (int) extra_data[3 * num_cells];
        real *sv = &solver->sv[i * NEQ];

        // if(i == 50) {
        // log_info("layer %d\n", layer);
        // log_info("zone %d\n", infarct_zone);
        // log_info("stage %d\n", infarct_stage);
        // }

        #include "ToRORd_dynCl_PhiCaL_IKCa_mixed_apicobasal_infarctionRemod_SS_constNai.common.c"
        sv[ 0]= v;
        sv[ 1]= nai;
        sv[ 2]= nass;
        sv[ 3]= ki;
        sv[ 4]= kss;
        sv[ 5]= cai;
        sv[ 6]= cass;
        sv[ 7]= cansr;
        sv[ 8]= cajsr;
        sv[ 9]= m;
        sv[10]= hp;
        sv[11]= h;
        sv[12]= j;
        sv[13]= jp;
        sv[14]= mL;
        sv[15]= hL;
        sv[16]= hLp;
        sv[17]= a;
        sv[18]= iF;
        sv[19]= iS;
        sv[20]= ap;
        sv[21]= iFp;
        sv[22]= iSp;
        sv[23]= d;
        sv[24]= ff;
        sv[25]= fs;
        sv[26]= fcaf;
        sv[27]= fcas;
        sv[28]= jca;
        sv[29]= nca;
        sv[30]= nca_i;
        sv[31]= ffp;
        sv[32]= fcafp;
        sv[33]= xs1;
        sv[34]= xs2;
        sv[35]= Jrel_np;
        sv[36]= CaMKt;
        sv[37]= ikr_c0;
        sv[38]= ikr_c1;
        sv[39]= ikr_c2;
        sv[40]= ikr_o;
        sv[41]= ikr_i;
        sv[42]= Jrel_p;
        sv[43]= cli;
        sv[44]= clss;
    }
}

SOLVE_MODEL_ODES(solve_model_odes_cpu) {
    //Get the extra_data array
    real *extra_data = NULL;
    if(ode_solver->ode_extra_data) {
        extra_data = (real*)ode_solver->ode_extra_data;
    } else {
        log_error_and_exit("You need to specify a mask function when using this mixed model!\n");
    }
    uint32_t sv_id;
    size_t num_cells_to_solve = ode_solver->num_cells_to_solve;
    uint32_t * cells_to_solve = ode_solver->cells_to_solve;
    real *sv = ode_solver->sv;
    real dt = ode_solver->min_dt;
    uint32_t num_steps = ode_solver->num_steps;
    int infarct_stage = (int) extra_data[3 * num_cells_to_solve];

    OMP(parallel for private(sv_id))
    for (int i = 0; i < num_cells_to_solve; i++) {
        int layer = (int) extra_data[i];
        int infarct_zone = (int) extra_data[i + num_cells_to_solve];
        real apicobasal = extra_data[i + (2*num_cells_to_solve)];

        // if(i == 50) {
        // log_info("layer %d\n", layer);
        // log_info("zone %d\n", infarct_zone);
        // log_info("stage %d\n", infarct_stage);
        // }

        if(cells_to_solve)
            sv_id = cells_to_solve[i];
        else
            sv_id = i;

        if(adpt) {
            solve_forward_euler_cpu_adpt(sv + (sv_id * NEQ), stim_currents[i], current_t + dt, sv_id, layer, infarct_zone, infarct_stage, apicobasal);
        } else {
            for (int j = 0; j < num_steps; ++j) {
                solve_model_ode_cpu(dt, sv + (sv_id * NEQ), stim_currents[i], layer, infarct_zone, infarct_stage, apicobasal);
            }
        }
    }
}

void solve_model_ode_cpu(real dt, real *sv, real stim_current, int layer, int infarct_zone, int infarct_stage, real apicobasal)  {
    real rY[NEQ], rDY[NEQ];
    for(int i = 0; i < NEQ; i++)
        rY[i] = sv[i];
    RHS_cpu(rY, rDY, stim_current, dt, layer, infarct_zone, infarct_stage, apicobasal);

    // Full Explicit Euler
    //for(int i = 0; i < NEQ; i++)
    //    sv[i] = dt*rDY[i] + rY[i];

    //Explicit Euler + RushLarsen
    //for(int i = 0; i < 9; i++)
    //    sv[i] = dt*rDY[i] + rY[i];
    //for(int i = 9; i < 29; i++)
    //    sv[i] = rDY[i];
    //for(int i = 29; i < 31; i++)
    //    sv[i] = dt*rDY[i] + rY[i];
    //for(int i = 31; i < 36; i++)
    //    sv[i] = rDY[i];
    //for(int i = 36; i < 42; i++)
    //    sv[i] = dt*rDY[i] + rY[i];
    //for(int i = 42; i < 43; i++)
    //    sv[i] = rDY[i];
    for(int i = 0; i < NEQ; i++)
        sv[i] = rDY[i];
}

void solve_forward_euler_cpu_adpt(real *sv, real stim_curr, real final_time, int sv_id, int layer, int infarct_zone, int infarct_stage, real apicobasal) {
  real rDY[NEQ];

  real _tolerances_[NEQ];
  real _aux_tol = 0.0;
  //initializes the variables
  real dt = ode_dt[sv_id];
  real time_new = ode_time_new[sv_id];
  real previous_dt = ode_previous_dt[sv_id];

  real edos_old_aux_[NEQ];
  real edos_new_euler_[NEQ];
  real *_k1__ = (real*) malloc(sizeof(real)*NEQ);
  real *_k2__ = (real*) malloc(sizeof(real)*NEQ);
  real *_k_aux__;

  const real _beta_safety_ = 0.8;
  const real __tiny_ = pow(abstol, 2.0f);

  if(time_new + dt > final_time) {
     dt = final_time - time_new;
  }

  RHS_cpu(sv, rDY, stim_curr, dt, layer, infarct_zone, infarct_stage, apicobasal);
  time_new += dt;

  for(int i = 0; i < NEQ; i++){
      _k1__[i] = rDY[i];
  }

  int count = 0;
  int count_limit = (final_time - time_new)/min_step;
  int aux_count_limit = count_limit+2000000;

  if(aux_count_limit > 0) {
      count_limit = aux_count_limit;
  }

  while(1) {

      for(int i = 0; i < NEQ; i++) {
          //stores the old variables in a vector
          edos_old_aux_[i] = sv[i];
          //computes euler method
          edos_new_euler_[i] = _k1__[i];
          //steps ahead to compute the rk2 method
          sv[i] = edos_new_euler_[i];
      }

      time_new += dt;
      RHS_cpu(sv, rDY, stim_curr, dt, layer, infarct_zone, infarct_stage, apicobasal);
      time_new -= dt;//step back

      double greatestError = 0.0, auxError = 0.0;
      for(int i = 0; i < NEQ; i++) {
          // stores the new evaluation
          _k2__[i] = rDY[i];
          _aux_tol = fabs(edos_new_euler_[i]) * reltol;
          _tolerances_[i] = (abstol > _aux_tol) ? abstol : _aux_tol;

          // finds the greatest error between  the steps
          auxError = fabs(((dt / 2.0) * (_k1__[i] - _k2__[i])) / _tolerances_[i]);

          greatestError = (auxError > greatestError) ? auxError : greatestError;
      }
      ///adapt the time step
      greatestError += __tiny_;
      previous_dt = dt;
      ///adapt the time step
      dt = _beta_safety_ * dt * sqrt(1.0f/greatestError);

      if (time_new + dt > final_time) {
          dt = final_time - time_new;
      }

      //it doesn't accept the solution
      if ( count < count_limit  && (greatestError >= 1.0f)) {
          //restore the old values to do it again
          for(int i = 0;  i < NEQ; i++) {
              sv[i] = edos_old_aux_[i];
          }
          count++;
          //throw the results away and compute again
      } else{//it accepts the solutions
          count = 0;

          if (dt < min_step) {
              dt = min_step;
          } else if (dt > max_step && max_step != 0) {
              dt = max_step;
          }
          if (time_new + dt > final_time) {
              dt = final_time - time_new;
          }

          _k_aux__ = _k2__;
          _k2__   = _k1__;
          _k1__   = _k_aux__;

          //it steps the method ahead, with euler solution
          for(int i = 0; i < NEQ; i++){
              sv[i] = edos_new_euler_[i];
          }

          if(time_new + previous_dt >= final_time) {
              if((fabs(final_time - time_new) < 1.0e-5)) {
                  break;
              } else if(time_new < final_time) {
                  dt = previous_dt = final_time - time_new;
                  time_new += previous_dt;
                  break;
              } else {
                  dt = previous_dt = min_step;
                  time_new += (final_time - time_new);
                  printf("Error: %lf\n", final_time - time_new);
                  break;
              }
          } else {
              time_new += previous_dt;
          }
      }
  }

  ode_dt[sv_id] = dt;
  ode_time_new[sv_id] = time_new;
  ode_previous_dt[sv_id] = previous_dt;

  free(_k1__);
  free(_k2__);
}
//     const real _beta_safety_ = 0.8;
//     int numEDO = NEQ;
//
//     real rDY[numEDO];
//
//     real _tolerances_[numEDO];
//     real _aux_tol = 0.0;
//     //initializes the variables
//     ode_previous_dt[sv_id] = ode_dt[sv_id];
//
//     real edos_old_aux_[numEDO];
//     real edos_new_euler_[numEDO];
//     real *_k1__ = (real*) malloc(sizeof(real)*numEDO);
//     real *_k2__ = (real*) malloc(sizeof(real)*numEDO);
//     real *_k_aux__;
//
//     real *dt = &ode_dt[sv_id];
//     real *time_new = &ode_time_new[sv_id];
//     real *previous_dt = &ode_previous_dt[sv_id];
//
//     if(*time_new + *dt > final_time) {
//        *dt = final_time - *time_new;
//     }
//
//     RHS_cpu(sv, rDY, stim_curr, *dt, layer, infarct_zone, infarct_stage, apicobasal);
//     *time_new += *dt;
//
//     //With and without RushLarsen
//     for(int i = 0; i < numEDO; i++){
//         _k1__[i] = rDY[i];
//     }
//
//     const double __tiny_ = pow(abstol, 2.0);
//     int count = 0;
//     int count_limit = (final_time - *time_new)/min_step;
//     int aux_count_limit = count_limit+2000000;
//
//     if(aux_count_limit > 0) {
//         count_limit = aux_count_limit;
//     }
//
//     while(1) {
//         //Without RushLarsen
//         //for(int i = 0; i < numEDO; i++) {
//             //stores the old variables in a vector
//         //    edos_old_aux_[i] = sv[i];
//             //computes euler method
//         //    edos_new_euler_[i] = _k1__[i] * *dt + edos_old_aux_[i];
//             //steps ahead to compute the rk2 method
//         //    sv[i] = edos_new_euler_[i];
//         //}
//         //With RushLarsen
//         for(int i = 0; i < 9; i++) {
//             edos_old_aux_[i] = sv[i];
//             edos_new_euler_[i] = _k1__[i] * *dt + edos_old_aux_[i];
//             sv[i] = edos_new_euler_[i];
//         }
//         for(int i = 9; i < 29; i++) {
//             edos_old_aux_[i] = sv[i];
//             edos_new_euler_[i] = _k1__[i];
//             sv[i] = edos_new_euler_[i];
//         }
//         for(int i = 29; i < 31; i++) {
//             edos_old_aux_[i] = sv[i];
//             edos_new_euler_[i] = _k1__[i] * *dt + edos_old_aux_[i];
//             sv[i] = edos_new_euler_[i];
//         }
//         for(int i = 31; i < 36; i++) {
//             edos_old_aux_[i] = sv[i];
//             edos_new_euler_[i] = _k1__[i];
//             sv[i] = edos_new_euler_[i];
//         }
//         for(int i = 36; i < 42; i++) {
//             edos_old_aux_[i] = sv[i];
//             edos_new_euler_[i] = _k1__[i] * *dt + edos_old_aux_[i];
//             sv[i] = edos_new_euler_[i];
//         }
//         for(int i = 42; i < 43; i++) {
//             edos_old_aux_[i] = sv[i];
//             edos_new_euler_[i] = _k1__[i];
//             sv[i] = edos_new_euler_[i];
//         }
//         for(int i = 43; i < numEDO; i++) {
//             edos_old_aux_[i] = sv[i];
//             edos_new_euler_[i] = _k1__[i] * *dt + edos_old_aux_[i];
//             sv[i] = edos_new_euler_[i];
//         }
//
//         *time_new += *dt;
//         RHS_cpu(sv, rDY, stim_curr, *dt, layer, infarct_zone, infarct_stage, apicobasal);
//         *time_new -= *dt;//step back
//
//         double greatestError = 0.0, auxError = 0.0;
//
//         for(int i = 0; i < 9; i++) {
//             _k2__[i] = rDY[i];
//             _aux_tol = fabs(edos_new_euler_[i]) * reltol;
//             _tolerances_[i] = (abstol > _aux_tol) ? abstol : _aux_tol;
//             auxError = fabs(((*dt / 2.0) * (_k1__[i] - _k2__[i])) / _tolerances_[i]);
//             greatestError = (auxError > greatestError) ? auxError : greatestError;
//         }
//         for(int i = 9; i < 29; i++) {
//             _k2__[i] = rDY[i];
//         }
//         for(int i = 29; i < 31; i++) {
//             _k2__[i] = rDY[i];
//             _aux_tol = fabs(edos_new_euler_[i]) * reltol;
//             _tolerances_[i] = (abstol > _aux_tol) ? abstol : _aux_tol;
//             auxError = fabs(((*dt / 2.0) * (_k1__[i] - _k2__[i])) / _tolerances_[i]);
//             greatestError = (auxError > greatestError) ? auxError : greatestError;
//         }
//         for(int i = 31; i < 36; i++) {
//             _k2__[i] = rDY[i];
//         }
//         for(int i = 36; i < 42; i++) {
//             _k2__[i] = rDY[i];
//             _aux_tol = fabs(edos_new_euler_[i]) * reltol;
//             _tolerances_[i] = (abstol > _aux_tol) ? abstol : _aux_tol;
//             auxError = fabs(((*dt / 2.0) * (_k1__[i] - _k2__[i])) / _tolerances_[i]);
//             greatestError = (auxError > greatestError) ? auxError : greatestError;
//         }
//         for(int i = 42; i < 43; i++) {
//             _k2__[i] = rDY[i];
//         }
//         for(int i = 43; i < numEDO; i++) {
//             _k2__[i] = rDY[i];
//             _aux_tol = fabs(edos_new_euler_[i]) * reltol;
//             _tolerances_[i] = (abstol > _aux_tol) ? abstol : _aux_tol;
//             auxError = fabs(((*dt / 2.0) * (_k1__[i] - _k2__[i])) / _tolerances_[i]);
//             greatestError = (auxError > greatestError) ? auxError : greatestError;
//         }
//         ///adapt the time step
//         greatestError += __tiny_;
//         *previous_dt = *dt;
//         ///adapt the time step
//         *dt = _beta_safety_ * (*dt) * sqrt(1.0f/greatestError);
//
//         if (*time_new + *dt > final_time) {
//             *dt = final_time - *time_new;
//         }
//
//         //it doesn't accept the solution
//         if ( count < count_limit  && (greatestError >= 1.0f)) {
//             //restore the old values to do it again
//             for(int i = 0;  i < numEDO; i++) {
//                 sv[i] = edos_old_aux_[i];
//             }
//
//             count++;
//             //throw the results away and compute again
//         } else {
//             //it accepts the solutions
//             if(greatestError >=1.0) {
//                 printf("Accepting solution with error > %lf \n", greatestError);
//             }
//
//             //printf("%e %e\n", _ode->time_new, edos_new_euler_[0]);
//             if (*dt < min_step) {
//                 *dt = min_step;
//             } else if (*dt > max_step && max_step != 0) {
//                 *dt = max_step;
//             }
//
//             if (*time_new + *dt > final_time) {
//                 *dt = final_time - *time_new;
//             }
//
//             _k_aux__ = _k2__;
//             _k2__   = _k1__;
//             _k1__   = _k_aux__;
//
//             //it steps the method ahead, with euler solution
//             for(int i = 0; i < numEDO; i++){
//                 sv[i] = edos_new_euler_[i];
//             }
//
//             if(*time_new + *previous_dt >= final_time){
//                 if((fabs(final_time - *time_new) < 1.0e-5) ){
//                     break;
//                 } else if(*time_new < final_time){
//                     *dt = *previous_dt = final_time - *time_new;
//                     *time_new += *previous_dt;
//                     break;
//                 } else{
//                     printf("Error: time_new %.20lf final_time %.20lf diff %e \n", *time_new , final_time, fabs(final_time - *time_new) );
//                     break;
//                 }
//             } else{
//                 *time_new += *previous_dt;
//             }
//         }
//     }
//
//     free(_k1__);
//     free(_k2__);
// }

void RHS_cpu(const real *sv, real *rDY, real stim_current, real dt, int layer, int infarct_zone, int infarct_stage, real apicobasal) {
    // State variables
    real v = sv[0];
    real nai = sv[1];
    real nass = sv[2];
    real ki = sv[3];
    real kss = sv[4];
    real cai = sv[5];
    real cass = sv[6];
    real cansr = sv[7];
    real cajsr = sv[8];
    real m = sv[9];
    real hp = sv[10];
    real h = sv[11];
    real j = sv[12];
    real jp = sv[13];
    real mL = sv[14];
    real hL = sv[15];
    real hLp = sv[16];
    real a = sv[17];
    real iF = sv[18];
    real iS = sv[19];
    real ap = sv[20];
    real iFp = sv[21];
    real iSp = sv[22];
    real d = sv[23];
    real ff = sv[24];
    real fs = sv[25];
    real fcaf = sv[26];
    real fcas = sv[27];
    real jca = sv[28];
    real nca = sv[29];
    real nca_i = sv[30];
    real ffp = sv[31];
    real fcafp = sv[32];
    real xs1 = sv[33];
    real xs2 = sv[34];
    real Jrel_np = sv[35];
    real CaMKt = sv[36];
    real ikr_c0 = sv[37];
    real ikr_c1 = sv[38];
    real ikr_c2 = sv[39];
    real ikr_o = sv[40];
    real ikr_i = sv[41];
    real Jrel_p = sv[42];
    real cli = sv[43];
    real clss = sv[44];

    #include "ToRORd_dynCl_PhiCaL_IKCa_mixed_apicobasal_infarctionRemod.common.c"
}
