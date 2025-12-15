#include <unistd.h>

#include "../config/extra_data_config.h"
#include "../config_helpers/config_helpers.h"
#include "../libraries_common/common_data_structures.h"
#include "../utils/file_utils.h"
#include "../domains_library/custom_mesh_info_data.h"

//Sets extra data array for transmurality, apicobasal gradient, and
//infarct zone from the mesh file and infarct stage from the .ini file
SET_EXTRA_DATA(set_extra_data_with_infarction) {
    uint32_t num_active_cells = the_grid->num_active_cells;
    struct cell_node ** ac = the_grid->active_cells;

    *extra_data_size = sizeof(real)*((num_active_cells)*3 + 1);
    real *extra_data = (real*)malloc(*extra_data_size);

    uint32_t infarct_stage = 5;
    GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(uint32_t, infarct_stage, config, "infarct_stage");

    uint32_t i;
    OMP(parallel for)
    for (i = 0; i < num_active_cells; i++) {
        extra_data[i] = LAYER(ac[i]); //fast-endo 0, endo 1, mid 2, epi 3
        extra_data[i+num_active_cells] = INFARCT_ZONE(ac[i]); //healthy 0, infarct 1, border 2
        extra_data[i+(2*num_active_cells)] = APICOBASAL(ac[i]); //float number between 0 and 1
    }
    extra_data[(3*num_active_cells)] = infarct_stage; //healthy 0, ischemic 1, acute 2, acute-chronic 3, chronic 4

    return extra_data;
}

SET_EXTRA_DATA(set_extra_data_with_infarction_HectorVF) {
    uint32_t num_active_cells = the_grid->num_active_cells;
    struct cell_node ** ac = the_grid->active_cells;

    *extra_data_size = sizeof(real)*((num_active_cells)*3 + 1);
    real *extra_data = (real*)malloc(*extra_data_size);

    uint32_t infarct_stage = 5;
    GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(uint32_t, infarct_stage, config, "infarct_stage");

    uint32_t modelVF = 20;
    GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(uint32_t, modelVF, config, "modelVF");


    uint32_t i;
    OMP(parallel for)
    for (i = 0; i < num_active_cells; i++) {
        extra_data[i] = LAYER(ac[i]); //fast-endo 0, endo 1, mid 2, epi 3
        extra_data[i+num_active_cells] = INFARCT_ZONE(ac[i]); //healthy 0, infarct 1, border 2
        extra_data[i+(2*num_active_cells)] = APICOBASAL(ac[i]); //float number between 0 and 1
    }
    extra_data[(3*num_active_cells)] = infarct_stage; //healthy 0, ischemic 1, acute 2, acute-chronic 3, chronic 4
    extra_data[(3*num_active_cells) + 1] = modelVF; //12 VF susceptible, 18 control

    return extra_data;
}
