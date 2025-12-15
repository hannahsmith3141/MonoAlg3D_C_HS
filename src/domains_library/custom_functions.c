#include "domain_helpers.h"

#include "custom_mesh_info_data.h"
#include "../3dparty/sds/sds.h"
#include "../3dparty/stb_ds.h"
#include "../config/domain_config.h"
#include "../config_helpers/config_helpers.h"
#include "../libraries_common/common_data_structures.h"
#include "../logger/logger.h"
#include "../utils/stop_watch.h"
#include "../utils/utils.h"
#include <assert.h>
#include <time.h>

#include <float.h>
#include <unistd.h>

void set_model_with_infarction_custom_data(struct cell_node *cell, real_cpu *custom_data) {
    LAYER(cell) = ((((int) custom_data[0]) - 1) % 3) + 1;
    if(custom_data[0] < 4) {
        INFARCT_ZONE(cell) = 0;
    }
    else if(custom_data[0] < 7) {
        INFARCT_ZONE(cell) = 1;
    }
    else if(custom_data[0] < 10) {
        INFARCT_ZONE(cell) = 2;
    }
    APICOBASAL(cell) = (real) custom_data[1];
    FAST_ENDO(cell) = (int) custom_data[2];
}

void set_model_with_infarction_custom_data_linearBZ(struct cell_node *cell, real_cpu *custom_data) {
    LAYER(cell) = ((int) (floor((double) custom_data[0]) - 1) % 3) + 1;
    if(custom_data[0] < 4) {
        INFARCT_ZONE(cell) = 0;
    }
    else if(custom_data[0] < 7) {
        INFARCT_ZONE(cell) = 1;
    }
    else if(custom_data[0] < 10) {
        INFARCT_ZONE(cell) = 2 + custom_data[0]-(long)custom_data[0];
    }
    APICOBASAL(cell) = (real) custom_data[1];
    FAST_ENDO(cell) = (int) custom_data[2];
}


uint32_t set_model_with_infarction(struct grid *the_grid, const char *mesh_file, uint32_t num_volumes, double start_h, uint8_t num_custom_data,
                                   set_custom_data_for_mesh_fn set_custom_data_for_mesh) {
    struct stop_watch sw = {0};
    start_stop_watch(&sw);

    FILE *file = fopen(mesh_file, "r");
    if(!file) {
        log_error_and_exit("Error opening mesh described in %s!!\n", mesh_file);
    }

    struct custom_mesh_basic_data_hash_entry *custom_mesh_data_hash = NULL;
    hmdefault(custom_mesh_data_hash, -1);

    real_cpu maxx = 0.0;
    real_cpu maxy = 0.0;
    real_cpu maxz = 0.0;
    real_cpu minx = DBL_MAX;
    real_cpu miny = DBL_MAX;
    real_cpu minz = DBL_MAX;

    bool load_custom_data = (set_custom_data_for_mesh != NULL && num_custom_data > 0);

    real_cpu **custom_data = NULL;

    if(load_custom_data) {
        custom_data = MALLOC_ARRAY_OF_TYPE(real_cpu *, num_volumes);
        if(custom_data == NULL) {
            log_error_and_exit("Failed to allocate memory\n");
        }
        for(uint32_t i = 0; i < num_volumes; i++) {
            custom_data[i] = MALLOC_ARRAY_OF_TYPE(real_cpu, num_custom_data);

            if(custom_data[i] == NULL) {
                log_error_and_exit("Failed to allocate memory\n");
            }
        }
    }

    char *line = NULL;
    size_t len;

    log_info("Start - reading mesh file\n");

    for(uint32_t i = 0; i < num_volumes; i++) {

        sds *data;
        int split_count;

        getline(&line, &len, file);

        char *tmp = line;
        data = sdssplit(tmp, ",", &split_count);

        if(split_count < 3) {
            log_error_and_exit("Not enough data to load the mesh geometry in line %d of file %s! [available=%d, required=3]\n", i+1, mesh_file, split_count);
        }

        real_cpu cx = strtod(data[0], NULL);
        real_cpu cy = strtod(data[1], NULL);
        real_cpu cz = strtod(data[2], NULL);

        if(load_custom_data) {
            // indexes 3, 4 and 5 are not used in this function
            for(int d = 0; d < num_custom_data; d++) {
                custom_data[i][d] = strtod(data[d + 6], NULL);
            }
        }

        hmput(custom_mesh_data_hash, POINT3D(cx,cy,cz), i);
        if(cx > maxx) {maxx = cx;}
        if(cx < minx) {minx = cx;}
        if(cy > maxy) {maxy = cy;}
        if(cy < miny) {miny = cy;}
        if(cz > maxz) {maxz = cz;}
        if(cz < minz) {minz = cz;}
        sdsfreesplitres(data, split_count);
    }

    log_info("Finish - reading mesh file\n");

    double cube_side = start_h;
    double min_cube_side = fmax(maxx, fmax(maxy, maxz)) + start_h;

    while(cube_side < min_cube_side) {
        cube_side = cube_side*2;
    }

    double tmp_size = cube_side / 2.0;
    uint16_t num_ref = 0;

    while(tmp_size > start_h) {
        tmp_size = tmp_size / 2;
        num_ref++;
    }

    initialize_and_construct_grid(the_grid, SAME_POINT3D(cube_side));

    uint32_t num_loaded = 0;

    struct point_3d min_bound = POINT3D(minx-start_h, miny-start_h, minz-start_h);
    struct point_3d max_bound = POINT3D(maxx+start_h, maxy+start_h, maxz+start_h);

    log_info("\nStart - refining the initial cube (refining %d times with a cube side of %lf)\n", num_ref, cube_side);
    refine_grid_with_bounds(the_grid, num_ref, min_bound, max_bound);
    log_info("Finish - refining the initial cube\n\n");

    log_info("Loading grid with cube side of %lf\n", cube_side);

    FOR_EACH_CELL(the_grid) {
        real_cpu x = cell->center.x;
        real_cpu y = cell->center.y;
        real_cpu z = cell->center.z;

        if(x > maxx || y > maxy || z > maxz || x < minx || y < miny || z < minz) {
            cell->active = false;
        } else {
            struct point_3d p = POINT3D(x,y,z);
            int index = hmget(custom_mesh_data_hash, p);

            if(index != -1) {
                cell->active = true;
                cell->original_position_in_file = index;

                INITIALIZE_INFARCTION_INFO(cell);
                if(load_custom_data) {
                    set_custom_data_for_mesh(cell, custom_data[cell->original_position_in_file]);
                }

                num_loaded++;
            } else {
                cell->active = false;
            }
        }
    }

    if(num_loaded > 0) {
        the_grid->mesh_side_length.x = maxx + start_h;
        the_grid->mesh_side_length.y = maxy + start_h;
        the_grid->mesh_side_length.z = maxz + start_h;

        log_info("Cleaning grid\n");

        for(uint16_t r = 0; r < num_ref; r++) {
            derefine_grid_inactive_cells(the_grid);
        }
    }

    free(line);
    hmfree(custom_mesh_data_hash);

    if(custom_data) {
        for(int i = 0; i < num_volumes; i++) {
            free(custom_data[i]);
        }
        free(custom_data);
    }

    fclose(file);
    log_info("\nTime to load the mesh: %ld μs\n\n", stop_stop_watch(&sw));
    return num_loaded;
}

SET_SPATIAL_DOMAIN(initialize_model_with_infarction) {
    char *mesh_file = NULL;
    GET_PARAMETER_STRING_VALUE_OR_REPORT_ERROR(mesh_file, config, "mesh_file");
    uint32_t total_number_mesh_points = 0;
    GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(uint32_t, total_number_mesh_points, config, "number_of_points");
    real_cpu start_h = 0.0;
    GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(real_cpu, start_h, config, "start_discretization");
    real_cpu max_h = start_h;
    GET_PARAMETER_NUMERIC_VALUE_OR_USE_DEFAULT(real_cpu, max_h, config, "maximum_discretization");

    the_grid->start_discretization = SAME_POINT3D(start_h);
    the_grid->max_discretization = SAME_POINT3D(max_h);

    return set_model_with_infarction(the_grid, mesh_file, total_number_mesh_points, start_h, 3, &set_model_with_infarction_custom_data);
}

SET_SPATIAL_DOMAIN(initialize_model_with_infarction_linearBZ) {
    char *mesh_file = NULL;
    GET_PARAMETER_STRING_VALUE_OR_REPORT_ERROR(mesh_file, config, "mesh_file");
    uint32_t total_number_mesh_points = 0;
    GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(uint32_t, total_number_mesh_points, config, "number_of_points");
    real_cpu start_h = 0.0;
    GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(real_cpu, start_h, config, "start_discretization");
    real_cpu max_h = start_h;
    GET_PARAMETER_NUMERIC_VALUE_OR_USE_DEFAULT(real_cpu, max_h, config, "maximum_discretization");

    the_grid->start_discretization = SAME_POINT3D(start_h);
    the_grid->max_discretization = SAME_POINT3D(max_h);

    return set_model_with_infarction(the_grid, mesh_file, total_number_mesh_points, start_h, 3, &set_model_with_infarction_custom_data_linearBZ);
}
