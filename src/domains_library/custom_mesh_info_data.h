#ifndef __CUSTOM_MESH_INFO_DATA_H
#define __CUSTOM_MESH_INFO_DATA_H

#include <stdbool.h>
#include "../common_types/common_types.h"

struct infarction_mesh_info {
	int layer;
  int infarct_zone;
  float apicobasal;
	int fast_endo;
};

#define INFARCTION_INFO(grid_cell) (struct infarction_mesh_info *)grid_cell->mesh_extra_info
#define LAYER(grid_cell) (INFARCTION_INFO(grid_cell))->layer
#define INFARCT_ZONE(grid_cell) (INFARCTION_INFO(grid_cell))->infarct_zone
#define APICOBASAL(grid_cell) (INFARCTION_INFO(grid_cell))->apicobasal
#define FAST_ENDO(grid_cell) (INFARCTION_INFO(grid_cell))->fast_endo

#define INITIALIZE_INFARCTION_INFO(grid_cell)                                                                            \
    do {                                                                                                           \
        size_t __size__ = sizeof (struct infarction_mesh_info);                                                    \
        (grid_cell)->mesh_extra_info = malloc (__size__);                                                          \
        (grid_cell)->mesh_extra_info_size = __size__;                                                                           \
        LAYER ((grid_cell)) = 4;                                                                             \
        INFARCT_ZONE ((grid_cell)) = 3;                                                                                   \
        APICOBASAL ((grid_cell)) = 1;                                                                                  \
        FAST_ENDO ((grid_cell)) = 2;                                                                             \
} while (0)

#endif /* __MESH_INFO_DATA_H */
