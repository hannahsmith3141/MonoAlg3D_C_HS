############## CRN ##############################
MODEL_FILE_CPU="ToRORd_dynCl_PhiCaL_IKCa_mixed_apicobasal_infarctionRemod_RZ1.c"
MODEL_FILE_GPU="ToRORd_dynCl_PhiCaL_IKCa_mixed_apicobasal_infarctionRemod_RZ1.cu"
COMMON_HEADERS="ToRORd_dynCl_PhiCaL_IKCa_mixed_apicobasal_infarctionRemod_RZ1.h"

COMPILE_MODEL_LIB "ToRORd_dynCl_PhiCaL_IKCa_mixed_apicobasal_infarctionRemod_RZ1" "$MODEL_FILE_CPU" "$MODEL_FILE_GPU" "$COMMON_HEADERS"
##########################################################

