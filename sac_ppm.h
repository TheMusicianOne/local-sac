#ifndef SAC_PPM_H
#define SAC_PPM_H
#include "sacinterface.h"

#ifdef __cplusplus
extern "C" {
#endif

int write_ppm_iter(unsigned char *base, SACarg *labels, int iter);

#ifdef __cplusplus
}
#endif

#endif
