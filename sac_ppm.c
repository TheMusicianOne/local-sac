// sac_ppm.c  — works with: int write_ppm_iter(unsigned char *base, SACarg *labels, int iter)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "sacinterface.h"
#include "sac_ppm.h"

static int ensure_dir(const char *dir) {
  if (access(dir, F_OK) == 0) return 0;
  if (mkdir(dir, 0775) == 0) return 0;
  if (errno == EEXIST) return 0;
  return -1;
}

int write_ppm_iter(unsigned char *base_cstr, SACarg *sa_labels, int iter)
{
  // labels is int[.,.] wrapped as SACarg
  assert(SACARGgetDim(sa_labels) == 2);
  size_t w = SACARGgetShape(sa_labels, 0);  // x-dimension (width)
  size_t h = SACARGgetShape(sa_labels, 1);  // y-dimension (height)
  const int *lab = SACARGgetSharedData(SACTYPE__MAIN__int, sa_labels);

  const char *outdir = "./";
  if (ensure_dir(outdir) != 0) {
    fprintf(stderr, "[write_ppm_iter] mkdir %s failed: %s\n", outdir, strerror(errno));
    return 0;
  }

  int need = snprintf(NULL, 0, "%s/%s_%04d.ppm", outdir, (const char*)base_cstr, iter) + 1;
  char *path = (char*)malloc(need);
  if (!path) return 0;
  snprintf(path, need, "%s/%s_%04d.ppm", outdir, (const char*)base_cstr, iter);

  FILE *f = fopen(path, "wb");
  if (!f) {
    fprintf(stderr, "[write_ppm_iter] fopen(%s) failed: %s\n", path, strerror(errno));
    free(path);
    return 0;
  }

  // Binary PPM (P6) header
  fprintf(f, "P6\n%zu %zu\n255\n", w, h);

  // Correct linear index for SaC: rightmost index (y) is fastest → idx = x*h + y
  for (size_t y = 0; y < h; y++) {
    for (size_t x = 0; x < w; x++) {
      size_t idx = x * h + y;      // ← key fix
      unsigned char rgb[3];
      switch (lab[idx]) {
        case 0: rgb[0]=255; rgb[1]=255; rgb[2]=255; break; // white
        case 1: rgb[0]=255; rgb[1]=  0; rgb[2]=  0; break; // red
        case 2: rgb[0]=  0; rgb[1]=  0; rgb[2]=255; break; // blue
        default: rgb[0]=  0; rgb[1]=  0; rgb[2]=  0; break; // black
      }
      fwrite(rgb, 1, 3, f);
    }
  }

  fclose(f);

  char cwd[4096];
  if (getcwd(cwd, sizeof(cwd))) {
    fprintf(stderr, "[write_ppm_iter] Wrote %s/%s\n", cwd, path);
  } else {
    fprintf(stderr, "[write_ppm_iter] Wrote %s\n", path);
  }

  free(path);
  return 1;  // consumed in SaC to prevent DCE
}
