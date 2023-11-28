
#include "colamd.h"
#include <assert.h>
#include <stdio.h>

int mindeg /* return (1) if OK, (0) otherwise */
    (
        int n,      /* number of rows and columns of A */
        int* ja,    /* col indices of A */
        int* ia,    /* col pointers of A */
        int* lorder /* output permutation, size n */
    ) {
  int stats[COLAMD_STATS]; /* output statistics and error codes */
  double* knobs;

  int* perm;
  int test = 1;
  int i;

  knobs = NULL; /* use default params */

  perm = (int*)calloc(n + 1, sizeof(int));
  assert(perm != NULL);
  /*  note this has to be of size n+1 */

  test = symamd(n, ja, ia, perm,
                knobs, stats, calloc, free);

  if (test == 0) {
    symamd_report(stats);
  }

  assert(test != 0);

  for (i = 0; i < n; i++) {
    lorder[i] = perm[i];
  }

  free(perm);

  return (test);
}
