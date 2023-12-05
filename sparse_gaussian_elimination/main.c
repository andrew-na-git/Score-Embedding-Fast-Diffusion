
#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*    prototypes  */

#include "solver.h"

#include "mindeg.h"

double* solve_system(double* A, double* b, int* ia, int* ja, int n, int norder) {

  /*  needs to be size n+1 for colamd */
  int* lorder = (int*)calloc(n + 1, sizeof(int));
  assert(lorder != NULL);

  int* invord = (int*)calloc(n + 1, sizeof(int));
  assert(invord != NULL);

  double* x = (double*)calloc(n, sizeof(double));
  assert(x != NULL);

  int test_order, i, ier, temp, nzero, status;
  int sfac_type = 1;

  if (norder == 0) {
    for (i = 0; i <= n - 1; i++) {
      lorder[i] = i;
    }
  }
  if (norder == 1) {
    rcm(ia, ja, n, lorder);
  }

  if (norder == 2) {
    test_order = mindeg(n, ja, ia, lorder);
    assert(test_order != 0);
  }

  /*     construct inverse order */

  for (i = 0; i <= n - 1; i++) {
    invord[lorder[i]] = i;
  }

  // printf("time before symbolic factor %12.4g\n",
  //        (double)clock() / CLOCKS_PER_SEC);

  if (sfac_type == 0) {
    sfac_dumb(ia, ja, n, lorder, invord, &nzero, &ier);
  } else {
    sfac_smart(ia, ja, n, lorder, invord, &nzero, &ier);
  }

  // printf("time after symm factor %12.4g\n",
  //        (double)clock() / CLOCKS_PER_SEC);

  if (ier != 0) {
    printf(" error in storage allocation: sfac2\n");
    status = 0;
    exit(status);
  }
  // printf(" nonzeros in factors = %d \n", nzero);

  // printf("time before num factor %12.4g\n",
  //        (double)clock() / CLOCKS_PER_SEC);

  factor(ia, ja, lorder, invord, n, A);

  // printf("time after num factor %12.4g\n",
  //        (double)clock() / CLOCKS_PER_SEC);

  solve(x, b, n, lorder);

  // printf("time after solve %12.4g\n",
  //        (double)clock() / CLOCKS_PER_SEC);

  /*       free storage for ilu factors   */

  clean(n);

  double maxerr = 0.0;
  for (i = 0; i <= n - 1; i++) {
    temp = x[i] - (double)(i + 1);
    temp = fabs(temp / x[i]);
    maxerr = dblemax(temp, maxerr);
  }
  // printf("max relative error = %12.4g \n", maxerr);

  /*     clean up memory    */

  free(lorder);
  free(invord);

  return x;
}

struct CSRMatrix {
  int index;
  const int n;
  const double* A;
  int total_nnz;
  int* row_nnz;
  int* ja;
  pthread_mutex_t lock;
};

void* thread_process(void* mat_struct) {
  struct CSRMatrix* mat = (struct CSRMatrix*)mat_struct;
  pthread_mutex_lock(&mat->lock);
  int cur, nnz, i;
  while (mat->index < mat->n) {
    cur = mat->index;
    mat->index += 1;
    pthread_mutex_unlock(&mat->lock);
    nnz = 0;

    for (i = 0; i < mat->n; ++i) {
      if (mat->A[cur * mat->n + i] != 0 || mat->A[i * mat->n + cur] != 0) {
        mat->ja[cur * mat->n + nnz] = i;
        nnz += 1;
      }
    }
    mat->row_nnz[cur] = nnz;
    pthread_mutex_lock(&mat->lock);
    mat->total_nnz += nnz;
  }
  pthread_mutex_unlock(&mat->lock);

  return NULL;
}

double* solve_sparse(double* arr, double* b, int n, int order) {

  if (order == 1 && arr[1] == 0) {
    order = 2;
  }

  struct CSRMatrix mat = {
      0,
      n,
      arr,
      0,
      (int*)malloc(sizeof(int) * n),
      (int*)malloc(sizeof(int) * n * n),
      PTHREAD_MUTEX_INITIALIZER};

  pthread_mutex_init(&mat.lock, NULL);

  int i, j;

  const int num_threads = 1;
  pthread_t* threads = malloc(sizeof(pthread_t) * num_threads);
  int create_status;
  for (i = 0; i < num_threads; ++i) {
    create_status = pthread_create(&threads[i], NULL, thread_process, (void*)&mat);
    assert(create_status == 0);
  }

  for (i = 0; i < num_threads; ++i) {
    pthread_join(threads[i], NULL);
  }

  int* ia = malloc(sizeof(int) * (n + 1));
  int* ja = malloc(sizeof(int) * mat.total_nnz);
  double* A = malloc(sizeof(double) * mat.total_nnz);

  ia[0] = 0;

  for (i = 0; i < n; ++i) {
    ia[i + 1] = ia[i] + mat.row_nnz[i];
  }

  int count = 0;

  for (i = 0; i < n; ++i) {
    for (j = 0; j < mat.row_nnz[i]; ++j) {
      ja[count] = mat.ja[i * n + j];
      A[count] = arr[i * n + ja[count]];
      count += 1;
    }
  }

  return solve_system(A, b, ia, ja, n, order);
}

// double* solve_sparse(double* arr, double* b, int n) {
//   int i = 0;
//   int j;
//   int nnz = 0;

//   double* A = (double*)malloc((n * n) * sizeof(double));
//   int* ia = (int*)malloc((n + 1) * sizeof(int));
//   int* ja = (int*)malloc((n * n) * sizeof(int));

//   assert(A != NULL);
//   assert(ia != NULL);
//   assert(ja != NULL);
//   ia[0] = nnz;
//   while (i < n) {
//     j = 0;
//     while (j < n) {
//       if (arr[i * n + j] != 0 || arr[j * n + i] != 0) {
//         A[nnz] = arr[i * n + j];
//         ja[nnz] = j;

//         nnz += 1;
//       }
//       j += 1;
//     }
//     i += 1;
//     ia[i] = nnz;
//   }

//   return solve_system(A, b, ia, ja, n);
// }

int main() {
  int n = 4;
  double* a = calloc(n * n, sizeof(double));
  double* b = calloc(n, sizeof(double));
  a[0] = 1;
  a[5] = 1;
  a[10] = 1;
  a[15] = 1;
  b[0] = 1;
  b[1] = 1;
  b[2] = 1;
  b[3] = 1;

  double* x = solve_sparse(a, b, n, 1);

  for (int i = 0; i < n; ++i) {
    printf("%f ", x[i]);
  }
  printf("\n");
  free(x);
}

int c_square(int i) {
  return i * i;
}

// int main()

// {

//   /*
//        variable definitions:

//         a                 matrix stored as linear array
//         b                 right hand side
//        ia,ja             usual ia, ja arrays
//        ier               =0 if enough space for symbolic factor
//                           ne 0 on return from symfac if not
//                           enough space

//         invord           inverse of lorder:  invord( lorder(i) ) = i

//         lorder           ordering vector: lorder( new_order ) = old_order
//         n                number of unknowns
//         nja              size of ja, a, arrays
//         norder           =0 original ordering
//                          =1 RCM ordering
//                          =2 Min Degree

//         nx, ny           graph of matrix is regular rectangular grid
//                          of size nx * ny
//         sfac_type        = 0 dumb
//                          = 1 smart
//         x                solution (original ordering)

//   */

//   int nx, ny, n, nja, norder, nz, sfac_type;

//   int *ia, *ja, *lorder, nzero, *invord,
//       ier;
//   int itemp, ix, iy, ibk,
//       id, i, isave, ii, iz, status;
//   double *a,
//       *x, *b, temp, maxerr;

//   int test_order;

//   nx = 5;
//   ny = 5;
//   nz = 3;

//   // n = nx * ny * nz;
//   n = 3;
//   // nja = 7 * n;
//   nja = 25;
//   norder = 0;

//   sfac_type = 1;

//   /*         allocate space for vectors    */

//   ia = (int*)calloc(n + 1, sizeof(int));
//   assert(ia != NULL);
//   ja = (int*)calloc(nja, sizeof(int));
//   assert(ja != NULL);

//   /*  needs to be size n+1 for colamd */
//   lorder = (int*)calloc(n + 1, sizeof(int));
//   assert(lorder != NULL);

//   invord = (int*)calloc(n + 1, sizeof(int));
//   assert(invord != NULL);

//   a = (double*)calloc(9, sizeof(double));
//   assert(a != NULL);
//   x = (double*)calloc(n, sizeof(double));
//   assert(x != NULL);
//   b = (double*)calloc(n, sizeof(double));
//   assert(b != NULL);

//   /*
//           set up matrix with nearest neighbour connections on a grid
//   */

//   ia[0] = 0;
//   ia[1] = 3;
//   ia[2] = 6;
//   ia[3] = 9;

//   for (int i = 0; i < 3; ++i) {
//     for (int j = 0; j < 3; ++j) {
//       a[i * 3 + j] = i * 3 + j + 1;
//       ja[i * 3 + j] = j;
//     }
//   }
//   a[8] = 10;
//   // itemp = -1;
//   // for (iz = 1; iz <= nz; iz++) {
//   //   for (iy = 1; iy <= ny; iy++) {
//   //     for (ix = 1; ix <= nx; ix++) {

//   //       ibk = (iz - 1) * nx * ny + (iy - 1) * nx + ix - 1;

//   //       if (iz - 1 >= 1) {
//   //         id = ibk - nx * ny;
//   //         itemp = itemp + 1;
//   //         if (itemp > nja) {
//   //           printf("nja too small\n");
//   //           status = 0;
//   //           exit(status);
//   //         }
//   //         ja[itemp] = id;
//   //       }

//   //       if (iy - 1 >= 1) {
//   //         id = ibk - nx;
//   //         itemp = itemp + 1;
//   //         if (itemp > nja) {
//   //           printf("nja too small\n");
//   //           status = 0;
//   //           exit(status);
//   //         }
//   //         ja[itemp] = id;
//   //       }

//   //       if (ix - 1 >= 1) {
//   //         id = ibk - 1;
//   //         itemp = itemp + 1;
//   //         if (itemp > nja) {
//   //           printf("nja too small\n");
//   //           status = 0;
//   //           exit(status);
//   //         }
//   //         ja[itemp] = id;
//   //       }

//   //       itemp = itemp + 1;
//   //       if (itemp > nja) {
//   //         printf("nja too small\n");
//   //         status = 0;
//   //         exit(status);
//   //       }
//   //       ja[itemp] = ibk;

//   //       if (ix + 1 <= nx) {
//   //         id = ibk + 1;
//   //         itemp = itemp + 1;
//   //         if (itemp > nja) {
//   //           printf("nja too small\n");
//   //           status = 0;
//   //           exit(status);
//   //         }
//   //         ja[itemp] = id;
//   //       }

//   //       if (iy + 1 <= ny) {
//   //         id = ibk + nx;
//   //         itemp = itemp + 1;
//   //         if (itemp > nja) {
//   //           printf("nja too small\n");
//   //           status = 0;
//   //           exit(status);
//   //         }
//   //         ja[itemp] = id;
//   //       }

//   //       if (iz + 1 <= nz) {
//   //         id = ibk + nx * ny;
//   //         itemp = itemp + 1;
//   //         if (itemp > nja) {
//   //           printf("nja too small\n");
//   //           status = 0;
//   //           exit(status);
//   //         }
//   //         ja[itemp] = id;
//   //       }

//   //       ia[ibk + 1] = itemp + 1;
//   //     }
//   //   }
//   // }

//   /*      construct test matrix--diag dom M-matrix */

//   // for (i = 0; i <= n - 1; i++) {
//   //   isave = -1;
//   //   temp = 0.0;
//   //   printf("row %d, %d \n", i, ia[i]);
//   //   for (ii = ia[i]; ii <= ia[i + 1] - 1; ii++) {
//   //     if (ja[ii] != i) {
//   //       a[ii] = -1.0;
//   //       temp += a[ii];
//   //       printf("%d ", -1);
//   //     } else {
//   //       isave = ii;
//   //       printf("save:%d ", isave);
//   //     }
//   //   }
//   //   assert(isave != -1);
//   //   printf("saved:%f \n", -temp);
//   //   a[isave] = -temp + .1;
//   // }

//   // for (i = 0; i < n; ++i) {
//   //   for (ii = ia[i]; ii < ia[i + 1] - 1; ++ii) {
//   //     a[ii] = ii;
//   //     printf("%d ", ii);
//   //   }

//   //   printf("\n");
//   // }
//   /*

//        construct rhs, so that answer should be 1,2,3,4,5....

//   */
//   // for (i = 0; i <= n - 1; i++) {
//   //   b[i] = 0.0;
//   //   for (ii = ia[i]; ii <= ia[i + 1] - 1; ii++) {
//   //     b[i] += a[ii] * (double)(ja[ii] + 1);

//   //     printf("%f ", a[ii] * (double)(ja[ii] + 1));
//   //   }
//   // }

//   for (i = 0; i < 3; ++i) {
//     b[i] = i + 1;
//   }

//   /*  ordering */

//   if (norder == 0) {
//     for (i = 0; i <= n - 1; i++) {
//       lorder[i] = i;
//     }
//   }
//   if (norder == 1) {
//     rcm(ia, ja, n, lorder);
//   }

//   if (norder == 2) {
//     test_order = mindeg(n, ja, ia, lorder);
//     assert(test_order != 0);
//   }

//   /*     construct inverse order */

//   for (i = 0; i <= n - 1; i++) {
//     invord[lorder[i]] = i;
//   }

//   printf("time before symbolic factor %12.4g\n",
//          (double)clock() / CLOCKS_PER_SEC);

//   if (sfac_type == 0) {
//     sfac_dumb(ia, ja, n, lorder, invord, &nzero, &ier);
//   } else {
//     sfac_smart(ia, ja, n, lorder, invord, &nzero, &ier);
//   }

//   printf("time after symm factor %12.4g\n",
//          (double)clock() / CLOCKS_PER_SEC);

//   if (ier != 0) {
//     printf(" error in storage allocation: sfac2\n");
//     status = 0;
//     exit(status);
//   }
//   printf(" nonzeros in factors = %d \n", nzero);

//   printf("time before num factor %12.4g\n",
//          (double)clock() / CLOCKS_PER_SEC);

//   factor(ia, ja, lorder, invord, n, a);

//   printf("time after num factor %12.4g\n",
//          (double)clock() / CLOCKS_PER_SEC);

//   solve(x, b, n, lorder);

//   for (int i = 0; i < n; ++i) {
//     printf("%f ", x[i]);
//   }

//   printf("time after solve %12.4g\n",
//          (double)clock() / CLOCKS_PER_SEC);

//   /*       free storage for ilu factors   */

//   clean(n);

//   maxerr = 0.0;
//   for (i = 0; i <= n - 1; i++) {
//     temp = x[i] - (double)(i + 1);
//     temp = fabs(temp / x[i]);
//     maxerr = dblemax(temp, maxerr);
//   }
//   printf("max relative error = %12.4g \n", maxerr);

//   /*     clean up memory    */

//   free(ia);
//   free(ja);
//   free(lorder);
//   free(invord);

//   free(a);
//   free(x);
//   free(b);

//   return (0);
// }
