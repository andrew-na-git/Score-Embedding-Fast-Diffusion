
#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*    prototypes  */

#include "solver.h"

#include "mindeg.h"

double* solve_system(double* A, double* b, int* ia, int* ja, int n) {

  /*  needs to be size n+1 for colamd */
  int* lorder = (int*)calloc(n + 1, sizeof(int));
  assert(lorder != NULL);

  int* invord = (int*)calloc(n + 1, sizeof(int));
  assert(invord != NULL);

  double* x = (double*)calloc(n, sizeof(double));
  assert(x != NULL);

  int norder = 2;
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
  int cur, nnz, i;
  pthread_mutex_lock(&mat->lock);

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

double* solve_sparse(double* arr, double* b, int n) {
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

  free(mat.ja);
  free(mat.row_nnz);
  return solve_system(A, b, ia, ja, n);
}

int main() {
  double* a = calloc(16, sizeof(double));
  double* b = calloc(4, sizeof(double));
  a[0] = 1;
  a[5] = 1;
  a[10] = 1;
  a[15] = 1;
  b[0] = 1;
  b[1] = 1;
  b[2] = 1;
  b[3] = 1;

  solve_sparse(a, b, 4);
  // int* ja = calloc(7, sizeof(int));
  // int* ia = calloc(5, sizeof(int));

  // ia[0] = 0;
  // ia[1] = 3;
  // ia[2] = 6;
  // ia[3] = 11;
  // ia[4] = 14;

  // a[0] = 0.5779919165240965;
  // a[1] = 0.0;
  // a[2] = 0.021420236007671623;
  // a[3] = 0.6475808726201425;
  // a[4] = 0.4898129709214931;
  // a[5] = 0.827702815824488;
  // a[6] = 0.0;
  // a[7] = 0.2958860195973627;
  // a[8] = 0.5029956372352566;
  // a[9] = 0.9802269040754582;
  // a[10] = 0.9271438579403336;
  // a[11] = 0.0;
  // a[12] = 0.3469931522844616;
  // a[13] = 0.0;

  // ja[0] = 1;
  // ja[1] = 2;
  // ja[2] = 3;
  // ja[3] = 0;
  // ja[4] = 1;
  // ja[5] = 2;
  // ja[6] = 3;
  // ja[7] = 0;
  // ja[8] = 1;
  // ja[9] = 2;
  // ja[10] = 3;
  // ja[11] = 0;
  // ja[12] = 1;
  // ja[13] = 2;
  // // for (int i = 0; i < 3; ++i) {
  // //   for (int j = 0; j < 3; ++j) {
  // //     a[i * 3 + j] = i * 3 + j + 1;
  // //     ja[i * 3 + j] = j;
  // //   }
  // // }
  // // a[8] = 10;

  // b[0] = 1;
  // b[1] = 2;
  // b[2] = 3;
  // b[3] = 4;

  // double* x = solve_system(a, b, ia, ja, 3);

  // for (int i = 0; i < 3; ++i) {
  //   printf("%f ", x[i]);
  // }
  // printf("\n");
}
