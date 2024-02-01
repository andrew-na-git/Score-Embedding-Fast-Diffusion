
#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*    prototypes  */

#include "solver.h"

#include "mindeg.h"

void solve_system(double* A, double* b, int* ia, int* ja, double* x, int n, int norder) {

  /*  needs to be size n+1 for colamd */
  int* lorder = (int*)calloc(n + 1, sizeof(int));
  assert(lorder != NULL);

  int* invord = (int*)calloc(n + 1, sizeof(int));
  assert(invord != NULL);

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

  struct sparse_row* rowsp;
  if (sfac_type == 0) {
    rowsp = sfac_dumb(ia, ja, n, lorder, invord, &nzero, &ier);
  } else {
    rowsp = sfac_smart(ia, ja, n, lorder, invord, &nzero, &ier);
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

  factor(ia, ja, lorder, invord, n, A, rowsp);

  // printf("time after num factor %12.4g\n",
  //        (double)clock() / CLOCKS_PER_SEC);

  solve(x, b, n, lorder, rowsp);

  // printf("time after solve %12.4g\n",
  //        (double)clock() / CLOCKS_PER_SEC);

  /*       free storage for ilu factors   */

  clean(n, rowsp);

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

void solve_sparse(double* arr, double* b, double* x, int n, int order) {

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

  solve_system(A, b, ia, ja, x, n, order);
}

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

  double* x = (double*)malloc(n * sizeof(double));
  solve_sparse(a, b, x, n, 1);

  for (int i = 0; i < n; ++i) {
    printf("%f ", x[i]);
  }
  printf("\n");
  free(x);
}
