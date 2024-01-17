#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* prototypes   */

void rcm(int ia[], int ja[], int n, int lorder[]);

void levels(int ia[], int ja[], int n, int iroot,
            int* nlvl, int ial[], int jal[]);

void rcm1(int ia[], int ja[], int n, int iroot,
          int nlvl, int ial[], int jal[], int lorder[]);

void pseudo(int ia[], int ja[], int n, int ial[],
            int jal[], int* iroot, int* nlvl);

void rcm1(int ia[], int ja[], int n, int iroot,
          int nlvl, int ial[], int jal[], int lorder[]);

/* function defs  */

void rcm(int ia[], int ja[], int n, int lorder[]) {

  /*    determine level structure of ia-ja graph

    input:
         ia, ja  structure of graph
          n       number of nodes
          nja     size of ja() array

  NOTE:   assumes symmetric incidence matrix


     output:

       lorder( new_order ) = old_order


  */

  int i, id, ii, ichek,
      jj, status, nlvl, *ial, *jal, nja, iroot;

  /*      set up working arrays    */

  ial = (int*)calloc(n + 1, sizeof(int));
  assert(ial != NULL);
  nja = (ia[n] - 1) + 1;
  jal = (int*)calloc(nja, sizeof(int));
  assert(jal != NULL);

  /*     check that incidence matrix is symmetric */

  for (i = 0; i <= n - 1; i++) {
    for (ii = ia[i]; ii <= ia[i + 1] - 1; ii++) {
      id = ja[ii];
      ichek = 0;
      for (jj = ia[id]; jj <= ia[id + 1] - 1; jj++) {
        if (ja[jj] == i) {
          if (ichek != 0) {
            printf(" same index appears twice\n");
            printf(" row= %d col= %d \n", id, i);
            status = 1;
            exit(status);
          } else {
            ichek = 1;
          }
        }
      }
      if (ichek == 0) {
        printf("nonsymmetric structure\n");
        status = 1;
        exit(status);
      }
    }
  }
  iroot = 0;
  pseudo(ia, ja, n, ial, jal,
         &iroot, &nlvl);
  rcm1(ia, ja, n, iroot,
       nlvl, ial, jal, lorder);

  free(ial);
  free(jal);

  return;
}

void levels(int ia[], int ja[], int n, int iroot,
            int* nlvl, int ial[], int jal[]) {

  /*    get level structure starting at node "iroot"

     input:

          ia, ja  structure of graph
          n       number of nodes
          nja     size of ja() array
          iroot   starting node

       output:


           nlvl  number of levels in level struc
                 starting at iroot
           ial(), jal()   ptrs which describe nodes
                  in level structure


  */

  int *proces, i, id, idd, icount, nlold, jj;

  proces = (int*)calloc(n, sizeof(int));
  assert(proces != NULL);

  /*   mark all nodes as unprocessed   */

  for (i = 0; i <= n - 1; i++) {
    proces[i] = 1;
  }

  ial[0] = 0;
  icount = 0;
  jal[icount] = iroot;
  proces[iroot] = 0;
  nlold = 0;
  ial[nlold + 1] = icount + 1;
  *nlvl = nlold;

  while (icount < n - 1) {
    *nlvl = *nlvl + 1;
    for (i = ial[nlold]; i <= ial[nlold + 1] - 1; i++) {
      id = jal[i];
      for (jj = ia[id]; jj <= ia[id + 1] - 1; jj++) {
        idd = ja[jj];
        if (proces[idd] != 0) {
          icount++;
          jal[icount] = idd;
          proces[idd] = 0;
        }
      }
    }

    ial[*nlvl + 1] = icount + 1;
    nlold = *nlvl;
  }

  assert(
      (ial[*nlvl + 1] - 1) == (n - 1));

  free(proces);

  return;
}

void pseudo(int ia[], int ja[], int n, int ial[],
            int jal[], int* iroot, int* nlvl)

{

  /*     get pseudo periperhal node

     input:

          ia, ja  structure of graph
          n       number of nodes
          nja     size of ja() array
          iroot   starting node

     output:

        iroot   periperal node found
        nlvl    number of levels in level struc
        ial, jal  nodes in level structure
  */

  int nlold, icount, id, mindeg, i;

  levels(ia, ja, n, *iroot,
         nlvl, ial, jal);

  if (*nlvl == 1 || *nlvl == n) {
    return;
  }

  nlold = -1;
  while (*nlvl > nlold && *nlvl < n - 1) {
    nlold = *nlvl;
    mindeg = n + 1;
    *iroot = jal[ial[*nlvl]];
    for (i = ial[*nlvl]; i <= ial[*nlvl + 1] - 1; i++) {
      id = jal[i];
      icount = ia[id + 1] - ia[id];
      if (icount < mindeg) {
        mindeg = icount;
        *iroot = id;
      }
    }
    assert(mindeg != n + 1);

    levels(ia, ja, n, *iroot,
           nlvl, ial, jal);
  }

  return;
}

void rcm1(int ia[], int ja[], int n, int iroot,
          int nlvl, int ial[], int jal[], int lorder[]) {
  /*
     input:

          ia, ja  structure of graph
          ial, jal   level structure
                     starting at iroot
          n       number of nodes
          nja     size of ja() array
          iroot   starting node

       output:

        lorder(new_order) = old_order

  */

  int *proces, deg, i, ii, jj, id, ilvl, icold;
  int icount, idd, itemp, iii;

  /*    temp vector   */

  proces = (int*)calloc(n, sizeof(int));

  for (i = 0; i <= n - 1; i++) {
    proces[i] = 1;
    lorder[i] = -1;
  }

  proces[iroot] = 0;
  icount = 0;
  lorder[icount] = iroot;

  for (ilvl = 0; ilvl <= nlvl; ilvl++) {

    for (iii = ial[ilvl]; iii <= ial[ilvl + 1] - 1; iii++) {
      icold = icount;
      id = lorder[iii];
      assert(id != -1);
      for (jj = ia[id]; jj <= ia[id + 1] - 1; jj++) {
        idd = ja[jj];
        if (proces[idd] != 0) {
          icount++;
          assert(lorder[icount] == -1);
          lorder[icount] = idd;
          proces[idd] = 0;
        }
      }
      /*   sort nodes in order of incresing degree */

      for (ii = icold + 2; ii <= icount; ii++) {
        deg = ia[lorder[ii] + 1] - ia[lorder[ii]];
        for (jj = ii - 1; jj >= icold + 1; jj--) {
          if (deg >= (ia[lorder[jj] + 1] - ia[lorder[jj]])) {
            break;
          }
          itemp = lorder[jj];
          lorder[jj] = lorder[jj + 1];
          lorder[jj + 1] = itemp;
        }
      }

      /*          end of sort   */

      /*           check ordering    */

      for (ii = icold + 2; ii <= icount; ii++) {
        assert(
            (ia[lorder[ii] + 1] - ia[lorder[ii]]) >= (ia[lorder[ii - 1] + 1] - ia[lorder[ii - 1]]));
      }
    }
  }

  /*   chek total  */

  assert(icount == n - 1);

  /*     reverse order, use ial as temp   */

  for (i = 0, itemp = n - 1; i <= n - 1; i++, itemp--) {
    ial[itemp] = lorder[i];
  }

  for (i = 0; i <= n - 1; i++) {
    lorder[i] = ial[i];
  }

  /*    check ordering     */

  for (i = 0; i <= n - 1; i++) {
    ial[i] = -1;
  }
  for (i = 0; i <= n - 1; i++) {
    assert(lorder[i] <= n - 1 && lorder[i] >= 0);
    assert(ial[lorder[i]] == -1);
    ial[lorder[i]] = +1;
  }

  free(proces);

  return;
}
