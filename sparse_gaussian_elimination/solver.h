
void shell(int* v, int n);
/*    shellsort:  sort v[0],... v[n-1]
      into increasing order */

void sortq( int* v, int left, int right);
        /*  quicksort   */

void swap( int* v, int i, int j);
         /* used by sortq */


int min(  int i,  int j);
 /*    find min of two integers     */

double dblemax( double a, double b);
  /*  max of two doubles */


void  merge_smart( int i,  int* list,
           int n ,  int* link);
/*  see comments in source code */


void  merge_dumb( int i,  int* list,
           int n, int first);
/* see comments in source file */


void sfac_smart(
  int* ia, int* ja, int n, int* lorder, int* invord,
    int* nzero, int* ier);

/*   smart symbolic factor

  input

    ia, ja     structure of a
    n          number of unknowns
    lorder[]   lorder[new_order] = old_order
    invord[]   invord[old_order] = new_order

  output:
    *nzero    number of nonzeros in ILU
    *ier      error flag
    *rowsp    structure contains symbolic ILU info
              (global static)
*/


void sfac_dumb(
   int* ia, int* ja, int n, int* lorder, int* invord,
    int* nzero, int* ier);

/*   dumb symbolic factor

  input

    ia, ja     structure of a
    n          number of unknowns
    lorder[]   lorder[new_order] = old_order
    invord[]   invord[old_order] = new_order

  output:
    *nzero    number of nonzeros in ILU
    *ier      error flag
    *rowsp    stucture contains symbolic ILU info
              (global static)
*/



void factor(int* ia, int* ja, int* lorder, int* invord,
                int n,  double* a);

/* numeric factor   */

/*
   input
     ia, ja     stucture of a
    lorder[]    lorder[new_order] = old_order
    invord      invord[old_order] = new_order
    n           number of unknowns
    a[]         real values of a
    *rowsp      contains info about symbolic factors
                of the ILU (global static)
output
    *rowsp      now contains real LU as well as symbolic
                (ptr to global structure)
*/



void solve( double* x, double* b,  int n,
                  int* lorder);

/*     forward and back solve
         solve LU x = b

input:

   b[]    rhs
   n      number of unknowns
   lorder[]   lorder[new_order] = old_order
   *rowsp  ptr to structure containing ILU
           (global in this file)

output:

    x[]   original order (must be allocated in caller)
*/


void clean( int n );

/*
   input:
     n    number of unknowns
     *rowsp  structure containing LU (global static)

   output:
     *rowsp  structure deallocated

*/


 void rcm( int ia[], int ja[], int n, int lorder []);
   /* rcm ordering */
/*
   input:
      ia, ja, n
   output:
     lorder[0,..,n-1]   lorder[new_order] = old_order
                        must be allocated in caller
*/


