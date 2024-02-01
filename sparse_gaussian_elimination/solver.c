#include <stdio.h>   
#include <math.h>
#include <assert.h>   
#include <stdlib.h>

#include "solver.h"
      /* sparse_row defined in solver.h */

//struct sparse_row *rowsp;
        /* globally visible structure */

/*   
     this simulates an object in C, using a single
     globally visible data structure and functions
     which operate on that structure
*/



/*    function definitions     */

 
void  merge_smart( int i,  int list[],
           int n ,  int link[], struct sparse_row *rowsp) 

/*
  input

     i     current row being factored (Dolittle form)
     list[]  implied linked list of nonzeros in current
             row.  On entry, only original nonzero columsn
             in this row.  1st nonzero column = first
             , list[first], list[ list[first]], ...., until
             list[  [....] ] = n+1 -> signals end of list
    n       number of nonzeros
    *rowsp  ptr to global structure containing info about
            previous rows

    link[]  implied linked list of rows whose first non-zero is in col i, i.e.
            list of prev rows whose first nonzero col is in col i given by
            link[i], link[link[i]}, ..., until link[  [....] ] = n+1 -> signals end of list

  output

    list[]   now contains all nonzeros in this factored row i
             (implied linked list)
*/


{
      //extern struct sparse_row *rowsp;
   int row, next, oldlst,  ii;
   int nxtlst;


      next = link[i];

   while (  next  != n+1) {

        oldlst = i;
        nxtlst = list[i];
        row = next;
 

/*          scan row "row" of U   */
 
    for( ii = rowsp[row].diag+1 ; ii <= rowsp[row].nz - 1 ; ii++) {
 
/*     skip thru linked list until we find possible insertion pt */

           while( rowsp[row].jaf[ii] > nxtlst) {
                 oldlst = nxtlst;
                 nxtlst = list[oldlst]; 
             }
                
           if( rowsp[row].jaf[ii] < nxtlst ){

 
                  list[ oldlst] = rowsp[row].jaf[ii];
                  list[ rowsp[row].jaf[ii]  ] = nxtlst;
                  oldlst = rowsp[row].jaf[ii];
             }
           else {
              oldlst = nxtlst;
              nxtlst = list[oldlst ];
            }
        }
           next = link[ next ];

  }
      return  ;
}

/*    find min of two integers     */

int min(  int i,  int j)

  { if( i < j ) {
         return i; }
      else {
         return j; }
   }

double dblemax( double a, double b)
 /*  max of two doubles */

{
  if( a > b){
       return a;
   }
   else{
       return b;
    }
 }/* end dblemax */


/*    shellsort:  sort v[0],... v[*n_ptr-1] 
      into increasing order */

void shell(int* v, int n)

{  int gap, i, j, temp;

    for (gap = n/2; gap > 0; gap = gap/2){
       for (i = gap; i < n; i++){
          for( j = i - gap; j>=0 && v[j]>v[j+gap]; j=j-gap){
              temp = v[j];
              v[j] = v[j+gap];
              v[j+gap] = temp;
          }
       }
    }
}


void swap( int* v, int i, int j)
 {
  /* function used by sortq */

            int temp;
            temp = v[i];
            v[i] = v[j];
            v[j] = temp;
 }/* end swap */


void sortq( int* v, int left, int right)
 {
   /* simple quick sort */

    int i;
    int last;

   int num_sort;

    num_sort = right - left + 1;

    if( num_sort < 40 ){
         shell( &v[left], num_sort);
         return;
      }


     if( left >= right){
         return;
     }

    swap( v, left, (left + right)/2);

    last = left;

    for(i=left+1; i<= right; i++){
        if( v[i] < v[left]){
             swap(v, ++last, i);
         }
     }

     swap(v, left, last);

     sortq( v, left, last-1);
     sortq( v, last+1, right);

 }/* end sortq  */

/*    symblic factor */

 struct sparse_row* sfac_smart(
  int* ia, int* ja, int n, int* lorder, int* invord,
    int* nzero, int* ier)

/*   symbolic factor

  input

    ia, ja     structure of a
    n          number of unknowns
    lorder[]   lorder[new_order] = old_order
    invord[]   invord[old_order] = new_order
 
  output:
    *nzero    number of nonzeros in LU
    *ier      error flag
    *rowsp    stucture contains symbolic ILU info
              (global static)
*/


{
      int  *list;
      int i,   ii;
     int   iold, iend, num,  next;
      int nrow_tmp, icol, itemp;
      int  nrow,  found;
      //extern struct sparse_row *rowsp;
      int *int_temp, *link, *icount;
 

/*     allocate array of structures, each structue points to  row  */

    struct sparse_row* rowsp = (struct sparse_row *) calloc( n, sizeof( struct sparse_row) );
         assert( rowsp != NULL);

/*     allocate temp wkspace      */

      list = (int *) calloc( n , sizeof( int ) );
         assert( list != NULL);
      int_temp = (int *) calloc( n , sizeof( int ) );
         assert( int_temp != NULL);
      link = (int *) calloc( n , sizeof( int ) );
         assert( link != NULL);
       icount = (int *) calloc( n , sizeof( int ) );
         assert( icount != NULL);


 
       *ier = 0;
 
 
       for (i=0; i <= n-1; i++){
          list[i] = n+1;
          link[i] = n+1;
          icount[i] = 0; /* used to keep trak of number of
                         nonzeros in a row, based on symmetry
                         since we get only struct of U from
                         smart algorithm
                        */
       }
 
 
       itemp = 0;
       *nzero = 0;
     for( i = 0; i <= n-1; i++ ) {
         iold = lorder[i];
 
         iend = 0;
         for( ii=ia[iold]; ii <= ia[iold+1]-1; ii++){
               int_temp[iend] =  invord[ ja[ii] ] ;
               iend++;
         }

/*           sort entries */

         num = ia[iold+1] - ia[iold];
        
         assert( num == iend);
         assert( num >= 1);
         if( num > 1){
             /* shell( int_temp , num); */
             sortq( int_temp, 0, num -1 ); 
         }

         found = 0;
         for (ii=0; ii<=num-1; ii++){
             if( int_temp[ii] == i){
               found = 1;
             }
         }

         assert( found == 1);

          for( ii= 0; ii <= num-2; ii++){
                if( int_temp[ii] >= i ){
                  list[ int_temp[ii] ] = int_temp[ii+1];
                }
          }

          list[ int_temp[num-1]  ] = n+1;


/*          printf(" links for row %d \n",i);
          for(itemp= link[i]; itemp !=0; itemp = link[itemp] ){
               printf(" pre rows %d", itemp);
          }
          printf(" \n");    */
            merge_smart( i,  list, n,  link, rowsp);
 
/*                count up nonzeros in this row  */

           nrow=0;
           next = i;
           while(next != n+1){
              nrow++;
              next = list[ next ];
           }
            nrow = nrow + icount[i];
            rowsp[i].nz =  nrow ;
            *nzero = *nzero + rowsp[i].nz;

/*      allocate space for this row     */

            rowsp[i].jaf = (int *) calloc( nrow , sizeof( int ) );
               assert( rowsp[i].jaf != NULL );
/*    allocate real space for this row   */
 
      rowsp[i].af = (double *) calloc(nrow, sizeof( double ) );
         assert( rowsp[i].af != NULL);

              
 
           next = i;
           nrow_tmp= icount[i];
           while( next != n+1){
                 rowsp[i].jaf[nrow_tmp] = next;
               if( next == i){
                    rowsp[i].diag = nrow_tmp;
               }
               icount[next] = icount[next] + 1;
               next = list[next];
               nrow_tmp++;
            }
               assert( rowsp[i].jaf[ rowsp[i].diag ] == i);

/*                linked list of rows whose first non-zero is in col icol  */

            if( rowsp[i].diag + 1 <= rowsp[i].nz - 1){
                icol = rowsp[i].jaf[ rowsp[i].diag + 1];
                itemp = link[icol];
                link[icol] = i;
                link[i] = itemp;
            }

/*     printf("row = %d \n", i);
      for (ii=rowsp[i].diag; ii<= rowsp[i].nz -1; ii++){
            printf(" icol = %d  ", rowsp[i].jaf[ii]);
      }
      printf(" \n ");   */


    }



/*        now fill in lower triangular    */

        for( i=0; i<=n-1; i++){
           icount[i] = 0;
        }

        for( i=0; i<=n-1; i++){
           for(ii=rowsp[i].diag+1; ii<= rowsp[i].nz-1; ii++){
              icol = rowsp[i].jaf[ii];
              assert( icount[icol] < rowsp[icol].diag );
              rowsp[icol].jaf[ icount[icol] ] = i;
              icount[icol]++;
           }
        }



/*       free integer temp   */

         free(int_temp);
         free( list );
         free( link);
         free( icount);

      return rowsp;
}
      
/*       numeric factor  */

 void factor(int* ia, int* ja, int* lorder, int* invord, 
                int n,  double* a, struct sparse_row *rowsp) 
 
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



{
      int i, iold, id, ii, idd, iii;
      int    *ptr_jaf;
      double mult, *row, *ptr_af;
     //extern struct sparse_row *rowsp;


/*      allocate temp wkspace    */

     row  = (double *) calloc( n , sizeof( double ) );
        assert( row != NULL);

     for(i=0; i<=n-1; i++){ 
         row[i] = 0.0;
    }
 
 
/*  loop over rows   */

   for(i=0; i <=n-1; i++){

         iold = lorder[i];

    /*    load orig matrix elements   */

         for( ii=ia[iold]; ii <= ia[iold+1]-1; ii++){
                row[ invord[ ja[ii]  ]  ] = a[ii];
         }


    /*     now eliminate    */
 
        for( ii = 0; ii <= rowsp[i].diag - 1; ii++){

         id = rowsp[i].jaf[ii];
 
/*          get multiplier    */
 
         mult = row[id] / rowsp[ id ].af[ rowsp[id].diag ];
         row[ id  ] = mult ;
 
         ptr_jaf = rowsp[id].jaf;
         ptr_af = rowsp[id].af;
         for( iii=rowsp[id].diag+1; iii <= rowsp[id].nz - 1; iii++){
            idd = ptr_jaf[ iii ];
              row[idd ] = row[idd ] - mult * ptr_af[ iii  ];
         }
       }
 
   /*     end of elimination for row   */

   /*    gather and reset row[] array   */


         for( ii=0; ii <= rowsp[i].nz - 1; ii++){
             rowsp[i].af[ ii ]= row[ rowsp[i].jaf[ii] ];
             row[ rowsp[i].jaf[ii]  ] = 0.0;
      }

 /*    end of loop over rows   */
 
 }

/*      deallocate temp wkspace   */

       free ( row );
 
      return;
}




         void solve( double* x, double* b,  int n,
                  int* lorder, struct sparse_row *rowsp)

/*     forward and back solve
         solve LU x = b

input:

   b[]    rhs
   n      number of unknowns
   lorder[]   lorder[new_order] = old_order
   *rowsp  ptr to structure containing ILU
           (global in this file)

output:

    x[]   original order
*/


{
      //extern struct sparse_row *rowsp;
      int i, ii, *ptr_jaf;
      double *temp, *ptr_af;
 
/*       forward solve:  Lz = b
         (L has unit diagonal)      */
 

/*    allocate temp   */

      temp = (double *) calloc( n , sizeof( double ) );
         assert( temp != NULL);


   for( i=0; i<n; i++) {
          x[i] = b[ lorder[i] ];

           ptr_jaf = rowsp[i].jaf;
           ptr_af = rowsp[i].af;
           for (ii=0; ii<= rowsp[i].diag - 1; ii++){
               x[i] = x[i] -ptr_af[ii] * x[ ptr_jaf[ii] ];
           }

   }
 
 
/*       back solve Ux = z   */
/*           (U does not have unit diag)    */
      
     for( i=n-1; i >= 0; i--){

           ptr_jaf = rowsp[i].jaf;
           ptr_af = rowsp[i].af;

         for (ii = rowsp[i].diag+1; ii <= rowsp[i].nz -1; ii++){
               x[i] = x[i] - ptr_af[ii] * x[ ptr_jaf[ii] ];
         }
         x[i] = x[i] /rowsp[i].af[ rowsp[i].diag ];
            
     }
 
 
/*      reorder     */

      for( i=0; i<= n-1; i++){
          temp[ lorder[i] ] = x[i];
      }
 
      for( i=0; i<=n-1; i++){
              x[i] = temp[i];
      }
 
      free(temp);

      return;
}

/*    free space for lu in after finished  */
 
 void clean( int n, struct sparse_row *rowsp)

/*
   input:
     n    number of unknowns
     *rowsp  structure containing LU (global static)

   output:
     *rowsp  structure deallocated

*/

 
{
    int   i;
    //extern struct sparse_row *rowsp;
 
 
       for(i=0; i<=n-1; i++){
             free( rowsp[i].jaf );
             free( rowsp[i].af );
         }
 
  /*    now free up space for "rowsp" itself  */
 
        free (rowsp );
 
        return;
}


 struct sparse_row* sfac_dumb(
   int* ia, int* ja, int n, int* lorder, int* invord,
    int* nzero, int* ier)
{
/*   symbolic factor

  input

    ia, ja     structure of a
    n          number of unknowns
    lorder[]   lorder[new_order] = old_order
    invord[]   invord[old_order] = new_order

  output:
    *nzero    number of nonzeros in LU
    *ier      error flag
    *rowsp    stucture contains symbolic ILU info
              (global static)
*/

      int  *list;
      int i,   ii;
     int  first, iold, iend, num, next;
      int  nrow_tmp;
      int itemp, nrow;
      //extern struct sparse_row *rowsp;
      int *int_temp;
 

/*     allocate array of structures, each structue points to  row  */

    struct sparse_row* rowsp = (struct sparse_row *) calloc( n, sizeof( struct sparse_row) );
         assert( rowsp != NULL);

/*     allocate temp wkspace      */

      list = (int *) calloc( n , sizeof( int ) );
         assert( list != NULL);
      int_temp = (int *) calloc( n , sizeof( int ) );
         assert( int_temp != NULL);



 
 
       *ier = 0;
 
 
       for (i=0; i <= n-1; i++){
          list[i] = n+1;
       }
 
 
       itemp = 0;
       *nzero = 0;
     for( i = 0; i <= n-1; i++ ) {
         iold = lorder[i];
 
         iend = 0;
         for( ii=ia[iold]; ii <= ia[iold+1]-1; ii++){
               int_temp[iend] =  invord[ ja[ii] ] ;
               iend++;
         }

/*           sort entries */

         num = ia[iold+1] - ia[iold];
        
         assert( num == iend);
         if( num > 1){
              /* shell( int_temp , num); */
             sortq( int_temp, 0, num -1 ); 

         }

          first = int_temp[ 0 ];
          for( ii= 1; ii <= num-1; ii++){
                  list[ int_temp[ii-1] ] = int_temp[ii];
          }
          list[ int_temp[num-1]  ] = n+1;


 
            merge_dumb( i,  list,
              n, first, rowsp);
 
/*                count up nonzeros in this row  */

           nrow=0;
           next = first;
           while(next != n+1){
              nrow++;
              next = list[ next ];
           }
            rowsp[i].nz =  nrow;
            *nzero = *nzero + nrow;

/*      allocate space for this row     */

            rowsp[i].jaf = (int *) calloc(nrow, sizeof( int ) );
               assert( rowsp[i].jaf != NULL );
              
/*    allocate real space for this row   */

      rowsp[i].af = (double *) calloc(nrow, sizeof( double ) );
         assert( rowsp[i].af != NULL);

 
           next = first;
           nrow_tmp=0;
           while( next != n+1){
                 rowsp[i].jaf[nrow_tmp] = next;
               if( next == i){
                    rowsp[i].diag = nrow_tmp;
               }
               next = list[next];
               nrow_tmp++;
            }
               assert( rowsp[i].jaf[ rowsp[i].diag ] == i);
               assert( nrow_tmp == nrow);

/*    printf("row = %d \n", i);
      for (ii=rowsp[i].diag; ii<= rowsp[i].nz -1; ii++){
            printf(" icol = %d  ", rowsp[i].jaf[ii]);
      }
      printf(" \n ");     */

    }



/*       free integer temp   */

         free(int_temp);
         free( list );

      return rowsp;
}
      
void  merge_dumb( int i,  int* list,
           int n, int first, struct sparse_row *rowsp) 

/*
  input

     i     current row being factored (Dolittle form)
     list[]  implied linked list of nonzeros in current
             row.  On entry, only original nonzero columsn
             in this row.  1st nonzero column = first
             , list[first], list[ list[first]], ...., until
             list[  [....] ] = n+1 -> signals end of list
    n       number of nonzeros
    first   (see list[]) first nonzero in this row
    *rowsp  ptr to global structure containing info about
            previous rows

  output

    list[]   now contains all nonzeros in this factored row i
             (implied linked list)
*/


{
      //extern struct sparse_row *rowsp;
   int row, next, oldlst,  ii;
   int nxtlst;


      next = first;

   while (  next < i) {

        oldlst = next;
        nxtlst = list[next];
        row = next;
 

/*          scan row "row" of U   */
 
    for( ii = rowsp[row].diag+1 ; ii <= rowsp[row].nz - 1 ; ii++) {
 
/*     skip thru linked list until we find possible insertion pt */

           while( rowsp[row].jaf[ii] > nxtlst) {
                 oldlst = nxtlst;
                 nxtlst = list[oldlst]; 
             }
                
           if( rowsp[row].jaf[ii] < nxtlst ){

 
                  list[ oldlst] = rowsp[row].jaf[ii];
                  list[ rowsp[row].jaf[ii]  ] = nxtlst;
                  oldlst = rowsp[row].jaf[ii];
             }
           else {
              oldlst = nxtlst;
              nxtlst = list[oldlst ];
            }
        }
           next = list[ next ];

  }
      return  ;
}

