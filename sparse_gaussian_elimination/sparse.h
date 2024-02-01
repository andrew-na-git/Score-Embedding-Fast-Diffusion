#ifndef SPARSE_H
#define SPARSE_H
  struct sparse_row {
       int *jaf;
       double *af;
       int nz;
       int diag;
   };
/*   
    each row is held as a structure
       for k=0,.., nz-1
       jaf[k] = col index
       af[k]  = value
       k = diag is the location of the diagonal
        of the row  ie k = row.diag -> af[k] = diag entry 
*/

#endif
