
 int mindeg( int n, /*   input: number of rows of A */
             int* ia,  /* input */
             int* ja,  /* input */
             int* lorder /* output */
            );
/*
    n = number of rows
    ia, ja usual ia,ja
    lorder[new_order] = old_order
         must be length n, allocated in caller

   returns 1 if all ok
           0 if failure
*/
        
