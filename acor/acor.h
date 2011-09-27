
int acor(         // The version of acor that takes the data as an argument. 
                  // This is the one that does the actual work.
				  
   double *mean,    //   A return value -- the mean of X, or possibly a slightly shorter sequence.
   double *sigma,   //   A return value -- an estimate of the standard devation of the sample mean.
   double *tau,     //   A return value -- an estimate of the autocorrelation time.
   double X[],      //   The time series to be analized.
   int L);          //   The length of the array X[].