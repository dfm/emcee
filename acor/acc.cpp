/*  The code that does the acor analysis of the time series.  See the README file for details.  */

#include <iostream>
#include <string>
#include <cstdio>
#include <vector>
#include <math.h>

#include "acor.h"

using namespace std;

#define TAUMAX  2                 /*   Compute tau directly only if tau < TAUMAX.  
                                       Otherwise compute tau using the pairwise sum series          */
#define WINMULT 5                 /*   Compute autocovariances up to lag s = WINMULT*TAU            */
#define MAXLAG  TAUMAX*WINMULT    /*   The autocovariance array is double C[MAXLAG+1] so that C[s]
                                       makes sense for s = MAXLAG.                                  */
#define MINFAC   5                /*   Stop and print an error message if the array is shorter
                                       than MINFAC * MAXLAG.                                        */


/*  Jonathan Goodman, March 2009, goodman@cims.nyu.edu  */

int acor( double *mean, double *sigma, double *tau, double X[], int L){
   
   *mean = 0.;                                   // Compute the mean of X ... 
   for ( int i = 0; i < L; i++) *mean += X[i];
   *mean = *mean / L;
   for ( int i = 0; i <  L; i++ ) X[i] -= *mean;    //  ... and subtract it away.
   
   if ( L < MINFAC*MAXLAG ) {
      cout << "Acor error 1: The autocorrelation time is too long relative to the variance." << endl;
	  return 1; }
   
   double C[MAXLAG+1]; 
   for ( int s = 0; s <= MAXLAG; s++ )  C[s] = 0.;  // Here, s=0 is the variance, s = MAXLAG is the last one computed.
     
   int iMax = L - MAXLAG;                                 // Compute the autocovariance function . . . 
   for ( int i = 0; i < iMax; i++ ) 
      for ( int s = 0; s <= MAXLAG; s++ )
	     C[s] += X[i]*X[i+s];                              // ...  first the inner products ...
   for ( int s = 0; s <= MAXLAG; s++ ) C[s] = C[s]/iMax;   // ...  then the normalization.
      
   double D = C[0];   // The "diffusion coefficient" is the sum of the autocovariances
   for ( int s = 1; s <= MAXLAG; s++ ) D += 2*C[s];   // The rest of the C[s] are double counted since C[-s] = C[s].
   *sigma = sqrt( D / L );                            // The standard error bar formula, if D were the complete sum.
   *tau   = D / C[0];                                 // A provisional estimate, since D is only part of the complete sum.
   
   if ( *tau*WINMULT < MAXLAG ) return 0;             // Stop if the D sum includes the given multiple of tau.
                                                      // This is the self consistent window approach.
													  
   else {                                             // If the provisional tau is so large that we don't think tau
                                                      // is accurate, apply the acor procedure to the pairwase sums
													  // of X.
      int Lh = L/2;                                   // The pairwise sequence is half the length (if L is even)
	  double newMean;                                 // The mean of the new sequence, to throw away.
	  int j1 = 0;
	  int j2 = 1;
	  for ( int i = 0; i < Lh; i++ ) {
	     X[i] = X[j1] + X[j2];
		 j1  += 2;
		 j2  += 2; }
	  acor( &newMean, sigma, tau, X, Lh);
	  D      = .25*(*sigma) * (*sigma) * L;    // Reconstruct the fine time series numbers from the coarse series numbers.
	  *tau   = D/C[0];                         // As before, but with a corrected D.
	  *sigma = sqrt( D/L );                    // As before, again.
	}
	  
	 
   return 0;
  }