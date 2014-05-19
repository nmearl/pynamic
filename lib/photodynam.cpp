#include "n_body_state.h"
#include "n_body_lc.h"
#include "omp.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

using namespace std;

extern "C"
{
  void start(double *time, int time_size,
            int N, double t0, double maxh, double orbit_error,
            double *masses, double *radii, double *fluxes, double *u1, double *u2, double *a, double *e,
            double *inc, double *om, double *ln, double *ma,
            double *mod_flux, double *mod_rv, int rv_body);
}

void start(double *time, int time_size,
            int N, double t0, double maxh, double orbit_error,
            double *masses, double *radii, double *fluxes, double *u1, double *u2, double *a, double *e,
            double *inc, double *om, double *ln, double *ma,
            double *mod_flux, double *mod_rv, int rv_body)
{

    // Instantiate state; time t0 is epoch of above coordinates
    NBodyState state(masses, a, e, inc, om, ln, ma, N, t0);

    // Integrate forward in time with stepsize (maxh), error tolerance (orbit_error)
    // and minimum step size of 1e-20

    // Uncomment the line below if you want mpi multithreading
    // #pragma omp parallel for
    for (int i = 0; i < time_size; i++)
    {
        // Evaluate the flux at time t0 using the getBaryLT() member method
        // of NBodyState which returns NX3 array of barycentric, light-time
        // corrected coordinates
        state(time[i], maxh, orbit_error, 1.0e-20);

        // Now get the flux at the new time
        mod_flux[i] = occultn(state.getBaryLT(),radii,u1,u2,fluxes,N);
        mod_rv[i] = state.V_Z_LT(rv_body);
    }

    // for (int i = 0; i < rv_time_size; i++)
    // {
    //   state(rv_time[i], maxh, orbit_error, 1.0e-20);

    //   mod_rv[i] = state.V_Z_LT(2);      
    // }
}
