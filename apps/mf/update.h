
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/timer/timer.hpp>

typedef boost::numeric::ublas::coordinate_matrix<double, boost::numeric::ublas::row_major> SparseMatrix;

using namespace std;

const double TRUNC = 10000;

class UpdateNsqlL2 {
  uint mf_rank;
  double eps;
  double lambda;

 public:
  UpdateNsqlL2 (const uint _mf_rank, const double _eps, const double _lambda) :mf_rank{_mf_rank}, eps{_eps}, lambda{_lambda} { }

  void operator() (const unsigned long i, const long j, std::vector<double> &factors, std::vector<double> &factors_update, const double x, const int row_nnz_i, const int col_nnz_j) {
    double* w = factors.data();
    double* h = factors.data() + mf_rank;
    double* w_update = factors_update.data();
    double* h_update = factors_update.data() + mf_rank;

    /* stringstream os; */
    /* os << "Train on " << x << " (" << i << "," << j << "), rnnz " << row_nnz_i << ", cnnz " << col_nnz_j << ". w/h:"; */
    /* for(unsigned int z=0; z!=mf_rank; ++z) { */
    /*   os << "  (" << z << ") " << w[z] << "/" << h[z]; */
    /* } */
    /* ADLOG(os.str()); */

    // Inner product of factors
    double wh = 0;
    for(unsigned int z=0; z!=mf_rank; ++z) {
      wh += h[z] * w[z];
    }

    // Pre-calculations
    double f1 = eps * -2. * (x-wh);
    double f2 = eps * 2. * lambda;
    double f3 = 1./row_nnz_i;
    double f4 = 1./col_nnz_j;

    // Assignments
    for (unsigned z=0; z!=mf_rank; ++z) {

      w_update[z] = -(f1 * h[z] + f2 * w[z] * f3);
      h_update[z] = -(f1 * w[z] + f2 * h[z] * f4);

      // Truncate updates
      if(w_update[z] >  TRUNC) w_update[z] =  TRUNC;
      if(w_update[z] < -TRUNC) w_update[z] = -TRUNC;
      if(h_update[z] >  TRUNC) h_update[z] =  TRUNC;
      if(h_update[z] < -TRUNC) h_update[z] = -TRUNC;
    }
  }

  void update_step_size(double factor) {
    eps = eps * factor;
  }

  double current_step_size() {
    return eps;
  }
};
