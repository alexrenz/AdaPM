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
#include <boost/generator_iterator.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/timer/timer.hpp>

typedef boost::numeric::ublas::coordinate_matrix<double, boost::numeric::ublas::row_major> SparseMatrix;

using namespace std;

const double ADAGRAD_EPS = 0.000001;

class UpdateNsqlL2Adagrad {
  uint mf_rank;
  uint param_len;
  double eps;
  double lambda;

 public:
  UpdateNsqlL2Adagrad (const uint _mf_rank, const uint _param_len, const double _eps, const double _lambda) :mf_rank{_mf_rank}, param_len{_param_len}, eps{_eps}, lambda{_lambda} { }

  void operator() (const unsigned long i, const long j,
                   std::vector<double> &factors, std::vector<double> &factors_update,
                   const double x, const int row_nnz_i, const int col_nnz_j) {
    double* w = factors.data();
    double* h = factors.data() + param_len;
    double* w_update = factors_update.data();
    double* h_update = factors_update.data() + param_len;

    // pointers to AdaGrad values
    double* w_ag = factors.data() + mf_rank;
    double* h_ag = factors.data() + param_len + mf_rank;
    double* w_update_ag = factors_update.data() + mf_rank;
    double* h_update_ag = factors_update.data() + param_len + mf_rank;

    // Inner product of factors
    double wh = 0;
    for(unsigned int z=0; z!=mf_rank; ++z) {
      wh += h[z] * w[z];
    }

    // Pre-calculations
    double f1 = -2. * (x-wh);
    double f2 = 2. * lambda;
    double f3 = 1./row_nnz_i;
    double f4 = 1./col_nnz_j;

    // Assignments
    for (unsigned z=0; z!=mf_rank; ++z) {

      w_update[z] = -(f1 * h[z] + f2 * w[z] * f3);
      h_update[z] = -(f1 * w[z] + f2 * h[z] * f4);

      w_update_ag[z] = w_update[z]*w_update[z];
      h_update_ag[z] = h_update[z]*h_update[z];

      w_update[z] = eps * w_update[z] / sqrt(w_ag[z] + w_update_ag[z] + ADAGRAD_EPS);
      h_update[z] = eps * h_update[z] / sqrt(h_ag[z] + h_update_ag[z] + ADAGRAD_EPS);
    }
  }

  void update_step_size(double factor) {
    eps = eps * factor;
  }

  double current_step_size() {
    return eps;
  }
};
