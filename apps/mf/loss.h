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


/** Calculate l2 regularization for the local part of W and one part of H */
double l2(std::vector<double> &w, std::vector<double> &h, uint h_start_col, uint h_num_cols, const bool add_w_l2=true) {
  double regularization = 0.;
  if (add_w_l2) {
    // w: worker adds reg for local block of w
    auto num_local_rows = static_cast<long>(w.size())/param_len;
    for (long row=0; row!=num_local_rows && row<num_rows; ++row) {
      for (long r=0; r!=mf_rank; ++r) {
        auto z = row*param_len + r;
        regularization += w[z] * w[z];
      }
    }
  }

  // h: worker i adds reg for i'th block of h
  for (long col=h_start_col; col!=h_start_col+h_num_cols && col<num_cols; ++col) {
    for (long r=0; r!=mf_rank; ++r) {
      auto z = col*param_len + r;
      regularization += h[z] * h[z];
    }
  }

  return regularization;
}


/** Calculate non-zero squared loss with L2 regularization for local part of data */
std::pair<double, double> loss_Nzsl_L2(mf::DataPart &data,
                    std::vector<double> &w, std::vector<double> &h,
                    const double lambda, const int pos, const bool add_w_l2) {
  double loss = 0;

  // iterate over local part of the data
  for (unsigned long z=0; z!=data.num_nnz(); ++z) {
    double ip = 0;
    auto x = data.data()[z].x;
    auto i_pos = (data.data()[z].i - data.start_row()) * param_len;
    auto j_pos = (data.data()[z].j) * param_len;

    for (uint r=0; r!=mf_rank; ++r) {
      ip += w[i_pos+r] * h[j_pos+r];
    }

    double diff = x - ip;

    /* LLOG("Loss for " << x << ": " << diff*diff << ", wh: " << ip << ", h[0]: " << h[j_pos+0] << ", w[0]: " << w[i_pos+0]); */
    loss += diff * diff;
  }

  // add L2 regularization
  double l2_loss = 0;
  if (lambda > 0.) {
    l2_loss = lambda * l2(w, h, pos*data.num_cols_per_block(), data.num_cols_per_block(), add_w_l2);
  }

  return std::make_pair(loss, l2_loss);
}


/** Calculate non-zero squared loss with L2 regularization for local part of data,
    without a full copy of the model. Instead, pull factors from the PS on demand.
    This is slow in the distributed setting, but saves memory (e.g., when running on
    a single node).
 */
std::pair<double, double> loss_Nzsl_L2_pull(mf::DataPart &data,
                  WorkerT& kv, std::vector<Key>& local_w_keys,
                  const double lambda, const int pos, const bool add_w_l2) {
  double loss = 0;

  std::vector<Key> w_key (1);
  std::vector<Key> h_key (1);
  std::vector<ValT> w_i (param_len);
  std::vector<ValT> h_j (param_len);

  // iterate over local part of the data
  for (unsigned long z=0; z!=data.num_nnz(); ++z) {
    double ip = 0;
    const mf::DataPoint& dp = data.data()[z];
    w_key[0] = row_key(dp.i);
    h_key[0] = col_key(dp.j);
    kv.Wait(kv.Pull(w_key, &w_i), kv.Pull(h_key, &h_j));

    for (uint r=0; r!=mf_rank; ++r) {
      ip += w_i[r] * h_j[r];
    }

    double diff = dp.x - ip;
    loss += diff * diff;
  }

  // add L2 regularization
  double l2_loss = 0;
  if (lambda > 0.) {
    double regularization = 0.;

    // w: worker adds reg for local block of w
    if (add_w_l2) {
      for (Key key : local_w_keys) {
        w_key[0] = key;
        kv.Wait(kv.Pull(w_key, &w_i));
        for (uint r=0; r!=mf_rank; r++) {
          regularization += w_i[r] * w_i[r];
        }
      }
    }

    // h: worker i adds reg for i'th block of h
    auto cols_per_block = data.num_cols_per_block();
    for (uint j=pos*cols_per_block; j!=(pos+1)*cols_per_block && j<static_cast<size_t>(num_cols); ++j) {
      h_key[0] = col_key(j);
      kv.Wait(kv.Pull(h_key, &h_j));
      for (uint r=0; r!=mf_rank; r++) {
        regularization += h_j[r] * h_j[r];
      }
    }

    l2_loss = lambda * regularization;
  }

  return std::make_pair(loss, l2_loss);
}

