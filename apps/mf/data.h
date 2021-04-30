#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <random>
#include <Eigen/Core>


using namespace std;

typedef Eigen::SparseMatrix<double,Eigen::RowMajor> SpMatrixRM;

typedef double ValT;
typedef unsigned long SizeT;


namespace mf {

  /** Data point for matrix factorization */
  struct DataPoint {
  DataPoint(SizeT ii, SizeT jj, ValT xx): i{ii}, j{jj}, x{xx} {}
    SizeT i;
    SizeT j;
    ValT x;
  };

  /** Column-major sort order (for sorting data points) */
  bool columnMajor(const DataPoint &a, const DataPoint &b) {
    if (a.j < b.j) return true;
    if (b.j < a.j) return false;

    if (a.i < b.i) return true;
    if (b.i < a.i) return false;

    return false;
  }

  /** Row-major sort order (for sorting data points) */
  bool rowMajor(const DataPoint &a, const DataPoint &b) {
    if (a.i < b.i) return true;
    if (b.i < a.i) return false;

    if (a.j < b.j) return true;
    if (b.j < a.j) return false;

    return false;
  }

  /** Part of the training data. Each worker thread holds one such part. */
  class DataPart {
  public:
    DataPart(SizeT num_rows, SizeT num_cols, const long nnz, SizeT start_row, uint num_workers,
             const uint num_rows_per_block, const bool use_dsgd):
      _num_rows{num_rows}, _num_cols{num_cols}, _total_nnz{nnz}, _start_row{start_row},
      _num_rows_per_block{num_rows_per_block}, _use_dsgd{use_dsgd} {
      _col_nnz = vector<SizeT> (num_cols, 0);
      _row_nnz = vector<SizeT> (num_rows, 0);
      _block_starts = vector<SizeT> ((use_dsgd ? num_workers+1 : num_cols+1), 0);
      _num_cols_per_block = round(ceil(1.0*num_cols/num_workers));
    }

    DataPart(): _total_nnz{0}, _use_dsgd{false} {}

    // add a data point to the data structure
    void addDataPoint(SizeT i, SizeT j, ValT x) {
      _data.push_back(DataPoint(i, j, x));
      ++(_row_nnz[i-_start_row]);
      ++(_col_nnz[j]);
    }

    // add external column count (so that we can use global column count in training)
    void addColNnz(unsigned long j) { ++(_col_nnz[j]); }

    inline const uint row_nnz(int i) { return _row_nnz[i-_start_row]; }
    inline const uint col_nnz(int j) { return _col_nnz[j]; }
    inline const SizeT num_nnz() { return _data.size(); }
    inline const SizeT num_cols() { return _num_cols; }
    inline const SizeT num_rows() { return _num_rows; }
    inline const long total_nnz() { return _total_nnz; }

    inline const SizeT num_cols_per_block() { return _num_cols_per_block; }
    inline const SizeT num_rows_per_block() { return _num_rows_per_block; }
    inline const std::vector<DataPoint>& data() { return _data; }
    inline const std::vector<SizeT>& permutation() { return _permutation; }
    inline const std::vector<SizeT>& block_starts() { return _block_starts; }
    inline const SizeT start_row() { return _start_row; }

    inline const SizeT block_start(uint block) { return _block_starts[block]; }
    inline const SizeT block_end(uint block) { return _block_starts[block+1]; }
    inline const SizeT block_size(uint block) { return block_end(block) - block_start(block); }
    inline const SizeT block_has_nnz(uint block) { return block_start(block) != block_end(block); }

    bool empty() { return _num_rows == 0 || _num_cols == 0; }

    // prepare data structure for training
    void freeze() {
      if (num_nnz() == 0) {
        ALOG("Warning: a worker has 0 data points");
        return;
      };

      // sort by columns, then rows
      std::sort(_data.begin(), _data.end(), columnMajor);

      // Identify column starting points and store them in block_starts
      if (_use_dsgd) {
        _block_starts[0] = 0;
        uint next_block = 1;
        uint z = 0;
        while (z!=num_nnz()) {
          if (_data[z].j >= next_block * _num_cols_per_block) {
            _block_starts[next_block] = z;
            ++next_block;
          } else {
            ++z;
          }
        }
        while (next_block != _block_starts.size()) {
          _block_starts[next_block] = z;
          ++next_block;
        }
      } else {// if we use columnwise SGD, one block corresponds to the data points of one column
        size_t current_j = 0;
        uint z = 0;
        while (z!=num_nnz()) {
          if (_data[z].j >= current_j) {
            _block_starts[current_j] = z;
            ++current_j;
          } else {
            ++z;
          }
        }
        while (current_j != _block_starts.size()) {
          _block_starts[current_j] = z;
          ++current_j;
        }
      }

      // fill permute vector
      _permutation.resize(num_nnz());
      std::iota(_permutation.begin(), _permutation.end(), 0);

      // (temporary)
      // sort blocks by row
      if (_use_dsgd) {
        for (uint b=0; b!=_block_starts.size()-1; ++b) {
          if (block_has_nnz(b)) std::sort(_data.begin()+block_start(b), _data.begin()+block_end(b), rowMajor);
        }
      }
    }

    // permute one block of the data structure
    void permuteBlock(uint block) {
      if (block_has_nnz(block)) std::random_shuffle(_permutation.begin() + block_start(block), _permutation.begin() + block_end(block));
    }

    // permute all data points
    void permuteData() {
      std::random_shuffle(_permutation.begin(), _permutation.end());
    }

  private:
    std::vector<DataPoint> _data;
    std::vector<SizeT> _col_nnz;
    std::vector<SizeT> _row_nnz;
    std::vector<SizeT> _permutation;
    std::vector<SizeT> _block_starts;
    SizeT _num_rows=0;
    SizeT _num_cols=0;
    const long _total_nnz;
    SizeT _start_row;
    SizeT _num_cols_per_block;
    SizeT _num_rows_per_block;
    const bool _use_dsgd;
  };

  /** Initializes a matrix with a sequential block schedule **/
  Eigen::MatrixXi initBlockSchedule(const int num_workers) {
    Eigen::MatrixXi schedule (num_workers, num_workers);

    for (int subepoch=0; subepoch<num_workers; ++subepoch) {
      for (int id=0; id<num_workers; ++id) {
        schedule(subepoch,id) = (subepoch + id) % num_workers;
      }
    }

    return schedule;
  }

  /** Creates a new WOR block schedule from an existing one by permuting columns and rows */
  template<typename T>
  void newWorSchedule(Eigen::MatrixXi& schedule, const int epoch, T& rng) {
    // create empty permutation matrix
    Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> perm(schedule.rows());
    perm.setIdentity();

    rng.seed(epoch); // make sure all nodes create the same schedule at each epoch

    // permute columns
    std::shuffle(perm.indices().data(), perm.indices().data()+perm.indices().size(), rng);
    schedule = schedule * perm; // permute columns

    // permute rows
    std::shuffle(perm.indices().data(), perm.indices().data()+perm.indices().size(), rng);
    schedule = perm * schedule; // permute rows
  }
}

// output a data point
std::ostream& operator<<(std::ostream& os, const mf::DataPoint& dp) {
  std::stringstream ss;
  ss.setf(std::ios::fixed);
  ss.precision(3);
  ss << "[" << dp.i << "," << dp.j << ": " << dp.x << "]";
  os << ss.str();
  return os;
}

// output a data part
std::ostream& operator<<(std::ostream& os, mf::DataPart& data) {
  std::stringstream ss1;
  std::stringstream ss2;
  std::stringstream ss3; ss3.setf(std::ios::fixed); ss3.precision(2);
  std::stringstream ss4;
  std::stringstream ss5;
  std::stringstream ss6;
  std::stringstream ss7;

  ss1 << "i:\t";
  ss2 << "j:\t";
  ss3 << "x:\t";
  ss4 << "p:\t";
  ss5 << "blocks: ";
  ss6 << "col_nnz: ";
  ss7 << "row_nnz: ";

  for(SizeT z=0; z!=data.num_nnz(); ++z) {
    ss1 << data.data()[z].i << "\t";
    ss2 << data.data()[z].j << "\t";
    ss3 << data.data()[z].x << "\t";
    ss4 << data.permutation()[z] << "\t";
  }

  for(SizeT k=0; k!=data.block_starts().size(); ++k) {
    ss5 << data.block_starts()[k] << " ";
  }
  for(SizeT k=0; k!=data.num_cols(); ++k) {
    ss6 << data.col_nnz(k) << " ";
  }
  for(SizeT k=0; k!=data.num_rows(); ++k) {
    ss7 << data.row_nnz(k+data.start_row()) << " ";
  }

  os << ss1.str() << "\n" << ss2.str() << "\n" << ss3.str() << "\n" << ss4.str() << "\n" <<
    ss5.str() << "\n" << ss6.str() << "\n" << ss7.str() << "\n";
  return os;
}


