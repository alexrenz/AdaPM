#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>
#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/timer/timer.hpp>
#include <boost/utility/string_ref.hpp>
#include <boost/algorithm/string.hpp>


using namespace std;
using namespace util;

enum class Alg {dsgd, columnwise, plain_sgd};
std::ostream& operator<<(std::ostream &o, const Alg& rsm) {
  switch(rsm) {
  case Alg::dsgd: return o << "dsgd";
  case Alg::columnwise: return o << "columnwise";
  case Alg::plain_sgd: return o << "plain_sgd";
  default: return o << "unkown";
  }
}
std::istream& operator>>(std::istream& in, Alg& rsm) {
  std::string token; in >> token;
  if (token == "dsgd") rsm = Alg::dsgd;
  else if (token == "columnwise") rsm = Alg::columnwise;
  else if (token == "plain_sgd") rsm = Alg::plain_sgd;
  else { CHECK(false) << "Fatal! Unknown MF algorithm " << token; }
  return in;
}


// create a binary file from a mmc matrix
// format: [int size1] [int size 2] [long nnz]  [int dp1.i] [int dp1.j] [double dp1.x]  [int dp2.i] ...
bool create_binary_from_mmc (std::string fname, std::string fname_binary) {
  ifstream in(fname.c_str());
  int size1;
  int size2;
  long nnz;
	std::string line;
  int lineNumber = 0;


  if(!in.is_open()) {
    cout << "Cannot open file " << fname << endl;
    return false;
  }

  // check file format
  if(!getline(in, line)) {
    cout << "Unexpected EOF in file " << fname << endl;
    return false;
  }
  lineNumber++;
	if (!boost::trim_right_copy(line).compare("%%MatrixMarket matrix coordinate real general") == 0) {
    cout << "Wrong matrix-market banner in file " << fname << endl;
    return false;
  }

	// skip comment lines
	while (getline(in, line) && line.at(0)=='%') {
		lineNumber++;
	}
	lineNumber++;
	if (line.at(0)=='%') {
    cout << "Unexpected EOF in file " << fname << endl;
    return false;
  }

  // read dimension
  if (sscanf(line.c_str(), "%d %d %ld[^\n]", &size1, &size2, &nnz) < 3) {
    cout << "(3) Invalid matrix dimensions in file " << fname <<  ": " <<  line << endl;
    return false;
  }


  // output file
  ofstream out(fname_binary, ios::out | ios::binary);
  std::cout << "Read text file of " << size1 << " x " << size2 << " matrix with " << nnz << " nnz" << endl;
  // wf.write((char *) &size1, sizeof(Student));
  out.write(reinterpret_cast<const char *>(&size1), sizeof(size1));
  out.write(reinterpret_cast<const char *>(&size2), sizeof(size2));
  out.write(reinterpret_cast<const char *>(&nnz), sizeof(nnz));

  int i = 0, j = 0;
  double x = 0;
  for(int p=0; p<nnz; p++) {
    // if(p % 1000 == 0)
    // cout << "Line " << p << "/" << nnz << endl;
    if(!getline(in, line)) {
      cout << "Unexpected EOF in file " << fname << endl;
      return false;
    }
    lineNumber++;
    sscanf(line.c_str(), "%d %d %lg\n", &i, &j, &x);

    // write to binary file
    out.write(reinterpret_cast<const char *>(&i), sizeof(i));
    out.write(reinterpret_cast<const char *>(&j), sizeof(j));
    out.write(reinterpret_cast<const char *>(&x), sizeof(x));
  }
  out.close();
  return true;
}

// data point that is read in from a binary file
struct ReadDp {
  int i;
  int j;
  double x;
};

bool compareByColThenRow(const ReadDp &a, const ReadDp &b) {
  return (a.j == b.j ? a.i < b.i : a.j < b.j);
}


// read matrix for matrix factorization. creates a binary version of the passed mmc file if it doesn't exist yet
template<typename WorkerT>
mf::DataPart read_sparse_matrix_part(std::string fname, bool read_partial, int num_workers,
                                     int worker_id, int num_servers, int my_rank,
                                     const size_t num_rows, const size_t num_cols,
                                     const int customer_id, const Alg algorithm, WorkerT& kv) {
  long nnz;

  std::string fname_binary = fname + ".bin";

  ifstream test_binary(fname_binary.c_str());
  if (!test_binary.good()) { // no binary file found. try to create one
    if (customer_id == 1) {
      // create a binary dump
      std::cout << "Worker " << worker_id << ": no binary file for dataset " << fname << " yet. Creating one... " << endl;
      create_binary_from_mmc(fname, fname_binary);
      std::cout << "Worker " << worker_id << ": done creating binary file (" << fname_binary << ")" << endl;
    }
  }
  test_binary.close();
  kv.Barrier();

  // open binary file
  ifstream in(fname_binary, ios::in | ios::binary);
  if(!in) {
    cout << "Cannot open binary file for reading: " << fname_binary << endl;
    return mf::DataPart();
  }

  // read dimensions
  int size1, size2;
  in.read((char *) &size1, sizeof(size1));
  in.read((char *) &size2, sizeof(size2));
  in.read((char *) &nnz, sizeof(nnz));
  if(static_cast<long>(num_rows) != size1 || static_cast<long>(num_cols) != size2) {
    ADLOG("Dimensions of the read matrix (" << size1 << "x" << size2 << ") do not match the specified dimensions (" << num_rows << "x" << num_cols << ")");
    abort();
  }

  // Each worker reads only one block of rows
  int min_row = 0;
  int max_row = std::numeric_limits<int>::max();
  int rows_per_worker = round(ceil(1.0*size1/num_workers));
  if(read_partial) {
    if (algorithm != Alg::columnwise) { // usually, each workers holds one partition of rows
      min_row = rows_per_worker * worker_id;
      max_row = rows_per_worker * (worker_id+1);
    } else { // in columnwise, data points are not partitioned by row exclusively (see below)
      rows_per_worker = round(ceil(1.0*size1/num_servers));
      min_row = rows_per_worker * my_rank;
      max_row = rows_per_worker * (my_rank+1);
    }
  }

  // init matrices (one for each h-block)
  mf::DataPart data (max_row-min_row, size2, nnz, min_row, num_workers, rows_per_worker, algorithm == Alg::dsgd);

  // read matrix
  ReadDp dp;

  // For the columnwise algorithm, we partition data points to nodes _by row_,
  // and then we partition data points to workers roughly _by column_. "Roughly"
  // because we assign data points to workers evenly. To achieve this, we first
  // collect all data points for this node, sort the data points by column, and
  // then assign the data points to workers evenly. That is not the most
  // efficient way, but this works for now (and is not timed).
  std::vector<ReadDp> datapoints_node {};
  if (algorithm == Alg::columnwise) {
    datapoints_node.reserve(nnz / num_servers);
  }

  for(int p=0; p<nnz; p++) {
    // read one data point from the binary file
    in.read((char *) &dp, sizeof(dp));

    // read only the rows for this worker
    if(read_partial && dp.i > min_row && dp.i <= max_row) { // read w_min <= i < max_row (but with index starting at 1 in file)
      if (algorithm == Alg::columnwise) {
        // collect data points first (so we can sort them by column)
        datapoints_node.push_back(dp);
      } else {
        // append to the matrix for the corresponding block
        data.addDataPoint(dp.i-1, dp.j-1, dp.x);
      }
    } else {
      // otherwise, still increase nnz count of column j
      data.addColNnz(dp.j-1);
    }
  }

  // columnwise: assign data points to workers evenly, by column
  if (algorithm == Alg::columnwise) {
    std::sort(datapoints_node.begin(),
              datapoints_node.end(), compareByColThenRow);

    int num_local_workers = num_workers/num_servers;
    long dps_per_worker = round(ceil(1.0*datapoints_node.size()/num_local_workers));
    size_t min_dp = dps_per_worker * (customer_id-1);
    size_t max_dp = dps_per_worker * (customer_id-1+1);

    for (size_t z=0; z!=datapoints_node.size(); ++z) {
      auto& dp = datapoints_node[z];
      if (z >= min_dp && z < max_dp) {
        data.addDataPoint(dp.i-1, dp.j-1, dp.x);
      } else {
        data.addRowNnz(dp.i-1);
        data.addColNnz(dp.j-1);
      }
    }
  }

  // check that we are at the end of the binary file
  auto pos = in.tellg();
  in.seekg(0, ios::end);
  if (pos != in.tellg()) {
    ALOG("Something is wrong with the binary file " << fname_binary << " in worker " << worker_id << ": it is longer than expected.");
    abort();
  }

  in.close();
  data.freeze();
  std::stringstream sc;
  sc << data.num_nnz() << " of " << datapoints_node.size() << " data points of";
  LLOG("Data (" << fname.substr(fname.find_last_of("/\\") + 1) << "): " << size1 << "x" << size2 << ". Blocks: " << data.num_rows_per_block() << "x" << data.num_cols_per_block() << ". Worker " << worker_id << " has " << (algorithm == Alg::columnwise ? sc.str() : "") << " rows [" << min_row << "," << max_row << ") (" << data.num_nnz() << " nnz)" );
  return data;
}


template<class M>
M read_dense_matrix(std::string fname, bool row_major=true, int row_parts=1, int row_read_part=0, int col_parts=1, int col_read_part=0) {
	std::string line;
  int lineNumber = 0;
  long rows, cols;

  ifstream in(fname.c_str());
  if(!in.is_open()) {
    cout << "Cannot open file " << fname << endl;
    return M {};
  }

  // check file format
  if(!getline(in, line)) {
    cout << "Unexpected EOF in file " << fname << endl;
    return M {};
  }
  ++lineNumber;
  if (!boost::trim_right_copy(line).compare("%%MatrixMarket matrix array real general") == 0) {
    cout << "Wrong matrix-market banner in file " << fname << endl;
    return M {};
  }

	// skip comment lines
	while (getline(in, line) && line.at(0)=='%') {
		++lineNumber;
	}
	++lineNumber;
	if (line.at(0)=='%') {
      cout << "Unexpected EOF in file " << fname << endl;
    return M {};
  }

  // read dimension
  if (sscanf(line.c_str(), "%ld %ld[^\n]", &rows, &cols) < 2) {
    cout << "Invalid matrix dimensions in file " << fname <<  ": " <<  line << endl;
    return M {};
  }

  int i_read_min = rows / row_parts * row_read_part;
  int i_read_max = rows / row_parts * (row_read_part+1);
  long local_rows = i_read_max-i_read_min;

  int j_read_min = cols / col_parts * col_read_part;
  int j_read_max = cols / col_parts * (col_read_part+1);
  long local_cols = j_read_max-j_read_min;
  /* stringstream ss; */
  /* ss << "(" << row_read_part << "/" << row_parts << ", " << col_read_part << "/" << col_parts << "): "; */
  /* ss << "Read rows " << i_read_min << " <= i < " << i_read_max << " and columns " << j_read_min << " <= j < " << j_read_max << ", " << (row_major ? "row-major" : "col-major") << "\n"; */
  /* std::cout << ss.str(); */

  // init matrix
  M m (local_rows*local_cols);
  double x = 0;

  // read matrix
  for (long j=0; j<cols; j++) {
    for (long i=0; i<rows; i++) {
			if (!getline(in, line)) {
        cout << "Unexpected EOF in file " << fname << endl;
        return M {};
      }
			lineNumber++;

      sscanf(line.c_str(), "%lg\n", &x);
      if(i >= i_read_min && i < i_read_max && j >= j_read_min && j < j_read_max) {
        if (row_major) {
          m[(i-i_read_min)*local_cols+(j-j_read_min)] = x;
        } else {
          m[(j-j_read_min)*local_rows+(i-i_read_min)] = x;
        }
      }
    }
  }


	// check that the rest of the file is empty
	while (getline(in, line)) {
		lineNumber++;
		if (!boost::trim_left_copy(std::string(line)).empty()) {
			cout << "Unexpected input at at line " << lineNumber << " of " << fname << ": " << line << endl;
      return M {};
		}
	}

  // no freeze for dense matrix
  return m;
}


std::vector<double> read_dense_matrix_part_into_vector(std::string fname, long worker_rank, long block_size, int mf_rank) {
	std::string line;
  int lineNumber = 0;

  long size1, size2;

  std::vector<double> error { };

  ifstream in(fname.c_str());
  if(!in.is_open()) {
    cout << "Cannot open file " << fname << endl;
    return error;
  }

  // check file format
  if(!getline(in, line)) {
    cout << "Unexpected EOF in file " << fname << endl;
    return error;
  }
  lineNumber++;
  if (!boost::trim_right_copy(line).compare("%%MatrixMarket matrix array real general") == 0) {
    cout << "Wrong matrix-market banner in file " << fname << endl;
    return error;
  }

	// skip comment lines
	while (getline(in, line) && line.at(0)=='%') {
		lineNumber++;
	}
	lineNumber++;
	if (line.at(0)=='%') {
      cout << "Unexpected EOF in file " << fname << endl;
    return error;
  }


  // read dimension
  if (sscanf(line.c_str(), "%ld %ld[^\n]", &size1, &size2) < 2) {
    cout << "(2) Invalid matrix dimensions in file " << fname <<  ": " <<  line << endl;
    return error;
  }


  // init matrix
  std::vector<double> v (block_size*mf_rank);
  double x = 0;


  long first_row = worker_rank * block_size;
  long last_row = first_row + block_size;

  // read matrix
  for (long j=0; j<size2; j++) {
    for (long i=0; i<size1; i++) {
			if (!getline(in, line)) {
        cout << "Unexpected EOF in file " << fname << endl;
        return error;
      }
			lineNumber++;

      sscanf(line.c_str(), "%lg\n", &x);

      if(i >= first_row && i < last_row) {
        v[(i-first_row)*mf_rank+j] = x;
      }
    }
  }


	// check that the rest of the file is empty
	while (getline(in, line)) {
		lineNumber++;
		if (!boost::trim_left_copy(std::string(line)).empty()) {
			cout << "Unexpected input at at line " << lineNumber << " of " << fname << ": " << line << endl;
      return error;
		}
	}

  return v;
}



