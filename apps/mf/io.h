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


mf::DataPart read_training_data(std::string fname, bool read_partial, int num_workers, int read_part, int& size1, int& size2) {
	std::string line;
  int lineNumber = 0;
  long nnz;

  ifstream in(fname.c_str());
  if(!in.is_open()) {
    cout << "Cannot open file " << fname << endl;
    return mf::DataPart();
  }

  // check file format
  if(!getline(in, line)) {
    cout << "Unexpected EOF in file " << fname << endl;
    return mf::DataPart();
  }
  lineNumber++;
	if (!boost::trim_right_copy(line).compare("%%MatrixMarket matrix coordinate real general") == 0) {
    cout << "Wrong matrix-market banner in file " << fname << endl;
    return mf::DataPart();
  }

	// skip comment lines
	while (getline(in, line) && line.at(0)=='%') {
		lineNumber++;
	}
	lineNumber++;
	if (line.at(0)=='%') {
      cout << "Unexpected EOF in file " << fname << endl;
      return mf::DataPart();
  }

  // read dimension
  if (sscanf(line.c_str(), "%d %d %ld[^\n]", &size1, &size2, &nnz) < 3) {
    cout << "(3) Invalid matrix dimensions in file " << fname <<  ": " <<  line << endl;
    return mf::DataPart();
  }

  // Each worker reads only one block of rows
  int read_min = 0;
  int read_max = std::numeric_limits<int>::max();
  int part_size =  round(ceil(1.0*size1/num_workers));
  if(read_partial) {
    read_min = part_size * read_part;
    read_max = part_size * (read_part+1);
  }

  // init matrices (one for each h-block)
  mf::DataPart data (read_max-read_min, size2, read_min, num_workers, part_size);

  // read matrix
  int i = 0, j = 0;
  double x = 0;
  for(int p=0; p<nnz; p++) {
    // if(p % 1000 == 0)
    // cout << "Line " << p << "/" << nnz << endl;
    if(!getline(in, line)) {
      cout << "Unexpected EOF in file " << fname << endl;
      return mf::DataPart();
    }
    lineNumber++;
    sscanf(line.c_str(), "%d %d %lg\n", &i, &j, &x);

    // read only the rows for this worker
    if(read_partial && i > read_min && i <= read_max) { // read w_min <= i < read_max (but with index starting at 1 in file)
      // append to the matrix for the corresponding block
      data.addDataPoint(i-1, j-1, x);
    } else {
      // otherwise, still increase nnz count of column j
      data.addColNnz(j-1);
    }
  }

	// check that the rest of the file is empty
	while (getline(in, line)) {
		lineNumber++;
		if (!boost::trim_left_copy(std::string(line)).empty()) {
			cout << "Unexpected input at at line " << lineNumber << " of " << fname << ": " << line << endl;
      return mf::DataPart();
		}
	}

  data.freeze();
  LLOG("Data: " << size1 << "x" << size2 << ". Blocks: " << data.num_rows_per_block() << "x" << data.num_cols_per_block() << ". Worker " << read_part << " has rows [" << read_min << "," << read_max << ") (" << data.num_nnz() << " nnz)" );
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



