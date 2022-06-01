#include "utils.h"
#include "ps/ps.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>
#include <thread>
#include <numeric>
#include <boost/program_options.hpp>
#include <limits>
#include <sstream>
#include <string>
#include <iostream>
#include <unistd.h>
#include <bitset>
#include <random>
#include <iomanip>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <tuple>
#include <cassert>
#include <cstring>
#include <regex>


using namespace ps;
using namespace std;

typedef double ValT;
typedef DefaultColoServerHandle<ValT> HandleT;
typedef ColoKVServer<ValT, HandleT> ServerT;
typedef ColoKVWorker<ValT, HandleT> WorkerT;

enum class Alg { ComplEx, RESCAL };


// Model and algorithm parameters
string  alg;
Alg     algorithm;
string  dataset;
uint    embed_dim;
uint    rel_dim;
double  eta;
double  gamma_entity;
double  gamma_relation;
double  dropout_entity;
double  dropout_relation;
int     neg_ratio;
uint    num_epochs;
uint    num_threads;
int     eval_freq;
long    ne; // number of entities
long    nr; // number of relations
string  model_path;
int save_every_nth_epoch;
bool write_end_checkpoint;

// Evaluation
size_t  eval_truncate_va;
size_t  eval_truncate_tr;
bool run_initial_evaluation;
unsigned model_seed;


// System parameters
bool async_push;
bool signal_initial_relations_intent;
uint signal_intent_ahead;
bool read_partitioned_dataset;
std::string init_parameters;
bool enforce_random_keys;
bool enforce_full_replication;
long max_N_per_thread;
long max_runtime;

uint num_workers = -1;
uint num_keys = -1;
uint entity_vector_length;
uint relation_vector_length;
string tr_file;
int num_serv; // number of servers

// random assignment of keys (if enabled)
std::vector<Key> key_assignment;

Key loss_key;
Key eval_key;

// positions in eval vector (for distributed evaluation)
const int _MRR_S     = 0;
const int _MRR_R     = 1;
const int _MRR_O     = 2;
const int _MRR_S_RAW = 3;
const int _MRR_O_RAW = 4;
const int _MR_S      = 5;
const int _MR_R      = 6;
const int _MR_O      = 7;
const int _MR_S_RAW  = 8;
const int _MR_O_RAW  = 9;
const int _HITS01_S  = 10;
const int _HITS01_R  = 11;
const int _HITS01_O  = 12;
const int _HITS03_S  = 13;
const int _HITS03_R  = 14;
const int _HITS03_O  = 15;
const int _HITS10_S  = 16;
const int _HITS10_R  = 17;
const int _HITS10_O  = 18;


static default_random_engine GLOBAL_GENERATOR;
static uniform_real_distribution<double> UNIFORM(0, 1);

typedef tuple<int, int, int> triplet;


inline Key entity_key  (const int e) {
  return enforce_random_keys ? key_assignment[e] : e;
}

inline Key relation_key (const int r) {
  return enforce_random_keys ? key_assignment[ne+r] : ne+r;
}


// Negative sampling distribution
std::mt19937 negs_gen;
std::uniform_int_distribution<int> negs_dist;

// Draw a negative sample
inline Key DrawEntity() {
  return entity_key(negs_dist(negs_gen));
}


std::ostream& operator<<(std::ostream& os, const triplet t) {
  std::stringstream ss;
  ss << "<" << get<0>(t) << "," << get<1>(t) << "," << get<2>(t) << ">";
  os << ss.str();
  return os;
}

// Process-level data structures
vector<triplet> sros_tr;
vector<triplet> sros_va;
vector<triplet> sros_te;
vector<triplet> sros_al;


vector<triplet> create_sros(const string& fname) {

    ifstream ifs(fname, ios::in);

    string line;
    int s, r, o;
    vector<triplet> sros;

    assert(!ifs.fail());

    while (getline(ifs, line)) {
        stringstream ss(line);
        ss >> s >> r >> o;
        sros.push_back( make_tuple(s, r, o) );
    }
    ifs.close();

    return sros;
}

vector<vector<double>> uniform_matrix(int m, int n, double l, double h) {
    vector<vector<double>> matrix;
    matrix.resize(m);
    for (int i = 0; i < m; i++)
        matrix[i].resize(n);

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            matrix[i][j] = (h-l)*UNIFORM(GLOBAL_GENERATOR) + l;

    return matrix;
}

vector<vector<double>> const_matrix(int m, int n, double c) {
    vector<vector<double>> matrix;
    matrix.resize(m);
    for (int i = 0; i < m; i++)
        matrix[i].resize(n);

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            matrix[i][j] = c;

    return matrix;
}

vector<int> range(int n, int start=0) {  // 0 ... n-1
  vector<int> v;
  v.reserve(n);
  for (int i = start; i < n+start; i++)
    v.push_back(i);
  return v;
}

// pull the entire model from the PS to local vectors
void pull_full_model(std::vector<ValT>& E, std::vector<ValT>& R, WorkerT& kv) {
  util::Stopwatch sw_pull;
  sw_pull.start();

  // construct data structures for pull
  vector<Key> R_keys (nr);
  for(unsigned int r=0; r!=nr; ++r) {
    R_keys[r] = relation_key(r);
  }
  R.resize(nr * relation_vector_length);
  vector<Key> E_keys (ne);
  for(unsigned int e=0; e!=ne; ++e) {
    E_keys[e] = entity_key(e);
  }
  E.resize (ne * entity_vector_length);

  // pull
  kv.WaitSync();
  kv.Wait(kv.Pull(E_keys, &E), kv.Pull(R_keys, &R));

  sw_pull.stop();
  ADLOG("Model pulled (" << sw_pull << ")");
}



double sigmoid(double x, double cutoff=30) {
    if (x > +cutoff) return 1.;
    if (x < -cutoff) return 0.;
    return 1./(1.+exp(-x));
}

class SROBucket {
    unordered_set<int64_t> __sros;
    unordered_map<int64_t, vector<int>> __sr2o;
    unordered_map<int64_t, vector<int>> __or2s;
    size_t ne_bits;
    size_t nr_bits;
    long duplicates = 0;

    int64_t hash(int a, int b, int c) const {
        int64_t x = a;
        x = (x << nr_bits) + b;
        return (x << ne_bits) + c;
    }

    int64_t hash(int a, int b) const {
        int64_t x = a;
        return (x << 32) + b;
    }

public:
    SROBucket(const vector<triplet>& sros) {
      // make sure hash(int,int,int) does not overflow despite int64
      ne_bits = ceil(log2(ne*2));
      nr_bits = ceil(log2(nr*2));
      auto hash_needs = 2*ne_bits + nr_bits;
      assert(hash_needs < sizeof(int64_t)*8);

        for (auto sro : sros) {
            int s = get<0>(sro);
            int r = get<1>(sro);
            int o = get<2>(sro);

            // add identical data points to indexes only once
            if (contains(s,r,o)) {
              ++duplicates;
              continue;
            }

            int64_t __sro = hash(s, r, o);
            __sros.insert(__sro);

            int64_t __sr = hash(s, r);
            if (__sr2o.find(__sr) == __sr2o.end())
                __sr2o[__sr] = vector<int>();
            __sr2o[__sr].push_back(o);

            int64_t __or = hash(o, r);
            if (__or2s.find(__or) == __or2s.end())
                __or2s[__or] = vector<int>();
            __or2s[__or].push_back(s);
        }

        ADLOG("SROBucket: skipped " << duplicates << " duplicate data points");
    }

    bool contains(int a, int b, int c) const {
        return __sros.find( hash(a, b, c) ) != __sros.end();
    }

    vector<int> sr2o(int s, int r) const {
        return __sr2o.at(hash(s,r));
    }

    vector<int> or2s(int o, int r) const {
        return __or2s.at(hash(o,r));
    }
};


class Model {

protected:
    double eta;
    const double init_e = 1e-6;

    vector<vector<double>> E;
    vector<vector<double>> R;
    vector<vector<double>> E_g;
    vector<vector<double>> R_g;

public:

    Model(double eta) {
        this->eta = eta;
    }

  // saves a given model to disk
  void save(const uint epoch, const string& fname, WorkerT& kv, const bool export_for_eval=true, const bool write_checkpoint=false) {
    ADLOG("Save model (" << (export_for_eval ? "eval export" : "") << (write_checkpoint ? ", checkpoint" : "") <<
          ") for epoch " << epoch << " to " << fname << "*.epoch." << std::to_string(epoch) << ".*.bin");
    // pull the model
    std::vector<ValT> E {};
    std::vector<ValT> R {};
    pull_full_model(E, R, kv);

    // prepare output files
    std::ofstream file_E_eval;
    std::ofstream file_E_checkpoint;
    std::ofstream file_E_adagrad;
    std::ofstream file_R_eval;
    std::ofstream file_R_checkpoint;
    std::ofstream file_R_adagrad;
    if (export_for_eval) {
      file_E_eval.open(fname+"export.epoch." + std::to_string(epoch) + ".entities.bin", std::ios::binary);
      file_R_eval.open(fname+"export.epoch." + std::to_string(epoch) + ".relations.bin", std::ios::binary);
      if (!file_E_eval.is_open() || !file_R_eval.is_open()) {
        ALOG("Error opening files for eval export");
        abort();
      }
    }
    if (write_checkpoint) {
      file_E_checkpoint.open(fname+"checkpoint.epoch." + std::to_string(epoch) + ".entities.bin", std::ios::binary);
      file_R_checkpoint.open(fname+"checkpoint.epoch." + std::to_string(epoch) + ".relations.bin", std::ios::binary);
      file_E_adagrad.open(fname+"checkpoint.epoch." + std::to_string(epoch) + ".entities.adagrad.bin", std::ios::binary);
      file_R_adagrad.open(fname+"checkpoint.epoch." + std::to_string(epoch) + ".relations.adagrad.bin", std::ios::binary);
      if (!file_E_checkpoint.is_open() || !file_R_checkpoint.is_open() || !file_E_adagrad.is_open() || !file_R_adagrad.is_open()) {
        ALOG("Error opening files for writing checkpoint");
        abort();
      }
    }

    // write entity embeddings
    for (long e = 0; e != ne; e++) {
      if (export_for_eval) {
        for(size_t i = 0; i!=embed_dim; ++i) {
          float as_float = static_cast<float>(E[e * entity_vector_length + i]);
          file_E_eval.write((char*) &as_float, sizeof(float)); // embedding
        }
      }
      if (write_checkpoint) {
        ValT* pos = &(E[e * entity_vector_length]);
        file_E_checkpoint.write((char*) pos, sizeof(ValT) * embed_dim); // embedding
        file_E_adagrad.write((char*) pos+embed_dim, sizeof(ValT) * embed_dim); // adagrad
      }
    }

    // write relation embeddings
    for (long r = 0; r != nr; r++) {
      if (export_for_eval) {
        for(size_t i = 0; i!=rel_dim; ++i) {
          float as_float = static_cast<float>(R[r * relation_vector_length + i]);
          file_R_eval.write((char*) &as_float, sizeof(float)); // embedding
        }
      }
      if (write_checkpoint) {
        ValT* pos = &(R[r * relation_vector_length]);
        file_R_checkpoint.write((char*) pos, sizeof(ValT) * rel_dim); // embedding
        file_R_adagrad.write((char*) pos+rel_dim, sizeof(ValT) * rel_dim); // adagrad
      }
    }

    if (export_for_eval) {
      file_E_eval.close();
      file_R_eval.close();
    }
    if (write_checkpoint) {
      file_E_checkpoint.close();
      file_R_checkpoint.close();
      file_E_adagrad.close();
      file_R_adagrad.close();
    }
  }

  // randomly zeroes some elements of the given vector and scales
  // other elements by 1 / (1-p)
  void do_dropout(double* v, const size_t len, const double p) {
    for (unsigned i = 0; i != len; i++) {
      if (UNIFORM(GLOBAL_GENERATOR) < p) {
        v[i] = 0;
      } else {
        v[i] *= 1 / (1-p);
      }
    }
  }

    void adagrad_update(double* E_s, double* R_r, double* E_o,
                      double* d_s, double* d_r, double* d_o) {


      double* Eg_s = E_s + embed_dim;
      double* Rg_r = R_r + rel_dim;
      double* Eg_o = E_o + embed_dim;

      double* dg_s = d_s + embed_dim;
      double* dg_r = d_r + rel_dim;
      double* dg_o = d_o + embed_dim;

      for (unsigned i = 0; i < embed_dim; i++) dg_s[i] = d_s[i] * d_s[i];
      for (unsigned i = 0; i < rel_dim;   i++) dg_r[i] = d_r[i] * d_r[i];
      for (unsigned i = 0; i < embed_dim; i++) dg_o[i] = d_o[i] * d_o[i];


      for (unsigned i = 0; i < embed_dim; i++) d_s[i] = - eta * d_s[i] / sqrt(Eg_s[i] + dg_s[i]);
      for (unsigned i = 0; i < rel_dim;   i++) d_r[i] = - eta * d_r[i] / sqrt(Rg_r[i] + dg_r[i]);
      for (unsigned i = 0; i < embed_dim; i++) d_o[i] = - eta * d_o[i] / sqrt(Eg_o[i] + dg_o[i]);
  }

  void train(int s, int r, int o, bool is_positive, WorkerT& kv, double& bce_loss, double& reg_loss) {
      // Allocate memory and get embeddings from the server
      vector<double> embed_s   (entity_vector_length);
      vector<double> embed_r   (relation_vector_length);
      vector<double> embed_o   (entity_vector_length);
      vector<double> update_s  (embed_s.size());
      vector<double> update_r  (embed_r.size());
      vector<double> update_o  (embed_o.size());

      vector<Key> key_s (1);
      vector<Key> key_r {relation_key(r)};
      vector<Key> key_o (1);
      int ts_s, ts_r, ts_o;

      // negative sample for the subject?
      if (s < 0) { // negative sample
        ts_s = kv.PullSample(-s, key_s, embed_s);
      } else { // true subject
        key_s[0] = entity_key(s);
        ts_s = kv.Pull(key_s, &embed_s);
      }

      // negative sample for the object?
      if (o < 0) { // negative sample
        ts_o = kv.PullSample(-o, key_o, embed_o);
      } else { // true object
        key_o[0] = entity_key(o);
        ts_o = kv.Pull(key_o, &embed_o);
      }

      ts_r = kv.Pull(key_r, &embed_r);
      kv.Wait(ts_s, ts_r, ts_o);

      double* E_s = embed_s.data();
      double* R_r = embed_r.data();
      double* E_o = embed_o.data();
      double* d_s = update_s.data();
      double* d_r = update_r.data();
      double* d_o = update_o.data();

      // dropout
      if (dropout_entity != 0) {
        do_dropout(E_s, embed_dim, dropout_entity);
        do_dropout(E_o, embed_dim, dropout_entity);
      }
      if (dropout_relation != 0) {
        do_dropout(R_r, rel_dim, dropout_relation);
      }

      double offset = is_positive ? 1 : 0;
      double triple_score = score(E_s, R_r, E_o);
      double d_loss = sigmoid(triple_score) - offset;

      // record training loss
      double y = is_positive ? 1 : -1;
      double loss = -log(sigmoid(y*triple_score, 1e9));
      bce_loss += loss;

      // score_grad(s, r, o, d_s, d_r, d_o);
      score_grad(E_s, R_r, E_o, d_s, d_r, d_o);

      for (unsigned i = 0; i < embed_dim; i++) d_s[i] *= d_loss;
      for (unsigned i = 0; i < rel_dim;   i++) d_r[i] *= d_loss;
      for (unsigned i = 0; i < embed_dim; i++) d_o[i] *= d_loss;

      // regularization (only on positives)
      if (is_positive) {
        double reg_loss_s = 0;
        double reg_loss_r = 0;
        double reg_loss_o = 0;

        for (unsigned i = 0; i < embed_dim; i++) {
          d_s[i] += gamma_entity * E_s[i];
          reg_loss_s += E_s[i]*E_s[i];
        }
        for (unsigned i = 0; i < rel_dim;   i++) {
          d_r[i] += gamma_relation * R_r[i];
          reg_loss_r += R_r[i]*R_r[i];
        }
        for (unsigned i = 0; i < embed_dim; i++) {
          d_o[i] += gamma_entity * E_o[i];
          reg_loss_o += E_o[i]*E_o[i];
        }

        reg_loss += gamma_entity * reg_loss_s + gamma_relation * reg_loss_r + gamma_entity * reg_loss_o;
      }

      adagrad_update(E_s, R_r, E_o, d_s, d_r, d_o);

      // ADLOG("Push. Key: " << key_r << "\nValues: " << update_r);
      if (async_push) {
        kv.Push(key_s, update_s); kv.Push(key_r, update_r); kv.Push(key_o, update_o);
      } else {
        kv.Wait(kv.Push(key_s, update_s), kv.Push(key_r, update_r), kv.Push(key_o, update_o));
      }
  }

  virtual double score(double* E_s, double* R_r, double* E_o) const = 0;

  virtual void score_grad(double* E_s, double* R_r, double* E_o,
                          double* d_s, double* d_r, double* d_o) {};

  virtual vector<double> init_ps(std::function<double()> init_rand) = 0;

};// end Model



class Evaluator {
    long ne;
    long nr;
    const vector<triplet>& sros;
    const SROBucket& sro_bucket;

public:
    Evaluator(long ne, long nr, const vector<triplet>& sros, const SROBucket& sro_bucket) :
      ne(ne), nr(nr), sros(sros), sro_bucket(sro_bucket) {}

  // distributed evaluation: each node processes a part of the data points. The results are then aggregated via the PS
  unordered_map<string, double> evaluate(const Model *model, int truncate, vector<double>& E, vector<double>& R, const int worker_id, WorkerT& kv) {

        double mrr_s = 0.;
        double mrr_r = 0.;
        double mrr_o = 0.;

        double mrr_s_raw = 0.;
        double mrr_o_raw = 0.;

        double mr_s = 0.;
        double mr_r = 0.;
        double mr_o = 0.;

        double mr_s_raw = 0.;
        double mr_o_raw = 0.;

        double hits01_s = 0.;
        double hits01_r = 0.;
        double hits01_o = 0.;

        double hits03_s = 0.;
        double hits03_r = 0.;
        double hits03_o = 0.;

        double hits10_s = 0.;
        double hits10_r = 0.;
        double hits10_o = 0.;


        // calculate total number of data points and number of data points per node
        int N = this->sros.size();
        if (truncate > 0) {
          N = min(N, truncate);
        }
        int N_per_node = ceil(1.0 * N / num_serv);

        // allocate evaluation vector (used to aggregated partial evaluation results on PS)
        std::vector<Key> eval_key_vec { eval_key };
        std::vector<ValT> eval (entity_vector_length);

        // reset the evaluation vector (might contain old values from last eval)
        if (worker_id == 0) {
          kv.Wait(kv.Pull(eval_key_vec, &eval));
          for(size_t i=0; i!=eval.size(); ++i) {
            eval[i] = -eval[i];
          }
          kv.Wait(kv.Push(eval_key_vec, eval));
        }

        // wait for reset to finish
        Postoffice::Get()->Barrier(1, kServerGroup);

        // use all threads to evaluate
#pragma omp parallel for reduction(+: mrr_s, mrr_r, mrr_o, mr_s, mr_r, mr_o, \
                                   hits01_s, hits01_r, hits01_o, hits03_s, hits03_r, hits03_o, hits10_s, hits10_r, hits10_o)
        for (int z = 0; z < N_per_node; z++) {
          auto i = ps::MyRank() * N_per_node + z;
          // last worker might have less data points
          if(i >= N) continue;

          auto ranks = this->rank(model, sros[i], E, R);

          double rank_s = get<0>(ranks);
          double rank_r = get<1>(ranks);
          double rank_o = get<2>(ranks);
          double rank_s_raw = get<3>(ranks);
          double rank_o_raw = get<4>(ranks);


          mrr_s += 1./rank_s;
          mrr_r += 1./rank_r;
          mrr_o += 1./rank_o;
          mrr_s_raw += 1./rank_s_raw;
          mrr_o_raw += 1./rank_o_raw;

          mr_s += rank_s;
          mr_r += rank_r;
          mr_o += rank_o;
          mr_s_raw += rank_s_raw;
          mr_o_raw += rank_o_raw;

          hits01_s += rank_s <= 01;
          hits01_r += rank_r <= 01;
          hits01_o += rank_o <= 01;

          hits03_s += rank_s <= 03;
          hits03_r += rank_r <= 03;
          hits03_o += rank_o <= 03;

          hits10_s += rank_s <= 10;
          hits10_r += rank_r <= 10;
          hits10_o += rank_o <= 10;
        }


        // aggregate partial evaluation results on the PS
        eval[_MRR_S] = mrr_s;
        eval[_MRR_R] = mrr_r;
        eval[_MRR_O] = mrr_o;
        eval[_MRR_S_RAW] = mrr_s_raw;
        eval[_MRR_O_RAW] = mrr_o_raw;

        eval[_MR_S] = mr_s;
        eval[_MR_R] = mr_r;
        eval[_MR_O] = mr_o;
        eval[_MR_S_RAW] = mr_s_raw;
        eval[_MR_O_RAW] = mr_o_raw;

        eval[_HITS01_S] = hits01_s;
        eval[_HITS01_R] = hits01_r;
        eval[_HITS01_O] = hits01_o;

        eval[_HITS03_S] = hits03_s;
        eval[_HITS03_R] = hits03_r;
        eval[_HITS03_O] = hits03_o;

        eval[_HITS10_S] = hits10_s;
        eval[_HITS10_R] = hits10_r;
        eval[_HITS10_O] = hits10_o;

        kv.Wait(kv.Push(eval_key_vec, eval)); // push partial evaluation results

        // wait for all nodes to finish evaluation
        Postoffice::Get()->Barrier(1, kServerGroup);

        // calculate final results
        unordered_map<string, double> info;
        if (worker_id == 0) {
          // pull aggregated results form PS
          kv.Wait(kv.Pull(eval_key_vec, &eval));

          info["mrr_s"] = eval[_MRR_S] / N;
          info["mrr_r"] = eval[_MRR_R] / N;
          info["mrr_o"] = eval[_MRR_O] / N;
          info["mrr_s_raw"] = eval[_MRR_S_RAW] / N;
          info["mrr_o_raw"] = eval[_MRR_O_RAW] / N;

          info["mr_s"] = eval[_MR_S] / N;
          info["mr_r"] = eval[_MR_R] / N;
          info["mr_o"] = eval[_MR_O] / N;
          info["mr_s_raw"] = eval[_MR_S_RAW] / N;
          info["mr_o_raw"] = eval[_MR_O_RAW] / N;

          info["hits01_s"] = eval[_HITS01_S] / N;
          info["hits01_r"] = eval[_HITS01_R] / N;
          info["hits01_o"] = eval[_HITS01_O] / N;

          info["hits03_s"] = eval[_HITS03_S] / N;
          info["hits03_r"] = eval[_HITS03_R] / N;
          info["hits03_o"] = eval[_HITS03_O] / N;

          info["hits10_s"] = eval[_HITS10_S] / N;
          info["hits10_r"] = eval[_HITS10_R] / N;
          info["hits10_o"] = eval[_HITS10_O] / N;
        }

        return info;
    }

private:

  tuple<double, double, double, double, double> rank(const Model *model, const triplet& sro, vector<double>& E, vector<double>& R) {
        int rank_s = 1;
        int rank_r = 1;
        int rank_o = 1;

        long s = get<0>(sro);
        long r = get<1>(sro);
        long o = get<2>(sro);


        double* E_s = E.data() + s*entity_vector_length;

        double* R_r = R.data() + r*relation_vector_length;

        double* E_o = E.data() + o*entity_vector_length;

        // XXX:
        // There might be degenerated cases when all output scores == 0, leading to perfect but meaningless results.
        // A quick fix is to add a small offset to the base_score.
        double base_score = model->score(E_s, R_r, E_o) - 1e-32;

        // report nans if score produces nans
        if(std::isnan(base_score)) {
          return make_tuple(NAN, NAN, NAN, NAN, NAN);
        }

        for (long ss = 0; ss < ne; ss++) {
          double* E_ss = E.data() + ss*entity_vector_length;
          auto score = model->score(E_ss, R_r, E_o);
          if (score > base_score || std::isnan(score)) rank_s++;
        }

        for (long rr = 0; rr < nr; rr++) {
          double *R_rr = R.data() + rr*relation_vector_length;
          auto score = model->score(E_s, R_rr, E_o);
          if (score > base_score || std::isnan(score)) rank_r++;
        }

        for (long oo = 0; oo < ne; oo++) {
          double* E_oo = E.data() + oo*entity_vector_length;
          auto score = model->score(E_s, R_r, E_oo);
          if (score > base_score || std::isnan(score)) rank_o++;
        }

        int rank_s_raw = rank_s;
        int rank_o_raw = rank_o;

        for (long ss : sro_bucket.or2s(o, r)) {
          double* E_ss = E.data() + ss*entity_vector_length;
          if (model->score(E_ss, R_r, E_o) > base_score) rank_s--;
        }

        for (long oo : sro_bucket.sr2o(s, r)) {
          double* E_oo = E.data() + oo*entity_vector_length;
          if (model->score(E_s, R_r, E_oo) > base_score) rank_o--;
        }

        return make_tuple(rank_s, rank_r, rank_o, rank_s_raw, rank_o_raw);
    }
};

void pretty_print(std::string prefix_str, const unordered_map<string, double>& info) {
  const char* prefix = prefix_str.c_str();
    printf("%s  MRR    \t%.2f\t%.2f\t%.2f\n", prefix, 100*info.at("mrr_s"),    100*info.at("mrr_r"),    100*info.at("mrr_o"));
    printf("%s  MRR_RAW\t%.2f\t%.2f\n", prefix, 100*info.at("mrr_s_raw"),    100*info.at("mrr_o_raw"));
    printf("%s  MR     \t%.2f\t%.2f\t%.2f\n", prefix, info.at("mr_s"), info.at("mr_r"), info.at("mr_o"));
    printf("%s  MR_RAW \t%.2f\t%.2f\n", prefix, info.at("mr_s_raw"), info.at("mr_o_raw"));
    printf("%s  Hits@01\t%.2f\t%.2f\t%.2f\n", prefix, 100*info.at("hits01_s"), 100*info.at("hits01_r"), 100*info.at("hits01_o"));
    printf("%s  Hits@03\t%.2f\t%.2f\t%.2f\n", prefix, 100*info.at("hits03_s"), 100*info.at("hits03_r"), 100*info.at("hits03_o"));
    printf("%s  Hits@10\t%.2f\t%.2f\t%.2f\n", prefix, 100*info.at("hits10_s"), 100*info.at("hits10_r"), 100*info.at("hits10_o"));
}

Evaluator* evaluator_va;
Evaluator* evaluator_tr;
Model* model;

// based on Google's word2vec
int ArgPos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
        if (a == argc - 1) {
            printf("Argument missing for %s\n", str);
            exit(1);
        }
        return a;
    }
    return -1;
}



class Complex : public Model {
    int nh;


public:
  Complex(long ne, long nr, int nh, double eta) : Model(eta) {
        assert( nh % 2 == 0 );
        this->nh = nh;
  }

  vector<double> init_ps(std::function<double()> init_rand) {
    auto c = init_e;
    vector<double> matrix ((ne+nr) * embed_dim * 2);
    for (long i = 0; i!=ne+nr; ++i) {
      double* mpos = matrix.data() + i*nh*2;
      for (long j = 0; j!=nh; ++j) {
        mpos[j] = init_rand();
      }
      for (long j = nh; j != nh*2; ++j) {
        mpos[j] = c;
      }
    }
    return matrix;
  }

  double score(double* E_s, double* R_r, double* E_o) const {
    double dot = 0;

    int nh_2 = nh/2;
    for (int i = 0; i < nh_2; i++) {
      dot += R_r[i]      * E_s[i]      * E_o[i];
      dot += R_r[i]      * E_s[nh_2+i] * E_o[nh_2+i];
      dot += R_r[nh_2+i] * E_s[i]      * E_o[nh_2+i];
      dot -= R_r[nh_2+i] * E_s[nh_2+i] * E_o[i];
    }
    return dot;
  }
  void score_grad(double* E_s, double* R_r, double* E_o,
                  double* d_s, double* d_r, double* d_o) {

    int nh_2 = nh/2;
    for (int i = 0; i < nh_2; i++) {
      // re
      d_s[i] = R_r[i] * E_o[i] + R_r[nh_2+i] * E_o[nh_2+i];
      d_r[i] = E_s[i] * E_o[i] + E_s[nh_2+i] * E_o[nh_2+i];
      d_o[i] = R_r[i] * E_s[i] - R_r[nh_2+i] * E_s[nh_2+i];
      // im
      d_s[nh_2+i] = R_r[i] * E_o[nh_2+i] - R_r[nh_2+i] * E_o[i];
      d_r[nh_2+i] = E_s[i] * E_o[nh_2+i] - E_s[nh_2+i] * E_o[i];
      d_o[nh_2+i] = R_r[i] * E_s[nh_2+i] + R_r[nh_2+i] * E_s[i];
    }
  }
};


class Rescal : public Model {
  int nh;

public:
  Rescal(long ne, long nr, int nh, double eta) : Model(eta) {
    this->nh = nh;
  }

  vector<double> init_ps(std::function<double()> init_rand) {
    auto c = init_e;
    vector<double> matrix ((ne+nr*embed_dim) * embed_dim * 2);
    for (long i = 0; i!=ne; ++i) {
      double* mpos = matrix.data() + i*nh*2;
      for (long j = 0; j!=nh; ++j) {
        mpos[j] = init_rand();
      }
      for (long j = nh; j != nh*2; ++j) {
        mpos[j] = c;
      }
    }
    for (long i = 0; i!=nr; ++i) {
      double* mpos = matrix.data() + ne*nh*2 + i*nh*nh*2;
      for (long j = 0; j!=nh*nh; ++j) {
        mpos[j] = init_rand();
      }
      for (long j = nh*nh; j != nh*nh*2; ++j) {
        mpos[j] = c;
      }
    }
    return matrix;
  }

  // sizes:            h            h^2          h
  double score(double* E_s, double* R_r, double* E_o) const {
    double dot = 0;

    // assume R is row-major [row1col1, row1col2, ..., row2col1, row2col2, ...]
    for (int a = 0; a != nh; ++a) {
      auto anh = a*nh;
      for (int b = 0; b != nh; ++b) {
        dot += R_r[anh+b] * E_s[a] * E_o[b];
      }
    }
    return dot;
  }

  void score_grad(double* E_s, double* R_r, double* E_o,
                  double* d_s, double* d_r, double* d_o) {

    for (int a = 0; a != nh; ++a) {
      auto anh = a*nh;
      d_s[a] = 0;
      d_o[a] = 0;
      for (int b = 0; b != nh; ++b) {
        d_s[a] += R_r[anh+b] * E_o[b]; // gradient for subject
        d_o[a] += R_r[b*nh+a] * E_s[b]; // gradient for object
        d_r[anh+b] = E_s[a] * E_o[b]; // gradient for relation
      }
    }

  }
};

// evaluate the current model (on train and validation set)
void run_eval(const uint epoch, double& best_mrr, const int worker_id, WorkerT& kv) {

  std::vector<ValT> E {};
  std::vector<ValT> R {};
  pull_full_model(E, R, kv);

  // evaluate
  util::Stopwatch sw_eval;
  sw_eval.start();
  ADLOG("Evaluation (TR truncate " << eval_truncate_tr << ", VA truncate " << eval_truncate_va << ")");
  auto info_tr = evaluator_tr->evaluate(model, eval_truncate_tr, E, R, worker_id, kv);
  auto info_va = evaluator_va->evaluate(model, eval_truncate_va, E, R, worker_id, kv);
  sw_eval.stop();
  ADLOG("Worker " << worker_id << ": eval finished (" << sw_eval << ")");

  // save the best model to disk
  if (worker_id == 0) {
    double curr_mrr = (info_va["mrr_s"] + info_va["mrr_o"])/2;
    if (curr_mrr > best_mrr) {
      best_mrr = curr_mrr;
    }

    printf("\n");
    printf("    Dist    EV Elapse    %f\n", sw_eval.elapsed_s());
    printf("======================================\n");
    pretty_print(std::to_string(epoch)+std::string("-TR"), info_tr);
    printf("\n");
    pretty_print(std::to_string(epoch)+std::string("-VA"), info_va);
    printf("\n");
    printf("%i-VA  MRR_CURR    %.2f\n", epoch, 100*best_mrr);
    printf("%i-VA  MRR_BEST    %.2f\n", epoch, 100*best_mrr);
    printf("\n");
  }
}


void RunWorker(int customer_id, ServerT* server=nullptr) {
  std::unordered_map<std::string, util::Stopwatch> sw {};
  WorkerT kv(customer_id, *server);

  int worker_id = ps::MyRank()*num_threads+customer_id; // a unique id for this worker thread

  int N = sros_tr.size();
  int N_per_thread = ceil(1.0 * N / num_workers);
  vector<int> pi = range(N_per_thread, N_per_thread*worker_id);

  ADLOG("Worker " << worker_id << " reads data points " << pi[0] << ".." << pi[pi.size()-1]);
  std::mt19937 gen_shuffle (model_seed^worker_id);

  // replicate all keys on all nodes throughout training
  // (sensible only in ablation experiments)
  if (enforce_full_replication && customer_id == 0) {
  	std::vector<Key> keys (num_keys);
    std::iota(keys.begin(), keys.end(), 0);
    kv.Intent(keys, 0, CLOCK_MAX);
  }
  kv.WaitSync();
  kv.Barrier();
  kv.WaitSync();

  // Initialize model
  kv.BeginSetup();
  if (init_parameters != "none" && worker_id == 0) {
    vector<Key> keys (num_keys);
    std::iota(keys.begin(), keys.end(), 0);
    std::mt19937 gen_initial(model_seed);
    vector<double> init_vals;
    const std::regex uniform_regex("uniform\\{([-0-9\\+\\.e]+)/([-0-9\\+\\.e]+)\\}");
    const std::regex normal_regex("normal\\{([-0-9\\+\\.e]+)/([-0-9\\+\\.e]+)\\}");
    std::smatch matches;

    if (std::regex_match(init_parameters, matches, uniform_regex)) { // uniform distribution
      auto low = std::stod(matches[1].str());
      auto high = std::stod(matches[2].str());
      ADLOG("Init model (uniform{" << low << "/" << high <<"}, seed " << model_seed << ") ... ");
      std::uniform_real_distribution<double> dist (low, high);
      init_vals = model->init_ps([&dist, &gen_initial](){return dist(gen_initial);});

    } else if (std::regex_match(init_parameters, matches, normal_regex)) { // normal distribution
      auto mean = std::stod(matches[1].str());
      auto std = std::stod(matches[2].str());
      ADLOG("Init model (normal{" << mean << "/" << std <<"}, seed " << model_seed << ") ... ");
      std::normal_distribution<double> dist (mean, std);
      init_vals = model->init_ps([&dist, &gen_initial](){return dist(gen_initial);});

    } else {
      ALOG("Invalid init method: " << init_parameters);
      abort();
    }

    kv.Wait(kv.Push(keys, init_vals));
    ADLOG("Init model done: " << init_vals[0] << " " << init_vals[1] << " " << init_vals[2] << " ... ");
  }
  kv.EndSetup();

  // Signal long-term intent for the parameters of local relations
  if (signal_initial_relations_intent) {
    std::unordered_set<Key> local_relations;
    for (int i = 0; i < N_per_thread; i++) {
      if(pi[i] >= N) continue;
      triplet sro = sros_tr[pi[i]];
      int r = get<1>(sro);
      local_relations.insert(relation_key(r));
    }
    ADLOG("Worker " << worker_id << " has long-term intent for relations " << local_relations);
    kv.Intent(std::move(local_relations), 0, CLOCK_MAX);
  }
  kv.Barrier();

  vector<SampleID> sample_ids (N_per_thread, 1);
  double best_mrr = 0;

  // initial evaluation
  if (eval_freq != -1 && run_initial_evaluation) {
    // the first worker on each node runs the evaluation
    if (customer_id == 0) {
      run_eval(0, best_mrr, worker_id, kv);
    }
    kv.Barrier(); // wait for evaluation to finish
  }

  kv.Barrier(); // make sure all workers start the epoch at the same time
  for (uint epoch = 1; epoch <= num_epochs; epoch++) {
    ADLOG("Worker " << worker_id << " starts epoch " << epoch);
    sw["epoch"].start(); sw["epoch_worker"].start(); sw["runtime"].resume();

    // iterate over data points
    shuffle(pi.begin(), pi.end(), gen_shuffle);
    double bce_loss = 0;
    double reg_loss = 0;
    int i_future = 0;

    // iterate over data points
    for (int i = 0; i < N_per_thread; i++) {
      // Prepare future data points. In the first loop iteration, we prepare
      // multiple data points. In later iterations, we prepare exactly one
      // future data point
      while (i_future <= i+static_cast<int>(signal_intent_ahead) && i_future < N_per_thread) {
        if(pi[i_future] >= N) {
          ++i_future;
          continue;
        }

        auto futureClock = kv.currentClock()+i_future-i;

        // prepare sample (for negative samples)
        sample_ids[i_future] = kv.PrepareSample(neg_ratio * 2, futureClock);

        // signal intent for (positive) entities and relations of this data point
        if (signal_intent_ahead != 0) {
          auto sro_f = sros_tr[pi[i_future]];
          std::unordered_set<Key> intent;
          intent.insert(entity_key  (get<0>(sro_f)));
          intent.insert(relation_key(get<1>(sro_f)));
          intent.insert(entity_key  (get<2>(sro_f)));

          kv.Intent(std::move(intent), futureClock);
        }

        ++i_future;
      }

      // the last worker can have less than N_per_thread data points
      if(pi[i] >= N) {
        kv.advanceClock();
        continue;
      }

      // run only first N data points per thread (for fast testing)
      if (max_N_per_thread != -1 && i >= max_N_per_thread) {
        ADLOG("Worker " << worker_id << ": SHORT epoch. Ended after " << max_N_per_thread << " data points");
        break;
      }

      // train
      triplet sro = sros_tr[pi[i]];
      int s = get<0>(sro);
      int r = get<1>(sro);
      int o = get<2>(sro);

      // positive example
      model->train(s, r, o, true, kv, bce_loss, reg_loss);


      // negative examples
      for (int j = 0; j < neg_ratio; j++) {
        int oo = -sample_ids[i]; // pass on sample ID as a negative integer
        int ss = -sample_ids[i];

        // XXX: it is empirically beneficial to carry out updates even if oo == o || ss == s.
        // This might be related to regularization.
        model->train(s, r, oo, false, kv, bce_loss, reg_loss);
        model->train(ss, r, o, false, kv, bce_loss, reg_loss);

      }

      kv.advanceClock();
    }

    // make sure all workers finished
    sw["epoch_worker"].stop();
    ADLOG("Worker " << worker_id << " finished epoch " << epoch << " (" << sw["epoch_worker"] << ")");
    kv.Barrier();
    sw["epoch"].stop();
    sw["runtime"].stop();

    // calculate global training loss
    long long num_train_steps = static_cast<long long>(N) * (1+neg_ratio*2);
    auto global_bce_loss = ps_allreduce(bce_loss, worker_id, loss_key, kv) / num_train_steps;
    auto global_reg_loss = ps_allreduce(reg_loss, worker_id, loss_key, kv) / num_train_steps;
    auto global_loss = global_bce_loss + global_reg_loss;

    if (worker_id == 0) {
      ADLOG("All workers finished epoch " << epoch << " (epoch: " << sw["epoch"] << ", total: " << sw["runtime"] << "). Training loss: " << global_loss << " (" << global_bce_loss << " + " << global_reg_loss << ").");
    }

    // save checkpoint
    kv.WaitSync();
    kv.Barrier();
    if (worker_id == 0 && !model_path.empty() && epoch % save_every_nth_epoch == 0) {
      model->save(epoch, model_path, kv, true, write_end_checkpoint && epoch == num_epochs);
    }


    // evaluation
    if (eval_freq != -1 && epoch % eval_freq == 0) {
      // the first worker on each node runs the evaluation
      if (customer_id == 0) {
        run_eval(epoch, best_mrr, worker_id, kv);
      }
    }

    // maximum time
    if (sw["runtime"].elapsed_s() > max_runtime ||
        sw["runtime"].elapsed_s() + sw["epoch"].elapsed_s() > max_runtime * 1.05) {
      ADLOG("Worker " << worker_id << " stops after epoch " << epoch << " because max. time is reached: " << sw["runtime"].elapsed_s() << "s (+1 epoch) > " << max_runtime << "s (epoch: " << sw["epoch"].elapsed_s() << "s)");
      break;
    }

    kv.Barrier(); // wait for all workers to finish
  }

  kv.Finalize();
}


int process_program_options(const int argc, const char *const argv[]) {
  namespace po = boost::program_options;
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "produce help message")
    ("algorithm", po::value<string>(&alg)->default_value("ComplEx"), "KGE Model. ComplEx and RESCAL are supported")
    ("dataset", po::value<string>(&dataset)->default_value(""), "dataset")
    ("write_embeddings", po::value<string>(&model_path)->default_value(""), "directory to checkpoint model to after each epoch")
    ("write_end_checkpoint", po::value<bool>(&write_end_checkpoint)->default_value(false), "whether to write a model checkpoint at the end of the training job")
    ("write_every", po::value<int>(&save_every_nth_epoch)->default_value(1), "checkpoint the model only every nth epoch (default: 1, i.e., save every epoch)")
    ("embed_dim", po::value<uint>(&embed_dim)->default_value(10), "embedding depth")
    ("num_threads,t", po::value<uint>(&num_threads)->default_value(2), "number of worker threads to run (per process)")
    ("num_epochs", po::value<uint>(&num_epochs)->default_value(10), "number of epochs to run")
    ("eta", po::value<double>(&eta)->default_value(0.1), "initial learning rate")
    ("gamma_entity", po::value<double>(&gamma_entity)->default_value(1e-3), "regularization for entities")
    ("gamma_relation", po::value<double>(&gamma_relation)->default_value(1e-3), "regularization for relations")
    ("dropout_entity", po::value<double>(&dropout_entity)->default_value(0), "During training, randomly zeroes elements of entity embeddings with probability `dropout`. We scale other elements with a factor of 1/(1-p).")
    ("dropout_relation", po::value<double>(&dropout_relation)->default_value(0), "During training, randomly zeroes elements of relation embeddings with probability `dropout`. 0-> no dropout. We scale other elements with a factor of 1/(1-p).")
    ("neg_ratio", po::value<int>(&neg_ratio)->default_value(6), "negative ratio")
    ("eval_freq", po::value<int>(&eval_freq)->default_value(-1), "evaluation frequency")
    ("num_entities", po::value<long>(&ne)->default_value(14951), "number of entities")
    ("num_relations", po::value<long>(&nr)->default_value(1345), "number of relations")
    ("signal_intent_ahead", po::value<uint>(&signal_intent_ahead)->default_value(1000), "number of triples to look ahead")
    ("async_push", po::value<bool>(&async_push)->default_value(true), "push synchronously (false) or asynchronously (true, default)")
    ("signal_initial_relations_intent", po::value<bool>(&signal_initial_relations_intent)->default_value(false), "whether to signal long-term intent for relations or not")
    ("read_partitioned_dataset", po::value<bool>(&read_partitioned_dataset)->default_value(false), "read partitioned dataset")
    ("init_parameters", po::value<string>(&init_parameters)->default_value("normal{0,0.1}"), "initialize parameters, possible: 'none', 'uniform{a,b}', 'normal{mean,std}'")
    ("enforce_random_keys", po::value<bool>(&enforce_random_keys)->default_value(false), "enforce that keys are assigned randomly")
    ("enforce_full_replication", po::value<bool>(&enforce_full_replication)->default_value(false), "manually enforce full model replication")
    ("eval_truncate_tr", po::value<size_t>(&eval_truncate_tr)->default_value(2048), "truncate training dataset in evaluation (0 for no truncation)")
    ("eval_truncate_va", po::value<size_t>(&eval_truncate_va)->default_value(0), "truncate validation dataset in evaluation (0 for no truncation)")
    ("eval_initial", po::value<bool>(&run_initial_evaluation)->default_value(false), "run initial evaluation (default: no)")
    ("model_seed", po::value<unsigned>(&model_seed)->default_value(134827), "seed for model generation")
    ("max_N", po::value<long>(&max_N_per_thread)->default_value(-1), "set an artificial maximum of data points per worker thread (for fast testing)")
    ("max_runtime", po::value<long>(&max_runtime)->default_value(std::numeric_limits<long>::max()), "set a maximum run tim, after which the job will be terminated (in seconds)")
    ;

  // add system options
  ServerT::AddSystemOptions(desc);

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  // which file to read training data from? (determine this now because we need it in multiple places)
  num_serv =  atoi(Environment::Get()->find("DMLC_NUM_SERVER"));
  tr_file  = "train" + (read_partitioned_dataset ? ".partitioned."+to_string(num_serv)+"x"+to_string(num_threads) : "");

  if (vm.count("help")) {
    cout << desc << "\n";
    return 1;
  }

  if (dropout_entity < 0 || dropout_entity > 1 ||
      dropout_relation < 0 || dropout_relation > 1) {
    cout << "Dropout probability should be in the range 0 <= p <= 1. Given " << dropout_entity << " for entities and " << dropout_relation << " for relations. Usage:\n\n";
    cout << desc << "\n";
    return 1;
  }

  return 0;
}


int main(int argc, char *argv[]) {
  // Read cmd arguments
  int po_error = process_program_options(argc, argv);
  if(po_error) return 1;


  // set model
  entity_vector_length   = embed_dim * 2;
  num_keys = ne + nr;
  if (alg == "RESCAL") {
    rel_dim = embed_dim * embed_dim;
    relation_vector_length = embed_dim * embed_dim * 2;
    model = new Rescal (ne, nr, embed_dim, eta);
    algorithm = Alg::RESCAL;
  } else if (alg == "ComplEx") {
    rel_dim = embed_dim;
    relation_vector_length = entity_vector_length;
    model = new Complex (ne, nr, embed_dim, eta);
    algorithm = Alg::ComplEx;
  } else {
    ADLOG("Unkown KGE model '" << alg << "'! Supported: RESCAL and ComplEx");
    return 0;
  }

  // enforce random parameter allocation
  if (enforce_random_keys) {
    // manual mapping: parameter->key
    key_assignment.resize(ne+nr);
    iota(key_assignment.begin(), key_assignment.end(), 0);

    srand(2); // enforce same seed among different ranks
    random_shuffle(key_assignment.begin(), key_assignment.end());
  }

  // setup PS
  Setup(num_keys+2, num_threads);

  std::string role = std::string(getenv("DMLC_ROLE"));
  ALOG("kge. Starting " << role << ": running " << num_epochs << " epochs of " << alg << " on " << ne << " entities and " << nr << " relations (" << embed_dim << " depth) on " << dataset << "\n" << num_threads << " threads, " << neg_ratio << " neg_ratio, " << eval_freq << " eval_freq, " << eta << " eta, " << gamma_entity << " gamma entity, " << gamma_relation << " gamma relation, \nasync push " << async_push << ", signal intent ahead " << signal_intent_ahead << ", initial relations intent signal " << signal_initial_relations_intent << ", read partitioned dataset " << read_partitioned_dataset << " (" << tr_file << "), enforce_random_keys " << enforce_random_keys << ". \n" << (SyncManager<ValT,HandleT>::PrintOptions()));

  if (role.compare("scheduler") == 0) {
    Scheduler();
  } else if (role.compare("server") == 0) { // worker+server


    // Data read
    sros_tr = create_sros(dataset + tr_file + ".del");
    sros_va = create_sros(dataset + "valid.del");
    sros_te = create_sros(dataset + "test.del");

    sros_al.insert(sros_al.end(), sros_tr.begin(), sros_tr.end());
    sros_al.insert(sros_al.end(), sros_va.begin(), sros_va.end());
    sros_al.insert(sros_al.end(), sros_te.begin(), sros_te.end());

    SROBucket sro_bucket_al(sros_al);


    evaluator_va = new Evaluator(ne, nr, sros_va, sro_bucket_al);
    evaluator_tr = new Evaluator(ne, nr, sros_tr, sro_bucket_al);

    // Value lengths of keys
    std::vector<size_t> value_lengths {};
    value_lengths.reserve(ne+nr+2);
    for (long i=0; i!=ne; ++i) value_lengths.push_back(entity_vector_length); // entity embeddings
    for (long i=0; i!=nr; ++i) value_lengths.push_back(relation_vector_length); // relation embeddings
    // loss key
    loss_key = static_cast<Key>(num_keys);
    value_lengths.push_back(1);
    // eval key
    eval_key = static_cast<Key>(num_keys+1);
    value_lengths.push_back(20);

    // Start the server system
    auto server = new ServerT(value_lengths);
    RegisterExitCallback([server](){ delete server; });

    num_workers = ps::NumServers() * num_threads;

    // make sure all servers are set up
    server->Barrier();

    // set up negative sampling
    negs_gen = std::mt19937(model_seed^Postoffice::Get()->my_rank());
    negs_dist = std::uniform_int_distribution<int>{0, static_cast<int>(ne-1)};
    server->enable_sampling_support(&DrawEntity, 0, (enforce_random_keys ? 0 : ne)); // if we don't enforce random keys, the sampling range is continuous


    // run worker(s)
    std::vector<std::thread> workers {};
    for (size_t i=0; i!=num_threads; ++i) {
      workers.push_back(std::thread(RunWorker, i, server));
      std::string name = std::to_string(ps::MyRank())+"-worker-"+std::to_string(ps::MyRank()*num_threads + i);
      SET_THREAD_NAME((&workers[workers.size()-1]), name.c_str());
    }

    // wait for the workers to finish
    for (auto & w : workers)
      w.join();

    // stop the server
    server->shutdown();
  }
}
