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
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <random>
#include <iomanip>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <tuple>
#include <random>
#include <cassert>
#include <cstring>


using namespace ps;
using namespace std;

typedef double ValT;
typedef DefaultColoServerHandle<ValT> HandleT;
typedef ColoKVServer<ValT, HandleT> ServerT;
typedef ColoKVWorker<ValT, HandleT> WorkerT;

enum class Alg { ComplEx, RESCAL };


// Configuration
string  alg;
Alg     algorithm;
string  dataset;
uint    embed_dim;
uint    rel_dim;
double  eta;
double  gamma_param;
int     neg_ratio;
uint    num_epochs;
uint    num_threads;
int     eval_freq;
uint    ne; // number of entities
uint    nr; // number of relations
string  model_path;


// Scale parameters
bool async_push;
bool localize_relations;
uint localize_entities_ahead;
bool read_partitioned_dataset;
bool location_caches;

uint num_workers = -1;
uint num_keys = -1;
uint entity_vector_length;
uint relation_vector_length;


static default_random_engine GLOBAL_GENERATOR;
static uniform_real_distribution<double> UNIFORM(0, 1);

typedef tuple<int, int, int> triplet;


// Parameter server functions
inline const Key entity_key  (const int e) { return e; }


vector<Key> relation_keys(const int r){
    if(algorithm == Alg::ComplEx){
        vector<Key> vec_r{ne+r};
        return vec_r;
    } else {
        vector<Key> vec_r(embed_dim);
        std::iota(vec_r.begin(),vec_r.end(), ne+r*embed_dim);
        return vec_r;
    }
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


double sigmoid(double x, double cutoff=30) {
    if (x > +cutoff) return 1.;
    if (x < -cutoff) return 0.;
    return 1./(1.+exp(-x));
}

class SROBucket {
    unordered_set<int64_t> __sros;
    unordered_map<int64_t, vector<int>> __sr2o;
    unordered_map<int64_t, vector<int>> __or2s;

    int64_t hash(int a, int b, int c) const {
        int64_t x = a;
        x = (x << 20) + b;
        return (x << 20) + c;
    }

    int64_t hash(int a, int b) const {
        int64_t x = a;
        return (x << 32) + b;
    }

public:
    SROBucket(const vector<triplet>& sros) {
        for (auto sro : sros) {
            int s = get<0>(sro);
            int r = get<1>(sro);
            int o = get<2>(sro);

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

// try sample pairs
class NegativeSampler {
    uniform_int_distribution<int> unif_e;
    uniform_int_distribution<int> unif_r;
    default_random_engine generator;

public:
    NegativeSampler(int ne, int nr, int seed) :
        unif_e(0, ne-1), unif_r(0, nr-1), generator(seed) {}

    int random_entity() {
        return unif_e(generator);
    }

    int random_relation() {
        return unif_r(generator);
    }
};


class Model {

protected:
    double eta;
    double gamma;
    const double init_b = 1e-2;
    const double init_e = 1e-6;

    vector<vector<double>> E;
    vector<vector<double>> R;
    vector<vector<double>> E_g;
    vector<vector<double>> R_g;

public:

    Model(double eta, double gamma) {
        this->eta = eta;
        this->gamma = gamma;
    }

  // TODO: TEST
  void save(const string& fname, WorkerT& kv) {
      ofstream ofs(fname, ios::out);

      for (unsigned i = 0; i < ne; i++) {

          vector<double> embedding (entity_vector_length);
          vector<Key> key { entity_key(i) };
          kv.Wait(kv.Pull(key, &embedding));

          for (unsigned j = 0; j < embed_dim; j++)
              ofs << embedding[j] << ' ';
          ofs << endl;
      }

      //TODO: this does not store all relations for RESCAL
      for (unsigned i = 0; i < nr; i++) {

        vector<double> embedding (entity_vector_length);
        vector<Key> key {};
        vector<Key> vec_r = relation_keys(i);
        for(unsigned k = 0; k<vec_r.size(); k++){
          Key key_r = vec_r[k];
          key.push_back(key_r);
        }
        kv.Wait(kv.Pull(key, &embedding));

          for (unsigned j = 0; j < embed_dim; j++)
              ofs << embedding[j] << ' ';
          ofs << endl;
      }
      ofs.close();
  }


  // TEMP
  // void load(const string& fname) {
  //     ifstream ifs(fname, ios::in);

  //     for (unsigned i = 0; i < E.size(); i++)
  //         for (unsigned j = 0; j < E[i].size(); j++)
  //             ifs >> E[i][j];

  //     for (unsigned i = 0; i < R.size(); i++)
  //         for (unsigned j = 0; j < R[i].size(); j++)
  //             ifs >> R[i][j];

  //     ifs.close();
  // }

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

  void train(int s, int r, int o, bool is_positive, WorkerT& kv) {


      // Get embeddings from parameter server
      vector<double> embed_s   (entity_vector_length);
      vector<double> embed_r   (relation_vector_length);
      vector<double> embed_o   (entity_vector_length);
      vector<double> update_s  (embed_s.size());
      vector<double> update_r  (embed_r.size());
      vector<double> update_o  (embed_o.size());
      vector<Key> key_s { entity_key(s) };
      vector<Key> key_r = relation_keys(r);
      vector<Key> key_o { entity_key(o) };

      kv.Wait(kv.Pull(key_s, &embed_s), kv.Pull(key_r, &embed_r), kv.Pull(key_o, &embed_o));

      double* E_s = embed_s.data();
      double* R_r = embed_r.data();
      double* E_o = embed_o.data();
      double* d_s = update_s.data();
      double* d_r = update_r.data();
      double* d_o = update_o.data();


      double offset = is_positive ? 1 : 0;
      double d_loss = sigmoid(score(E_s, R_r, E_o)) - offset;

      // score_grad(s, r, o, d_s, d_r, d_o);
      score_grad(E_s, R_r, E_o, d_s, d_r, d_o);


      for (unsigned i = 0; i < embed_dim; i++) d_s[i] *= d_loss;
      for (unsigned i = 0; i < rel_dim;   i++) d_r[i] *= d_loss;
      for (unsigned i = 0; i < embed_dim; i++) d_o[i] *= d_loss;


      double gamma_s = gamma / embed_dim;
      double gamma_r = gamma / rel_dim;
      double gamma_o = gamma / embed_dim;


      for (unsigned i = 0; i < embed_dim; i++) d_s[i] += gamma_s * E_s[i];
      for (unsigned i = 0; i < rel_dim;   i++) d_r[i] += gamma_r * R_r[i];
      for (unsigned i = 0; i < embed_dim; i++) d_o[i] += gamma_o * E_o[i];

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

  virtual vector<double> init_ps() = 0;

};// end Model



class Evaluator {
    int ne;
    int nr;
    const vector<triplet>& sros;
    const SROBucket& sro_bucket;

public:
    Evaluator(int ne, int nr, const vector<triplet>& sros, const SROBucket& sro_bucket) :
      ne(ne), nr(nr), sros(sros), sro_bucket(sro_bucket) {}

  unordered_map<string, double> evaluate(const Model *model, int truncate, vector<double>& E, vector<double>& R) {
        int N = this->sros.size();

        if (truncate > 0)
            N = min(N, truncate);

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


        // #pragma omp parallel for reduction(+: mrr_s, mrr_r, mrr_o, mr_s, mr_r, mr_o,
        //       hits01_s, hits01_r, hits01_o, hits03_s, hits03_r, hits03_o, hits10_s, hits10_r, hits10_o)
        // TODO: parallelize
        for (int i = 0; i < N; i++) {
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

        unordered_map<string, double> info;

        info["mrr_s"] = mrr_s / N;
        info["mrr_r"] = mrr_r / N;
        info["mrr_o"] = mrr_o / N;
        info["mrr_s_raw"] = mrr_s_raw / N;
        info["mrr_o_raw"] = mrr_o_raw / N;

        info["mr_s"] = mr_s / N;
        info["mr_r"] = mr_r / N;
        info["mr_o"] = mr_o / N;
        info["mr_s_raw"] = mr_s_raw / N;
        info["mr_o_raw"] = mr_o_raw / N;

        info["hits01_s"] = hits01_s / N;
        info["hits01_r"] = hits01_r / N;
        info["hits01_o"] = hits01_o / N;

        info["hits03_s"] = hits03_s / N;
        info["hits03_r"] = hits03_r / N;
        info["hits03_o"] = hits03_o / N;

        info["hits10_s"] = hits10_s / N;
        info["hits10_r"] = hits10_r / N;
        info["hits10_o"] = hits10_o / N;

        return info;
    }

private:

  tuple<double, double, double, double, double> rank(const Model *model, const triplet& sro, vector<double>& E, vector<double>& R) {
        int rank_s = 1;
        int rank_r = 1;
        int rank_o = 1;

        int s = get<0>(sro);
        int r = get<1>(sro);
        int o = get<2>(sro);


        double* E_s = E.data() + s*entity_vector_length;

        double* R_r = R.data() + r*relation_vector_length;

        double* E_o = E.data() + o*entity_vector_length;

        // XXX:
        // There might be degenerated cases when all output scores == 0, leading to perfect but meaningless results.
        // A quick fix is to add a small offset to the base_score.
        double base_score = model->score(E_s, R_r, E_o) - 1e-32;

        for (int ss = 0; ss < ne; ss++) {
          double* E_ss = E.data() + ss*entity_vector_length;
          if (model->score(E_ss, R_r, E_o) > base_score) rank_s++;
        }

        for (int rr = 0; rr < nr; rr++) {
          double *R_rr = R.data() + rr*relation_vector_length;
          if (model->score(E_s, R_rr, E_o) > base_score) rank_r++;
        }

        for (int oo = 0; oo < ne; oo++) {
          double* E_oo = E.data() + oo*entity_vector_length;
          if (model->score(E_s, R_r, E_oo) > base_score) rank_o++;
        }

        int rank_s_raw = rank_s;
        int rank_o_raw = rank_o;

        for (auto ss : sro_bucket.or2s(o, r)) {
          double* E_ss = E.data() + ss*entity_vector_length;
          if (model->score(E_ss, R_r, E_o) > base_score) rank_s--;
        }

        for (auto oo : sro_bucket.sr2o(s, r)) {
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
  Complex(int ne, int nr, int nh, double eta, double gamma) : Model(eta, gamma) {
        assert( nh % 2 == 0 );
        this->nh = nh;
  }

  vector<double> init_ps() {
    auto l = -init_b;
    auto h = init_b;
    auto c = init_e;
    vector<double> matrix ((ne+nr) * embed_dim * 2);
    for (uint i = 0; i!=ne+nr; ++i) {
      double* mpos = matrix.data() + i*nh*2;
      for (int j = 0; j!=nh; ++j) {
        mpos[j] = (h-l)*UNIFORM(GLOBAL_GENERATOR) + l;
      }
      for (int j = nh; j != nh*2; ++j) {
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
  Rescal(int ne, int nr, int nh, double eta, double gamma) : Model(eta, gamma) {
    this->nh = nh;
  }

  vector<double> init_ps() {
    auto l = -init_b;
    auto h = init_b;
    auto c = init_e;
    vector<double> matrix ((ne+nr*embed_dim) * embed_dim * 2);
    for (uint i = 0; i!=ne; ++i) {
      double* mpos = matrix.data() + i*nh*2;
      for (int j = 0; j!=nh; ++j) {
        mpos[j] = (h-l)*UNIFORM(GLOBAL_GENERATOR) + l;
      }
      for (int j = nh; j != nh*2; ++j) {
        mpos[j] = c;
      }
    }
    for (uint i = 0; i!=nr; ++i) {
      double* mpos = matrix.data() + ne*nh*2 + i*nh*nh*2;
      for (int j = 0; j!=nh*nh; ++j) {
        mpos[j] = (h-l)*UNIFORM(GLOBAL_GENERATOR) + l;
      }
      for (int j = nh*nh; j != nh*nh*2; ++j) {
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


void RunWorker(int customer_id, ServerT* server=nullptr) {
  Start(customer_id);
  std::unordered_map<std::string, util::Stopwatch> sw {};
  WorkerT kv(0, customer_id, *server);

  int worker_id = ps::MyRank()*num_threads+customer_id-1; // a unique id for this worker thread

  NegativeSampler neg_sampler(ne, nr, rand() ^ worker_id);


  int N = sros_tr.size();
  int N_per_thread = ceil(1.0 * N / num_workers);
  vector<int> pi = range(N_per_thread, N_per_thread*worker_id);

  ADLOG("Worker " << worker_id << " reads data points " << pi[0] << ".." << pi[pi.size()-1]);

  // Initialize model
  if (worker_id == 0) {
    ADLOG("Init model ... ");
    vector<Key> keys (num_keys);
    std::iota(keys.begin(), keys.end(), 0);
    vector<double> init_vals = model->init_ps();
    kv.Wait(kv.Push(keys, init_vals));
    ADLOG("Init model done");
  }
  kv.Barrier();

  // Localize parameters of local relations
  if (localize_relations) {
    std::set<Key> local_relations;
    for (int i = 0; i < N_per_thread; i++) {
      if(pi[i] >= N) continue;
      triplet sro = sros_tr[pi[i]];
      int r = get<1>(sro);
      vector<Key> vec_r = relation_keys(r);
      for(unsigned k = 0; k < vec_r.size(); k++){
        Key key_r = vec_r[k];
        local_relations.insert(key_r);
      }
    }
    std::vector<ps::Key> to_localize;
    std::copy(local_relations.begin(), local_relations.end(), std::back_inserter(to_localize));
    ADLOG("Worker " << worker_id << " localizes relations " << to_localize);
    kv.Wait(kv.Localize(to_localize));
  }
  kv.Barrier();

  int ls = 2; // negatives-ahead step
  vector<int> negatives  (N_per_thread * neg_ratio * ls);
  int neg_pos = 0;

  double best_mrr = 0;

  for (uint epoch = 0; epoch < num_epochs; epoch++) {
    ADLOG("Worker " << worker_id << " starts epoch " << epoch);

    // evaluation
    if (epoch % eval_freq == 0 && eval_freq != -1) {
      if (worker_id == 0) {
        sw["eval"].start();
        // pull full model once
        ADLOG("Pull full model for EV..");
        vector<Key> R_keys (nr);
        if (algorithm == Alg::RESCAL){
          R_keys.resize(nr*embed_dim);
        }
        std::iota(R_keys.begin(), R_keys.end(), ne);
        vector<double> R (nr * relation_vector_length);
        vector<Key> E_keys (ne); std::iota(E_keys.begin(), E_keys.end(), 0);
        vector<double> E (ne * entity_vector_length);
        auto ts1 = kv.Pull(R_keys, &R); kv.Wait(kv.Pull(E_keys, &E)); kv.Wait(ts1);

        ADLOG("Model pulled");

        auto info_tr = evaluator_tr->evaluate(model, 2048, E, R);
        auto info_va = evaluator_va->evaluate(model, 2048, E, R);
        sw["eval"].stop();

        // save the best model to disk
        double curr_mrr = (info_va["mrr_s"] + info_va["mrr_o"])/2;
        if (curr_mrr > best_mrr) {
          best_mrr = curr_mrr;
          if ( !model_path.empty() ) {
            model->save(model_path, kv);
          }
        }

        printf("\n");
        printf("            EV Elapse    %f\n", sw["eval"].elapsed_s());
        printf("======================================\n");
        pretty_print(std::to_string(epoch)+std::string("-TR"), info_tr);
        printf("\n");
        pretty_print(std::to_string(epoch)+std::string("-VA"), info_va);
        printf("\n");
        printf("%i-VA  MRR_BEST    %.2f\n", epoch, 100*best_mrr);
        printf("\n");
      }

      // Wait for evaluation to finish
      kv.Barrier();
    } // end evaluation


    // pre-sample negative samples
    sw["negatives"].start();
    for (int z = 0; z < N_per_thread * neg_ratio; ++z) {
      negatives[z*ls+0] = neg_sampler.random_entity();
      negatives[z*ls+1] = neg_sampler.random_entity();
    }
    neg_pos = 0;
    sw["negatives"].stop();
    ADLOG("Worker " << worker_id << ": Sampled " << negatives.size() << " negatives (" << sw["negatives"] << ")");

    shuffle(pi.begin(), pi.end(), GLOBAL_GENERATOR);


    sw["epoch"].start(); sw["epoch_worker"].start();
    for (int i = 0; i < N_per_thread; i++) {
      // localize embeddings that we will need in a subsequent epoch
      int i_future = i+localize_entities_ahead;
      if (localize_entities_ahead != 0 && i_future < N_per_thread && pi[i_future] < N) {
        auto sro_f = sros_tr[pi[i_future]];
        vector<Key> localize; localize.reserve((1+neg_ratio) * ls);
        localize.push_back(entity_key  (get<0>(sro_f)));
        localize.push_back(entity_key  (get<2>(sro_f)));
        auto future_pos = neg_pos + localize_entities_ahead*neg_ratio*ls;
        for(auto z = 0; z!=neg_ratio; ++z) {
          localize.push_back(entity_key  (negatives[future_pos + z*ls + 0]));
          localize.push_back(entity_key  (negatives[future_pos + z*ls + 1]));
        }
        kv.Localize(localize);
      }

      // the last worker can have less than N_per_thread data points
      if(pi[i] >= N) continue;

      // train
      triplet sro = sros_tr[pi[i]];
      int s = get<0>(sro);
      int r = get<1>(sro);
      int o = get<2>(sro);

      // positive example
      model->train(s, r, o, true, kv);


      // negative examples
      for (int j = 0; j < neg_ratio; j++) {
        int oo = negatives[neg_pos++];
        int ss = negatives[neg_pos++];

        // XXX: it is empirically beneficial to carry out updates even if oo == o || ss == s.
        // This might be related to regularization.
        model->train(s, r, oo, false, kv);
        model->train(ss, r, o, false, kv);

      }
    }

    // make sure all workers finished
    sw["epoch_worker"].stop();
    ADLOG("Worker " << worker_id << " finished epoch " << epoch << " (" << sw["epoch_worker"] << ")");
    kv.Barrier();


    sw["epoch"].stop();
    if (worker_id == 0) {
      ADLOG("All workers finished epoch " << epoch << " (" << sw["epoch"] << ")");
    }

  }


  kv.WaitAll();
  kv.Barrier();
  if (customer_id != 0) {
    Finalize(customer_id, false); // if this is not the main thread, we shut down the system for this thread here
  }
}


int process_program_options(const int argc, const char *const argv[]) {
  namespace po = boost::program_options;
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "produce help message")
    ("algorithm", po::value<string>(&alg)->default_value("ComplEx"), "KGE Model. ComplEx and RESCAL are supported")
    ("dataset", po::value<string>(&dataset)->default_value(""), "dataset")
    ("embed_dim", po::value<uint>(&embed_dim)->default_value(10), "embedding depth")
    ("num_threads,t", po::value<uint>(&num_threads)->default_value(2), "number of worker threads to run (per process)")
    ("num_epochs", po::value<uint>(&num_epochs)->default_value(10), "number of epochs to run")
    ("eta", po::value<double>(&eta)->default_value(0.1), "initial learning rate")
    ("gamma", po::value<double>(&gamma_param)->default_value(1e-3), "gamma")
    ("neg_ratio", po::value<int>(&neg_ratio)->default_value(6), "negative ratio")
    ("eval_freq", po::value<int>(&eval_freq)->default_value(1), "evaluation frequency")
    ("num_entities", po::value<uint>(&ne)->default_value(14951), "number of entities")
    ("num_relations", po::value<uint>(&nr)->default_value(1345), "number of relations")
    ("localize_entities_ahead", po::value<uint>(&localize_entities_ahead)->default_value(0), "number of triples to look ahead")
    ("async_push", po::value<bool>(&async_push)->default_value(false), "push synchronously (default) or asynchronously (true)")
    ("localize_relations", po::value<bool>(&localize_relations)->default_value(false), "whether relations are partitioned (true) or not (default)")
    ("read_partitioned_dataset", po::value<bool>(&read_partitioned_dataset)->default_value(true), "read partitioned dataset")
    ("location_caches", po::value<bool>(&location_caches)->default_value(false), "use location caches")
    ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
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
  if (alg == "RESCAL") {
    rel_dim = embed_dim * embed_dim;
    relation_vector_length = embed_dim * embed_dim * 2;
    model = new Rescal (ne, nr, embed_dim, eta, gamma_param);
    num_keys = ne + nr * embed_dim;
    algorithm = Alg::RESCAL;
  } else if (alg == "ComplEx") {
    rel_dim = embed_dim;
    relation_vector_length = entity_vector_length;
    num_keys = ne + nr;
    model = new Complex (ne, nr, embed_dim, eta, gamma_param);
    algorithm = Alg::ComplEx;
  } else {
    ADLOG("Unkown KGE model '" << alg << "'! Supported: RESCAL and ComplEx");
    return 0;
  }


  Postoffice::Get()->enable_dynamic_allocation(num_keys, num_threads, location_caches);

  std::string role = std::string(getenv("DMLC_ROLE"));
  std::cout << "kge. Starting " << role << ": running " << num_epochs << " epochs of " << alg << " on " << ne << " entities and " << nr << " relations (" << embed_dim << " depth) on " << dataset << "\n" << num_threads << " threads, " << neg_ratio << " neg_ratio, " << eval_freq << " eval_freq, " << eta << " eta, " << gamma_param << " gamma\nasync push " << async_push << ", localize relations " << localize_relations << ", read partitioned dataset " << read_partitioned_dataset << "\n";

  if (role.compare("scheduler") == 0) {
    Start(0);
    Finalize(0, true);
  } else if (role.compare("server") == 0) { // worker+server


    // Data read
    int num_serv =  atoi(Environment::Get()->find("DMLC_NUM_SERVER"));
    sros_tr = create_sros(dataset + "train" + (read_partitioned_dataset ? ".partitioned."+to_string(num_serv)+"x"+to_string(num_threads) : "") + ".del");
    sros_va = create_sros(dataset + "valid.del");
    sros_te = create_sros(dataset + "test.del");

    // TODO-impr: could read this only on the first server
    sros_al.insert(sros_al.end(), sros_tr.begin(), sros_tr.end());
    sros_al.insert(sros_al.end(), sros_va.begin(), sros_va.end());
    sros_al.insert(sros_al.end(), sros_te.begin(), sros_te.end());

    SROBucket sro_bucket_al(sros_al);


    evaluator_va = new Evaluator(ne, nr, sros_va, sro_bucket_al);
    evaluator_tr = new Evaluator(ne, nr, sros_tr, sro_bucket_al);

    // Start the server system
    int server_customer_id = 0; // server gets customer_id=0, workers 1..n
    Start(server_customer_id);
    HandleT handle (num_keys, entity_vector_length); // the handle specifies how the server handles incoming Push() and Pull() calls
    auto server = new ServerT(server_customer_id, handle);
    RegisterExitCallback([server](){ delete server; });

    num_workers = ps::NumServers() * num_threads;


    // run worker(s)
    std::vector<std::thread> workers {};
    for (size_t i=0; i!=num_threads; ++i)
      workers.push_back(std::thread(RunWorker, i+1, server));

    // wait for the workers to finish
    for (auto & w : workers)
      w.join();

    // stop the server
    server->writeStats();
    Finalize(server_customer_id, true);
  }
}
