#include "stddef.h"
#include <cstddef>
#include "utils.h"
#include "ps/ps.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>
#include <unordered_set>
#include <thread>
#include <numeric>
#include <boost/program_options.hpp>
#include <limits>
#include <sstream>
#include <string>
#include <iostream>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <ctime>
#include <random>
#include <dirent.h>


#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40


const int vocab_hash_size = 30000000, table_size = 1e8;  // Maximum 30 * 0.7 = 21M words in the vocabulary

// Precision switchable here
typedef float ValT;

struct vocab_word {
  long long cn;
  int *point;
  char *word, *code, codelen;
};

string in_file, out_file, vocab_save, vocab_retrieve;

struct vocab_word *vocab;
int binary = 0, debug_mode = 2, window = 5, min_reduce = 1, min_count,hs = 0, negative = 25;
int *vocab_hash,*table;
long long vocab_max_size = 1000, vocab_size = 0;
long long train_words = 0, word_count_actual = 0, file_size = 0;
ValT alpha, starting_alpha, sample;
long long int embed_dim,data_words;

bool shuffle_b;
bool write_results;
bool sync_push;
bool clustered_input;
bool localize_positives;
int prep_context_ahead;
bool downsample_replicated_negative_samples;
ValT *expTable;
double neg_power;
string init_model;
unsigned long long model_seed;
long max_runtime;

//shuffled vector acts as a scramble mapping of words - > keys
vector<ps::Key> forwards; // word -> key
vector<long long> backwards; // key -> word

// number of parameters to replicate
size_t replicate_n;

size_t peek_ahead_n;


using namespace ps;
using namespace std;


// get the key for the syn0 parameter of a word
inline Key syn0KeyResolver(long long word) {
  if (shuffle_b) {
    return (forwards[word] * 2);
  } else {
    return (word * 2);
  }
}
// get the key for the syn1 parameter of a word
inline Key syn1KeyResolver(long long word) {
  if (shuffle_b) {
    return (forwards[word] * 2 + 1);
  } else {
    return (word * 2 + 1);
  }
}
// get the word from a syn1 key
inline long long syn1Reverse(Key key) {
  if (shuffle_b) {
    return backwards[(key - 1) / 2];
  } else {
    return (key - 1)/2;
  }
}


typedef DefaultColoServerHandle <ValT> HandleT;
typedef ColoKVServer <ValT, HandleT> ServerT;
typedef ColoKVWorker <ValT, HandleT> WorkerT;

// Config
uint num_workers = -1;
size_t num_iterations = 0;
size_t num_threads = 0;
Key num_keys = 0;


// calculates weight for negative sampling distribution
inline double unigram_pow(int a, unordered_set<Key>& replicated, int num_servers) {
  double pw = pow(vocab[a].cn, neg_power);

  // adjust weight of the negative samples that are replicated
  // (because such negative samples will locally available 100% of the time)
  if (downsample_replicated_negative_samples) {
    if (replicated.find(syn1KeyResolver(a)) != replicated.end()) {
      pw = pw / num_servers;
    }
  }
  return pw;
}

// construct a table of all negative samples for efficient sampling
void InitUnigramTable(const vector<Key>& replicated_parameters) {
  // replicated parameters as set
  unordered_set<Key> replicated (replicated_parameters.begin(), replicated_parameters.end());
  int num_servers = atoi(Environment::Get()->find("DMLC_NUM_SERVER"));

  int a, i;
  double train_words_pow = 0;
  double d1;
  table = (int *) malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += unigram_pow(a, replicated, num_servers);
  i = 0;
  d1 = unigram_pow(i, replicated, num_servers) / train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (double) table_size > d1) {
      if (i == 1 || i == 10 || i == 100 || i == 1000 || i == 10000) {
        ADLOG(setw(5) << setfill(' ') << i << " most frequent words have " << d1 << " sampling probability");
      }
      i++;
      d1 += unigram_pow(i, replicated, num_servers) / train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin, char *eof) {
  int a = 0, ch;
  while (1) {
    ch = fgetc_unlocked(fin);
    if (ch == EOF) {
      *eof = 1;
      break;
    }
    if (ch == 13) continue; // carriage return
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        std::strcpy(word, (char *) "</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin, char *eof) {
  char word[MAX_STRING], eof_l = 0;
  ReadWord(word, fin, &eof_l);

  if (eof_l) {
    *eof = 1;
    return -1;
  }

  return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *) calloc(length, sizeof(char));
  std::strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *) realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
  long long l = ((struct vocab_word *) b)->cn - ((struct vocab_word *) a)->cn;
  if (l > 0) return 1;
  if (l < 0) return -1;
  return 0;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if ((vocab[a].cn < min_count) && (a != 0)) {
      vocab_size--;
      free(vocab[a].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash = GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
  vocab = (struct vocab_word *) realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));

}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++)
    if (vocab[a].cn > min_reduce) {
      vocab[b].cn = vocab[a].cn;
      vocab[b].word = vocab[a].word;
      b++;
    } else free(vocab[a].word);
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

void LearnVocabFromTrainFile() {

  char word[MAX_STRING], eof = 0;
  FILE *fin;
  long long a, i, wc = 0;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;

  fin = fopen(in_file.c_str(), "rb");

  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocab_size = 0;
  AddWordToVocab((char *) "</s>");
  while (1) {
    ReadWord(word, fin, &eof);
    if (eof) break;
    train_words++;
    wc++;
    if ((debug_mode > 1) && (wc >= 1000000)) {
      printf("%lldM%c", train_words / 1000000, 13);
      fflush(stdout);
      wc = 0;
    }
    i = SearchVocab(word);
    if (i == -1) {
      a = AddWordToVocab(word);
      vocab[a].cn = 1;
    } else vocab[i].cn++;
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  file_size = ftell(fin);
  fclose(fin);
}

void SaveVocab() {
  long long i;
  FILE *fo = fopen(vocab_save.c_str(), "wb");
  if (fo == NULL) {
    perror("FAILED to open file to save vocab: ");
}
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

void ReadVocab() {
  long long a, i = 0;
  char c, eof = 0;
  int irrelevant_returnval = 0; // only here so the compiler does not warn anymore, as fscanf has a unused returnvalue
  irrelevant_returnval++;
  irrelevant_returnval--;
  char word[MAX_STRING];
  FILE *fin = fopen(vocab_retrieve.c_str(), "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  while (1) {
    ReadWord(word, fin, &eof);
    if (eof) break;
    a = AddWordToVocab(word);
    irrelevant_returnval = fscanf(fin, "%lld%c", &vocab[a].cn, &c);
    i++;
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fin = fopen(in_file.c_str(), "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}

// writes current embeddings either in binary or human readable format
void write_checkpoint(string output, WorkerT &kv, const bool write_syn1=false) {
  util::Stopwatch sw; sw.start();

  // pull entire model
  vector<Key> syn0_keys (vocab_size);
  vector<Key> syn1_keys (vocab_size);
  for (long a = 0; a < vocab_size; a++) {
    syn0_keys[a] = syn0KeyResolver(a);
    syn1_keys[a] = syn1KeyResolver(a);
  }

  vector<ValT> syn0 (embed_dim * vocab_size);
  vector<ValT> syn1 (embed_dim * vocab_size);

  kv.WaitReplicaSync();
  kv.Wait(kv.Pull(syn0_keys, &syn0));
  if (write_syn1) kv.Wait(kv.Pull(syn1_keys, &syn1));

  if (binary == 1) { // write binary files
    FILE *fo_syn0;
    FILE *fo_syn1;

    fo_syn0 = fopen(output.c_str(), "wb");
    if (write_syn1) fo_syn1 = fopen((output+".syn1").c_str(), "wb");
    fprintf(fo_syn0, "%lld %lld\n", vocab_size, embed_dim);
    for (long a = 0; a < vocab_size; a++) {
      fprintf(fo_syn0, "%s ", vocab[a].word);
      fwrite(&syn0[a*embed_dim], sizeof(ValT), embed_dim, fo_syn0);
      if (write_syn1) fwrite(&syn1[a*embed_dim], sizeof(ValT), embed_dim, fo_syn1);
      fprintf(fo_syn0, "\n");
    }
    fclose(fo_syn0);
    if (write_syn1) fclose(fo_syn1);

  } else { // write a text file
    ofstream file;
    file.open(output.c_str());

    file << vocab_size << " " << embed_dim << endl;
    for (long a = 0; a < vocab_size; a++) {
      file << vocab[a].word;
      for (long b = 0; b < embed_dim; b++) {
        file << syn0[a*embed_dim+b] << endl;
      }
    }
    file.close();
  }
  sw.stop();
  ADLOG("Current embeddings have been written to '" << output << "' (" << sw << ")");
}

unsigned long long negs_next_random;
// draw one negative sample (i.e., returns one syn1 key, sampled according to a specific distribution)
inline Key DrawNegSample() {
  negs_next_random = negs_next_random * (unsigned long long) 25214903917 + 11;
  unsigned long long target = table[(negs_next_random >> 16) % table_size];
  if (target == 0) {
    target = negs_next_random % (vocab_size - 1) + 1;
  }
  return syn1KeyResolver(target);
}


// A "peekable" random number generator. I.e., this generator allows for
// finding out random numbers that will be generated in the future (by "peeking")
template<typename T>
struct PeekableRandom {
  PeekableRandom(T seed): next_random{seed} {}

  // get the next random number
  T next() {
    // sample new items if necessary
    if (q.size() < 1) {
      sample(1);
    }
    // pop first element
    auto e = q[0];
    q.pop_front();
    return e;
  }

  // peek `num_steps_ahead` into the future
  T peek(size_t num_steps_ahead) {
    if (q.size() <= num_steps_ahead) {
      sample(num_steps_ahead);
    }
    return q[num_steps_ahead];
  }

private:
  // generate a batch of random values
  void sample(size_t min) {
    size_t to_sample = max(min, sample_batch);
    for(size_t i=0; i!=to_sample; ++i) {
      next_random = next_random * (T) 25214903917 + 11;
      q.push_back(next_random);
    }
  }

  // debug output
  std::string print() {
    std::stringstream s;
    for(size_t i=0; i!=q.size(); ++i) {
      s << q[i] << " ";
    }
    return s.str();
  }

  std::deque<T> q; // future random numbers
  T next_random; // random state
  const size_t sample_batch = 100; // minimum size of random numbers to generate
};


//training/computation happens here, equivalent to the original training-thread.
void training_thread(WorkerT &kv, int customer_id, int worker_id) {
  long long a, b, d, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long c, target, label, local_iter = num_iterations;
  PeekableRandom<unsigned long long> context_size_rand((long long) 7+worker_id*13); // rng for determining context size
  unsigned long long next_random_read = (long long) 11+worker_id*17; // rng for frequent word skips
  char eof = 0;


  ValT f, g;
  ADLOG("[w" << worker_id << "] begins work" << endl);

  // allocate memory for local parameters
  vector <ValT> syn0_vec     (embed_dim);
  vector <ValT> syn1neg_vec  (embed_dim);
  vector <ValT> neu1_vec     (embed_dim);
  vector <ValT> neu1e_vec    (embed_dim);
  vector <ValT> syn1neg_push (embed_dim);
  vector <Key>  syn0_key     (1);
  vector <Key>  syn1neg_key  (1);

  // groups of negative samples (one per context)
  std::unordered_map<long long, SampleID> sample_ids {};
  long long num_context = 0;
  long long future_context = 0;

  FILE *fi = fopen(in_file.c_str(), "rb");

  //partitions file for threads
  if (clustered_input) {
    fseek(fi, (file_size / (long long) num_threads) * (long long) (customer_id - 1), SEEK_SET);

  } else {
    auto pos = (file_size / (long long) num_workers) * (long long) worker_id;
    ADLOG("Worker " << worker_id << ": Start at position " << pos << " of " << file_size);
    fseek(fi, pos, SEEK_SET);
  }

  // wait until all workers are ready to start
  kv.Barrier();

  util::Stopwatch sw_epoch;
  util::Stopwatch sw_epoch_all;
  util::Stopwatch sw_train;
  util::Stopwatch sw_worker;
  sw_train.start();
  sw_worker.start();
  sw_epoch.start();
  sw_epoch_all.start();
  util:: Stopwatch sw_wait_syn0;
  util:: Stopwatch sw_wait_syn1_pos;
  util:: Stopwatch sw_find_neg;
  //train loop
  while (1) { //loop ends when thread has reached the end of its partition during its last iteration.

    //adjusts learning rate (alpha)
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count -last_word_count;
      last_word_count = word_count;

      alpha = starting_alpha *
        (1 - (word_count_actual * ps::NumServers()) / (ValT) (num_iterations * train_words + 1));

      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }

    // read next sentence
    if (sentence_length == 0) {
      std::unordered_set<Key> to_localize {};

      while (1) {

        word = ReadWordIndex(fi, &eof);  // gets the position in the vocab; common words come first, uncommon words last

        if (eof) break;
        if (word == -1) continue; // word not in vocab
        word_count++;
        if (word == 0) break;
        // The subsampling randomly discards frequent words while keeping the ranking same
        if (sample > 0) {
          ValT ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
          next_random_read= next_random_read * (unsigned long long) 25214903917 + 11;
          if (ran < (next_random_read & 0xFFFF) / (ValT) 65536) continue;
        }

        // localize parameters for the first couple of words of the sentence
        // (later words will be localized by the context prep)
        if (sentence_length < prep_context_ahead+window) {
          to_localize.insert(syn0KeyResolver(word));
          if (sentence_length < prep_context_ahead) {
            to_localize.insert(syn1KeyResolver(word));
          }
        }

        sen[sentence_length] = word;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;

      // "peek ahead": localize a few words from the beginning of the next sentence
      if (peek_ahead_n != 0 && localize_positives) {
        auto read_pos = ftell(fi); // store current read position (so we can jump back below)
        auto next_random_read_copy = next_random_read; // fork the generator state so we can look into the future
        for (size_t i=0; i!=peek_ahead_n; ++i) {
          long long next_word = ReadWordIndex(fi, &eof);
          // skip conditions (copied from above to achieve identical behavior)
          if (eof) break;
          if (next_word == -1) { --i; continue; }; // word not in vocab
          if (next_word == 0)  { break; }; // sentence end
          if (sample > 0) { // discard frequent words from time to time
            ValT ran = (sqrt(vocab[next_word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[next_word].cn;
            next_random_read_copy = next_random_read_copy * (unsigned long long) 25214903917 + 11;
            if (ran < (next_random_read_copy & 0xFFFF) / (ValT) 65536) {
              --i;
              continue;
            }
          }
          to_localize.insert(syn0KeyResolver(next_word));
          to_localize.insert(syn1KeyResolver(next_word));
        }
        fseek(fi, read_pos, SEEK_SET); // reset read position
      }

      // localize the words that we just read
      if (localize_positives) {
        std::vector<Key> to_localize_vec(to_localize.begin(), to_localize.end());
        kv.Localize(to_localize_vec);
      }
    }


    //finish iteration/epoch when either eof is reached or the # word_count
    if (eof || (word_count > min(train_words,data_words) / num_workers)) {
      word_count_actual += word_count - last_word_count;
      local_iter--;
      auto fepoch = num_iterations - local_iter; //finished epochs

      sw_epoch.stop();
      ADLOG("Worker " << worker_id << " finished epoch " << fepoch << " (" << sw_epoch << ")." );
      kv.Barrier();
      sw_epoch_all.stop();
      sw_train.stop();
      if (worker_id == 0) {
        ADLOG("All workers finished epoch " << fepoch << " (epoch: " << sw_epoch_all << ", total: " << sw_train << ")");
      }

      if (write_results) {
        kv.WaitAll();
        kv.Barrier();
        if (customer_id == 1 && ps::MyRank() == ps::NumServers()-1) {// last rank saves (usually this is the first node)
          util::Stopwatch sw_write; sw_write.start();
          ADLOG("Write epoch " << fepoch << " embeddings");
          write_checkpoint(out_file + ".epoch." + to_string(fepoch), kv);
        }
      }
      kv.Barrier();

      if (local_iter == 0) {
        ADLOG("[w" << worker_id << "] Wait syn0: " << sw_wait_syn0 << "  Wait syn1 positive: " << sw_wait_syn1_pos << "  Find negative: " << sw_find_neg);
        break;
      }

      // maximum time
      if (sw_train.elapsed_s() > max_runtime ||
          sw_train.elapsed_s() + sw_epoch_all.elapsed_s() > max_runtime * 1.05) {
        ADLOG("Worker " << worker_id << " stops after epoch " << fepoch << " because max. time is reached: " << sw_train.elapsed_s() << "s (+1 epoch) > " << max_runtime << "s (epoch: " << sw_epoch_all.elapsed_s() << "s)");
        break;
      }

      sw_epoch.start();
      sw_epoch_all.start();
      sw_train.resume();
      //variable reset for next epoch, worker starts at its determined starting-point again.
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      eof = 0;

      if (clustered_input) {
        fseek(fi, (file_size / (long long) num_threads) * (long long) (customer_id - 1), SEEK_SET);
        continue;
      } else {
        fseek(fi, (file_size / (long long) num_workers) * (long long) worker_id, SEEK_SET); // SEEK_SET begins at file start; computes offset via middle value
        continue;
      }
    }

    word = sen[sentence_position]; //word is extracted here again
    if (word == -1) continue;


    std::fill(neu1_vec.begin(), neu1_vec.end(), 0);
    std::fill(neu1e_vec.begin(), neu1e_vec.end(), 0);

    // prepare a future context
    while (future_context <= num_context + prep_context_ahead) {
      // find out the window size for the future context
      int b_future = context_size_rand.peek(future_context-num_context) % window;

      // prep a negative sample of adequate size for this future context
      sample_ids[future_context] = kv.PrepareSample((window-b_future) * 2 * negative);

      // localize words for this future context
      auto future_position = sentence_position + (future_context-num_context);
      if (localize_positives && future_position < sentence_length) {
        std::vector<Key> to_localize {};
        to_localize.reserve((window-b_future) * 2 + 1);
        to_localize.push_back(syn1KeyResolver(sen[future_position]));
        for (a = b_future; a < window * 2 + 1 - b_future; a++) {
          c = future_position - window + a;
          if (a != window && c >= 0 && c < sentence_length) {
            to_localize.push_back(syn0KeyResolver(sen[c]));
          }
        }
        kv.Localize(to_localize);
      }
      ++future_context;
    }

    // determine window size for the current context
    b = context_size_rand.next() % window;

    // Iterate through one context (number of word pairs = (window-b)*2)
    for (a = b; a < window * 2 + 1 - b; a++) {
      if (a != window) {
        c = sentence_position - window + a;

        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;

        syn0_key[0] = syn0KeyResolver(last_word);
        sw_wait_syn0.resume();
        kv.Wait(kv.Pull(syn0_key, &syn0_vec));
        sw_wait_syn0.stop();


        std::fill(neu1e_vec.begin(), neu1e_vec.end(), 0);


        //negative sampling
        for (d = 0; d < negative + 1; d++) {

          if (d == 0) { // positive (data point)
            label = 1;
            target = word;
            syn1neg_key[0] = syn1KeyResolver(target); // precomputed in preload neg_samples

            sw_wait_syn1_pos.resume();
            kv.Localize(syn1neg_key);
            kv.Wait(kv.Pull(syn1neg_key, &syn1neg_vec));
            sw_wait_syn1_pos.stop();
          } else { // negative sample
            label = 0;

            // Retrieve a negative sample
            sw_find_neg.resume();
            kv.Wait(kv.PullSample(sample_ids[num_context], syn1neg_key, syn1neg_vec));
            target = syn1Reverse(syn1neg_key[0]);
            sw_find_neg.stop();

            if (target == word)
              continue;
          }


          // retrieve output layer of negative-sampled word
          f = 0;
          for (c = 0; c < embed_dim; c++)f += syn0_vec[c] * syn1neg_vec[c];

          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int) ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;


          for (c = 0; c < embed_dim; c++) neu1e_vec[c] += g * syn1neg_vec[c];

          for (c = 0; c < embed_dim; c++) syn1neg_push[c] = g * syn0_vec[c];

          // publish/push updates;  hidden->output
          if (sync_push) {
            kv.Wait(kv.Push(syn1neg_key, syn1neg_push));
          } else {
            kv.Push(syn1neg_key, syn1neg_push);
          }


        }
        // Learn weights input -> hidden
        if (sync_push) {
          kv.Wait(kv.Push(syn0_key, neu1e_vec));
        } else {
          kv.Push(syn0_key, neu1e_vec);
        }
      }
    }
    kv.FinishSample(sample_ids[num_context]);
    sample_ids.erase(num_context);
    sentence_position++;
    num_context++;


    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }

  }


  sw_worker.stop();
  ADLOG("[w" << worker_id << "] finished training (" << sw_worker << "). Processed " << word_count << " words.");

  kv.Barrier();
  sw_train.stop();
  if (worker_id == 0) {
    ADLOG("All workers finished training (" << sw_train << ")");
  }


  kv.WaitAll(); // wait until all requests of this worker are answered
  kv.Barrier(); // make sure all requests in the system have been answered

  fclose(fi);
  ADLOG("[w" << worker_id << "] has passed the training barrier ");
}

// handles loading vocab into memory, found in the original TrainModel()
void load_vocab(int worker_id) {
  if (vocab_retrieve.size() != 0) {
    ReadVocab();

  } else {
    LearnVocabFromTrainFile();

    ADLOG("[w" << worker_id << "] ______ done loading vocab from'" << in_file << "' into memory" << endl);

  }

  if (vocab_save.size() != 0) {
    SaveVocab();
    ADLOG("______Vocab was written to::'" << vocab_save << "'");
  }
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++)
    if (!strcmp(str, argv[a])) {
      if (a == argc - 1) {
        printf("Argument missing for %s\n", str);
        exit(1);
      }
      return a;
    }
  return -1;
}

// Initialize first layer (syn0) of network
void initialize_model(WorkerT &kv) {
  if (init_model.compare("none") == 0) { // no model init
    ALOG("No parameter initialization");
  } else if (init_model.compare("random") == 0) { // init syn0 randomly, don't init syn1
    ALOG("Initialize model randomly (seed " << model_seed << ")");
    vector <ValT> syn_vec(embed_dim * vocab_size);
    vector <Key> syn_key(vocab_size);
    unsigned long long next_random = model_seed;
    for (int e = 0; e < vocab_size; ++e) {
      syn_key[e] = syn0KeyResolver(e);
      for (int j = 0; j < embed_dim; ++j) {
        next_random = next_random * (unsigned long long) 25214903917 + 11; // taken from original w2v
        syn_vec[e * embed_dim + j] =
          (((next_random & 0xFFFF) / (ValT) 65536) - 0.5) / embed_dim;
      }
    }
    kv.Wait(kv.Push(syn_key, syn_vec));
    ALOG("Model init finished: " << syn_vec[0] << " " << syn_vec[1] << " " << syn_vec[2] << " .. ");
  } else { // load syn0 and syn1 from a checkpoint
    ALOG("Initialize model from checkpoint:" << init_model);
    // create keys
    vector<Key> syn0_keys (vocab_size);
    vector<Key> syn1_keys (vocab_size);
    for (long a = 0; a < vocab_size; a++) {
      syn0_keys[a] = syn0KeyResolver(a);
      syn1_keys[a] = syn1KeyResolver(a);
    }

    vector <ValT> syn0 (embed_dim * vocab_size);
    vector<ValT> syn1 (embed_dim * vocab_size);

    // load files
    FILE *fo_syn0;
    FILE *fo_syn1;
    long long read_vocab_size;
    long long read_embed_dim;
    // long read_word;
    char read_word[MAX_STRING];
    fo_syn0 = fopen(init_model.c_str(), "rb");
    fo_syn1 = fopen((init_model+".syn1").c_str(), "rb");
    if (fo_syn0 == NULL || fo_syn1 == NULL) {
      ALOG("Error opening checkpoint files (" << init_model << ")");
      abort();
    }
    fscanf(fo_syn0, "%lld %lld\n", &read_vocab_size, &read_embed_dim);
    assert(read_vocab_size == vocab_size);
    assert(read_embed_dim == embed_dim);

    // read word by word
    for (long a = 0; a < vocab_size; a++) {
      fscanf(fo_syn0, "%s", read_word);
      fgetc(fo_syn0); // read space
      assert(strcmp(read_word, vocab[a].word) == 0);

      fread(&syn0[a*embed_dim], sizeof(ValT), embed_dim, fo_syn0);
      fread(&syn1[a*embed_dim], sizeof(ValT), embed_dim, fo_syn1);
      fgetc(fo_syn0); // read line break
    }
    fclose(fo_syn0);
    fclose(fo_syn1);

    // push parameters into the server
    kv.Wait(kv.Push(syn0_keys, syn0), kv.Push(syn1_keys, syn1));
  }
  kv.WaitReplicaSync();
}

// returns a vector of files inside given directory
void read_directory(const std::string &name, vector <string> &v) {
  DIR *dirp = opendir(name.c_str());
  struct dirent *dp;
  while ((dp = readdir(dirp)) != NULL) {
    v.push_back(dp->d_name);
  }
  closedir(dirp);
}

// returns "file.0x.of.08.txt" -type files(given "file"-pattern) for clustered inputs.
string clustered_ingest(string target_file) {
  vector <string> dirlist;
  string dirname = "";
  string result = "";


  // strips filename to get path to directory
  const size_t last_slash_idx = target_file.find_last_of("\\/");
  if (std::string::npos != last_slash_idx) {
    dirname = target_file.substr(0, last_slash_idx);
    target_file.erase(0, last_slash_idx + 1);
  }
  read_directory(dirname, dirlist);

  string sub_fid;
  string target = "";

  //selects filename which substring matches the given filename pattern

  for (auto file_in_dir: dirlist) {
    sub_fid = file_in_dir.substr(0, target_file.length());
    if (target_file.compare(sub_fid) == 0) {

      target = file_in_dir;
      break;
    }

  }
  dirname = dirname + "/";
  result = dirname + target;
  CHECK(result.length() > target_file.length()) << " no suitable Candidate found, check in " << dirname
                                                << "-directory" << endl;
  return result;
}

// initializes data structures that are shared among worker threads
void init_shared_datastructures () {


  // adapt input file name for reading pre-clustered data (if desired)
  if (clustered_input) {
    string slash = "/";
    std::size_t found = in_file.find(slash);
    CHECK(found != std::string::npos) << " full path needed for clustered input" << endl;
    in_file = clustered_ingest(in_file);
    ADLOG("Clustered Input:: Inputfile will be loaded as:: '" << in_file << "' " << endl);
  }

  // load vocabulary and initialize exponent table
  vocab = (struct vocab_word *) calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *) calloc(vocab_hash_size, sizeof(int));
  expTable = (ValT *) malloc((EXP_TABLE_SIZE + 1) * sizeof(ValT));
  for (int i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (ValT) EXP_TABLE_SIZE * 2 - 1) *
                      MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1); // Precompute f(x) = x / (x + 1)
  }
  // load vocabulary
  load_vocab(0);

  // initialize "shuffling tables"
  // we use this to assign random keys to parameters. this results in more even
  // load on the different servers
  forwards.resize(num_keys / 2 - 2);
  iota(forwards.begin(), forwards.end(), 0);
  unsigned seed = 2;
  shuffle(forwards.begin(), forwards.end(), default_random_engine(seed));
  backwards.resize(forwards.size());
  for(size_t i=0; i!=forwards.size(); ++i) {
    backwards[forwards[i]] = i;
  }

  // sanity checks on maximum key (given as program option)
  unsigned int min_key = vocab_size * 2 + 5, max_key = vocab_size * 2 + 10;
  //  +10 offset kept for possible keys being used as debug flags
  CHECK(num_keys >= min_key && num_keys <= max_key) << "number of keys should be set between " << min_key << " and "
                                                    << max_key << " (vocab size " << vocab_size << ")";

  ADLOG("______Global datastructures have been set up.______" << endl);
}

// sort a collection of pairs by the first element, in descending order
bool sortdesc(const pair<double, Key>& a, const pair<double, Key>& b) {
  return (a.first > b.first);
}


// determine which parameters to replicate.
// we calculate the expected access frequency for each parameter
// and then pick the N with the highest expectations for replication
std::vector<Key> determine_hotspot_keys(size_t N_hotspots) {
  std::vector<Key> to_replicate {};
  if (N_hotspots == 0) return to_replicate;

  if (N_hotspots > vocab_size*2) N_hotspots = vocab_size*2; // cannot replicate more than all keys

  to_replicate.reserve(N_hotspots);

  // calculate the expected access frequency for each parameter
  std::vector<std::pair<double,Key>> expected_frequency (num_keys);
  // calculate total count and pow
  double total_pow = 0;
  long total_cn = 0;
  for (long long a = 0; a != vocab_size; ++a) {
    total_cn += vocab[a].cn;
    total_pow += pow(vocab[a].cn, neg_power);
  }
  for (long long a=0; a != vocab_size; ++a) {
    // for syn0, access frequency depends on data frequency of the corresponding word
    expected_frequency[syn0KeyResolver(a)] = std::make_pair(1.0*vocab[a].cn/total_cn, syn0KeyResolver(a));
    // for syn1, access frequency depends on (1) sampling distribution (majority of accesses) and
    // (2) data frequency
    expected_frequency[syn1KeyResolver(a)] =
      std::make_pair(pow(vocab[a].cn, neg_power)/total_pow*negative+
                     1.0*vocab[a].cn/total_cn, syn1KeyResolver(a));
    // note: over all, parameters of layer syn1 are accessed much more frequently than parameters in syn0.
    // this is reflected here
  }

  // replicate the `N_hotspot` parameters that we expect to be accessed most frequently
  std::sort(expected_frequency.begin(), expected_frequency.end(), sortdesc);
  for (size_t i=0; i != N_hotspots; ++i) {
    // ADLOG("Replicate " << expected_frequency[i].second << ", expected accesses: " << expected_frequency[i].first);
    to_replicate.push_back(expected_frequency[i].second);
  }
  assert(to_replicate.size() == N_hotspots);

  return to_replicate;
}

void RunWorker(int customer_id, ServerT *server = nullptr) {

  Start(customer_id);
  std::unordered_map <std::string, util::Stopwatch> sw{};
  WorkerT kv(0, customer_id, *server);

  int worker_id = ps::MyRank() * num_threads + customer_id - 1; // a unique id for this worker thread

  // initialize the model with random parameter values
  kv.BeginSetup();
  if (customer_id == 1 && ps::MyRank() == ps::NumServers()-1) { // last rank initializes
    initialize_model(kv);
  }
  kv.EndSetup();

  training_thread(kv, customer_id, worker_id);

  kv.Finalize();
  Finalize(customer_id, false);
}

//boost program options
int process_program_options(const int argc, const char *const argv[]) {

  namespace po = boost::program_options;
  po::options_description desc("Allowed options");

  desc.add_options()
    ("help,h", "produce help message")
    ("num_threads,t", po::value<size_t>(&num_threads)->default_value(2),
     "number of worker threads to run (per process)")
    ("num_iterations,i", po::value<size_t>(&num_iterations)->default_value(15), "number of iterations to run")
    ("input_file,f", po::value<string>(&in_file), "name of the training file")
    ("output_file,o", po::value<string>(&out_file)->default_value("vectors.bin"), "output file (to store word vectors)")
    ("vocab_save", po::value<string>(&vocab_save), "name of the resulting vocab-file")
    ("vocab_retrieve", po::value<string>(&vocab_retrieve), "name of the source vocab-file")
    ("shuffle", po::bool_switch(&shuffle_b)->default_value(true), "boolean to scramble words on keys randomly")
    ("debug_mode,d", po::value<int>(&debug_mode)->default_value(2), "disables debug mode")
    ("window,w", po::value<int>(&window)->default_value(5), "adjusts sizing of word-window, default is 5")
    ("embed_dim,v", po::value<long long int>(&embed_dim)->default_value(200),
     "number of values per key; so embed_dim")
    ("num_keys,k", po::value<Key>(&num_keys)->default_value(10), "number of parameters")
    ("negative", po::value<int>(&negative)->default_value(25),
     "number of negative samples per context word pair")
    ("clustered_input", po::bool_switch(&clustered_input)->default_value(false),
     "flag to utilize separate files for each server in a distributed setting")
    ("write_results", po::value<bool>(&write_results)->default_value(false), "write out results")
    ("localize_pos", po::value<bool>(&localize_positives)->default_value(true), "localize contextual data beforehand (default: yes)")
    ("sync_push", po::value<bool>(&sync_push)->default_value(true), "use synchronous pushes? (default: yes)")
    ("data_words", po::value<long long int>(&data_words)->default_value(numeric_limits<long long int>::max()), "use synchronous pushes? (default: yes)")
    ("min_count", po::value<int>(&min_count)->default_value(5), "learn embeddings for all words with count larger than min_cout")
    ("neg_power", po::value<double>(&neg_power)->default_value(0.75), "power for negative sampling")
    ("starting_alpha", po::value<ValT>(&alpha)->default_value(0.025), "starting alpha (learning rate)")
    ("subsample", po::value<ValT>(&sample)->default_value(1e-4), "subsample frequent words")
    ("replicate", po::value<size_t>(&replicate_n)->default_value(0), "number of parameters to replicate")
    ("peek_ahead", po::value<size_t>(&peek_ahead_n)->default_value(0), "peek ahead into the next sentence and localize N words. 0 to disable")
    ("prep_context_ahead", po::value<int>(&prep_context_ahead)->default_value(10), "How many contexts ahead of time to localize positives and prepare negatives")
    ("downsample_replicated_negatives", po::value<bool>(&downsample_replicated_negative_samples)->default_value(false), "downsample replicated negative samples according to the number of servers")
    ("binary", po::value<int>(&binary)->default_value(1), "output in binary human-readable format(default)")
    ("init_model", po::value<string>(&init_model)->default_value("random"), "how to init the parameters. options: 'none', 'random', '[PATH TO CHECKPOINT]'")
    ("model_seed", po::value<unsigned long long>(&model_seed)->default_value(134827), "seed for model generation")
    ("max_runtime", po::value<long>(&max_runtime)->default_value(std::numeric_limits<long>::max()), "set a maximum run tim, after which the job will be terminated (in seconds)")
    ;

  // add system options
  ServerT::AddSystemOptions(desc);

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  // Warnings about nonsensical settings
  if (replicate_n != 0 && SamplingSupport<ValT, WorkerT>::sampling_strategy == SamplingSupportType::OnlyLocal &&
      !downsample_replicated_negative_samples) {
    ALOG("[WARNING] You are combining the 'only local' sampling strategy with replication, but downsampling of replicated parameters is turned off. This might deteriorate the sampling distribution even further.");
  }

  if (vm.count("help")) {
    cout << desc << "\n";
    return 1;
  }
  return 0;
}

int main(int argc, char *argv[]) {
  // Read cmd arguments
  int po_error = process_program_options(argc, argv);
  if (po_error) return 1;

  Postoffice::Get()->enable_dynamic_allocation(num_keys, num_threads);
  std::string role = std::string(getenv("DMLC_ROLE"));

  std::cout << "Word2vec: Starting " << role << ": running " << num_iterations << " iterations on " << num_keys
            << " keys in " << num_threads << " threads\n"
            << "embed_dim " << embed_dim << ", sync_push: " << sync_push << ", min_count: " << min_count << ", neg_power " << neg_power << "\n"
            << "file " << in_file << ", vocab: " << vocab_retrieve << ", data_words " << data_words << "\n"
            << "window " << window << ", negative " << negative << "\n"
            << "localize_pos " << localize_positives << ", " << "replicate_n " << replicate_n << "\n"
            << ReplicaManager<ValT,HandleT>::PrintOptions() << "\n";


  if (role.compare("scheduler") == 0) {
    Start(0);
    Finalize(0, true);
  } else if (role.compare("server") == 0) {

    // initialize data structures that are shared among threads
    init_shared_datastructures();
    starting_alpha = alpha;

    // replication
    vector<Key> hotspot_keys = determine_hotspot_keys(replicate_n);
    // std::sort(hotspot_keys.begin(), hotspot_keys.end());

    // construct unigram table (for efficient negative sampling)
    InitUnigramTable(hotspot_keys);

    // Start the server system
    int server_customer_id = 0; // server gets customer_id=0, workers 1..n
    Start(server_customer_id);
    HandleT handle(num_keys, embed_dim); // the handle specifies how the server handles incoming Push() and Pull() calls
    auto server = new ServerT(server_customer_id, handle, &hotspot_keys);
    RegisterExitCallback([server]() { delete server; });
    num_workers = ps::NumServers() * num_threads;


    // generate vocab out of the clustered files first, else nobody is on the same page
    CHECK(!clustered_input || (clustered_input && vocab_retrieve.size() > 0))
      << "ERROR_________clustered computing with several files requires a pregenerated vocab created out of those files,"
      << endl << " so create one first and relaunch accordingly" << endl;

    // make sure all servers are set up
    server->Barrier();

    // sampling support
    negs_next_random = (long long) 17+ps::MyRank()*123;
    server->enable_sampling_support(&DrawNegSample);

    // run worker(s)
    std::vector <std::thread> workers{};
    for (size_t i = 0; i != num_threads; ++i) {
      workers.push_back(std::thread(RunWorker, i + 1, server));
      std::string name = std::to_string(ps::MyRank())+"-worker-"+std::to_string(ps::MyRank()*num_threads + i);
      SET_THREAD_NAME((&workers[workers.size()-1]), name.c_str());
    }

    // wait for the workers to finish
    for (auto &w : workers)
      w.join();

    // stop the server
    server->shutdown();
    Finalize(server_customer_id, true);
  }
}
