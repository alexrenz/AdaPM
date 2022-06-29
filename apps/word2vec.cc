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

using namespace std;

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

bool enforce_random_keys;
bool enforce_full_replication;
bool write_results;
bool sync_push;
bool clustered_input;
bool signal_intent;
size_t read_sentences_ahead;
ValT *expTable;
double neg_power;
string init_model;
unsigned long long model_seed;
long max_runtime;

//shuffled vector acts as a scramble mapping of words - > keys
vector<ps::Key> forwards; // word -> key
vector<long long> backwards; // key -> word

// number of parameters to replicate


using namespace ps;
using namespace std;


// get the key for the syn0 parameter of a word
inline Key syn0KeyResolver(long long word) {
  if (enforce_random_keys) {
    return (forwards[word] * 2);
  } else {
    return (word * 2);
  }
}
// get the key for the syn1 parameter of a word
inline Key syn1KeyResolver(long long word) {
  if (enforce_random_keys) {
    return (forwards[word] * 2 + 1);
  } else {
    return (word * 2 + 1);
  }
}
// get the word from a syn1 key
inline long long syn1Reverse(Key key) {
  if (enforce_random_keys) {
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
inline double unigram_pow(int a) {
  return pow(vocab[a].cn, neg_power);
}

// construct a table of all negative samples for efficient sampling
void InitUnigramTable() {
  int a, i;
  double train_words_pow = 0;
  double d1;
  table = (int *) malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += unigram_pow(a);
  i = 0;
  d1 = unigram_pow(i) / train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (double) table_size > d1) {
      if (i == 1 || i == 10 || i == 100 || i == 1000 || i == 10000) {
        ALOG(setw(5) << setfill(' ') << i << " most frequent words have " << d1 << " sampling probability");
      }
      i++;
      d1 += unigram_pow(i) / train_words_pow;
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
    auto num_read = fscanf(fin, "%lld%c", &vocab[a].cn, &c);
    CHECK ( num_read != EOF) << "Error: ReadVocab()-fscanf encountered input failure or runtime constraint violation";

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

  vector<ValT> syn0 (embed_dim * 2 * vocab_size);
  vector<ValT> syn1 (embed_dim * 2 * vocab_size);

  kv.WaitSync();
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
      fwrite(&syn0[a*embed_dim*2], sizeof(ValT), embed_dim, fo_syn0); // write without adagrad for now
      if (write_syn1) fwrite(&syn1[a*embed_dim*2], sizeof(ValT), embed_dim, fo_syn1); // write without adagrad for now
      fprintf(fo_syn0, "\n");
    }
    fclose(fo_syn0);
    if (write_syn1) fclose(fo_syn1);

  } else { // write a text file
    std::ofstream file;
    file.open(output.c_str());

    file << vocab_size << " " << embed_dim << endl;
    for (long a = 0; a < vocab_size; a++) {
      file << vocab[a].word;
      for (long b = 0; b < embed_dim; b++) {
        file << syn0[a*embed_dim*2+b] << endl;
      }
    }
    file.close();
  }
  sw.stop();
  ALOG("Current embeddings have been written to '" << output << "' (" << sw << ")");
}

// update step using AdaGrad
template <typename ValT>
void adagrad_update(std::vector<ValT>& val, std::vector<ValT>& push, const ValT alpha) {

  ValT* val_ag = val.data() + embed_dim;
  ValT* push_ag = push.data() + embed_dim;

  for (unsigned i = 0; i != embed_dim; ++i) {
    push_ag[i] = push[i] * push[i];
    push[i] = alpha * push[i] / sqrt(val_ag[i] + push_ag[i]);
  }
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
  long long a, b, d, word, last_word, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, prep_word_count = 0;
  std::vector<std::vector<long long>> sentences (read_sentences_ahead+1);
  size_t cslot = 0;
  long long c, target, label, local_iter = num_iterations;
  PeekableRandom<unsigned long long> context_size_rand((long long) 7+worker_id*13); // rng for determining context size
  unsigned long long next_random_read = (long long) 11+worker_id*17; // rng for frequent word skips
  char eof = 0;


  ValT f, g;

  // allocate memory for local parameters
  vector <ValT> syn0        (embed_dim*2);
  vector <ValT> syn1        (embed_dim*2);
  vector <ValT> syn0_update (embed_dim*2);
  vector <ValT> syn1_update (embed_dim*2);
  vector <Key>  syn0_key     (1);
  vector <Key>  syn1neg_key  (1);

  // groups of negative samples (one per context)
  std::unordered_map<long long, SampleID> sample_ids {};
  long long num_context = 0;
  long long future_context = 0;

  FILE *fi = fopen(in_file.c_str(), "rb");

  //partitions file for threads
  if (clustered_input) {
    fseek(fi, (file_size / (long long) num_threads) * (long long) (customer_id), SEEK_SET);

  } else {
    auto pos = (file_size / (long long) num_workers) * (long long) worker_id;
    ALOG("Worker " << worker_id << ": Start at position " << pos << " of " << file_size);
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
  long current_sentence = 0;
  long future_sentence = 0;

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

    // read new sentence(s)
    // we read a couple of sentences into the future so that we can signal intent
    if (sentences[cslot].size() == 0) {
      // we read `read_sentences_ahead` into the future
      while (future_sentence <= current_sentence + static_cast<long>(read_sentences_ahead)) {
        ++future_sentence;
        size_t fslot = future_sentence % sentences.size();
        std::unordered_set<Key> intent {};
        size_t num_samples = 0;

        // read one sentence, word by word
        while (sentences[fslot].size() < MAX_SENTENCE_LENGTH) {
          word = ReadWordIndex(fi, &eof);
          if (eof) break; // end of file
          if (word == -1) continue; // word not in vocab
          prep_word_count++;
          if (word == 0) break; // end of sentence

          // skip frequent words from time to time
          if (sample > 0) {
            ValT ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
            next_random_read= next_random_read * (unsigned long long) 25214903917 + 11;
            if (ran < (next_random_read & 0xFFFF) / (ValT) 65536) continue;
          }

          // add word to the future sentence
          sentences[fslot].push_back(word);

          // intent: we will access syn0 and syn1 parameters of this word
          if (signal_intent) {
            intent.insert(syn0KeyResolver(word));
            intent.insert(syn1KeyResolver(word));
          }

          // estimate the number of samples for this word's context (upper bound)
          int b_future = context_size_rand.peek(future_context-num_context) % window; // window size for the future context
          num_samples += (window-b_future) * 2 * negative;
          ++future_context;
        }

        // prepare a sample of adequate size and signal intent for this sentence
        auto futureClock = kv.currentClock()+future_sentence-current_sentence;
        sample_ids[future_sentence] = kv.PrepareSample(num_samples, futureClock);
        if (signal_intent) {
          kv.Intent(std::move(intent), futureClock);
        }

        // end of epoch: move read position back to the start position
        if (eof || (prep_word_count > min(train_words,data_words) / num_workers)) {
          prep_word_count = 0;
          eof = 0;
          if (clustered_input) {
            fseek(fi, (file_size / (long long) num_threads) * (long long) (customer_id), SEEK_SET);
          } else {
            fseek(fi, (file_size / (long long) num_workers) * (long long) worker_id, SEEK_SET);
          }
        }
      }

      // start a new sentence
      ++current_sentence;
      sentence_position = 0;
      cslot = current_sentence % sentences.size();
      word_count += sentences[cslot].size() + 1;
      kv.advanceClock();

      // end of an epoch
      if (word_count > min(train_words,data_words) / num_workers) {
        word_count_actual += word_count - last_word_count;
        local_iter--;
        auto fepoch = num_iterations - local_iter; //finished epochs

        sw_epoch.stop();
        ALOG("Worker " << worker_id << " finished epoch " << fepoch << " (" << sw_epoch << "). [sentence " << current_sentence << "]" );
        kv.Barrier();
        sw_epoch_all.stop();
        sw_train.stop();
        if (worker_id == 0) {
          ALOG("All workers finished epoch " << fepoch << " (epoch: " << sw_epoch_all << ", total: " << sw_train << ")");
        }

        if (write_results) {
          kv.WaitAll();
          kv.WaitSync();
          kv.Barrier();
          if (customer_id == 0 && ps::MyRank() == ps::NumServers()-1) {// last rank saves (usually this is the first node)
            util::Stopwatch sw_write; sw_write.start();
            ALOG("Write epoch " << fepoch << " embeddings");
            write_checkpoint(out_file + ".epoch." + to_string(fepoch), kv);
          }
        }
        kv.Barrier();

        if (local_iter == 0) {
          break;
        }

        // maximum time
        if (sw_train.elapsed_s() > max_runtime ||
            sw_train.elapsed_s() + sw_epoch_all.elapsed_s() > max_runtime * 1.05) {
          ALOG("Worker " << worker_id << " stops after epoch " << fepoch << " because max. time is reached: " << sw_train.elapsed_s() << "s (+1 epoch) > " << max_runtime << "s (epoch: " << sw_epoch_all.elapsed_s() << "s)");
          break;
        }

        // reset word counts for new epoch
        word_count = 0;
        last_word_count = 0;
        sw_epoch.start();
        sw_epoch_all.start();
        sw_train.resume();
      }
    }

    if (sentences[cslot].size() == 0) continue; // skip empty sentences
    word = sentences[cslot][sentence_position];
    if (word == -1) continue;

    // determine window size for the current context
    b = context_size_rand.next() % window;

    // Iterate through one context (number of word pairs = (window-b)*2)
    for (a = b; a < window * 2 + 1 - b; a++) {
      if (a != window) {
        c = sentence_position - window + a;

        if (c < 0) continue;
        if (c >= static_cast<long long>(sentences[cslot].size())) continue;
        last_word = sentences[cslot][c];
        if (last_word == -1) continue;

        syn0_key[0] = syn0KeyResolver(last_word);
        kv.Wait(kv.Pull(syn0_key, &syn0));

        std::fill(syn0_update.begin(), syn0_update.end(), 0);

        //negative sampling
        for (d = 0; d < negative + 1; d++) {

          if (d == 0) { // positive (data point)
            label = 1;
            target = word;
            syn1neg_key[0] = syn1KeyResolver(target); // precomputed in preload neg_samples

            kv.Wait(kv.Pull(syn1neg_key, &syn1));
          } else { // negative sample
            label = 0;

            // Retrieve a negative sample
            kv.Wait(kv.PullSample(sample_ids[current_sentence], syn1neg_key, syn1));
            target = syn1Reverse(syn1neg_key[0]);

            if (target == word)
              continue;
          }


          // retrieve output layer of negative-sampled word
          f = 0;
          for (c = 0; c < embed_dim; c++)f += syn0[c] * syn1[c];

          if (f > MAX_EXP) g = (label - 1);
          else if (f < -MAX_EXP) g = (label - 0);
          else g = (label - expTable[(int) ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]);

          for (c = 0; c < embed_dim; c++) syn0_update[c] += g * syn1[c];
          for (c = 0; c < embed_dim; c++) syn1_update[c] = g * syn0[c];

          adagrad_update(syn1, syn1_update, alpha);

          // publish/push updates;  hidden->output
          if (sync_push) {
            kv.Wait(kv.Push(syn1neg_key, syn1_update));
          } else {
            kv.Push(syn1neg_key, syn1_update);
          }
        }
        // Learn weights input -> hidden
        adagrad_update(syn0, syn0_update, alpha);
        if (sync_push) {
          kv.Wait(kv.Push(syn0_key, syn0_update));
        } else {
          kv.Push(syn0_key, syn0_update);
        }
      }
    }
    sentence_position++;
    num_context++;

    // end of sentence
    if (sentence_position >= static_cast<long long>(sentences[cslot].size())) {
      sentences[cslot].clear();
      kv.FinishSample(sample_ids[current_sentence]);
      sample_ids.erase(current_sentence);
      continue;
    }
  }


  sw_worker.stop();
  ALOG("[w" << worker_id << "] finished training (" << sw_worker << "). Processed " << word_count << " words.");

  kv.Barrier();
  sw_train.stop();
  if (worker_id == 0) {
    ALOG("All workers finished training (" << sw_train << ")");
  }


  kv.WaitAll(); // wait until all requests of this worker are answered
  kv.Barrier(); // make sure all requests in the system have been answered

  fclose(fi);
}

// handles loading vocab into memory, found in the original TrainModel()
void load_vocab(int worker_id) {
  if (vocab_retrieve.size() != 0) {
    ReadVocab();

  } else {
    LearnVocabFromTrainFile();

  }

  if (vocab_save.size() != 0) {
    SaveVocab();
    ALOG("Vocab was written to '" << vocab_save << "'");
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
    vector <ValT> syn0(embed_dim * 2 * vocab_size);
    vector <Key> syn0_key(vocab_size);
    vector <ValT> syn1(embed_dim * 2 * vocab_size, 0); // we initialize only the AdaGrad values in syn1 (see below)
    vector <Key> syn1_key(vocab_size);
    unsigned long long next_random = model_seed;
    for (int e = 0; e < vocab_size; ++e) {
      syn0_key[e] = syn0KeyResolver(e);
      syn1_key[e] = syn1KeyResolver(e);
      for (int j = 0; j < embed_dim; ++j) {
        next_random = next_random * (unsigned long long) 25214903917 + 11; // taken from original w2v
        syn0[e * embed_dim * 2 + j] =
          (((next_random & 0xFFFF) / (ValT) 65536) - 0.5) / embed_dim;

        // adagrad init
        syn0[e * embed_dim * 2 + embed_dim + j] = 1e-6;
        syn1[e * embed_dim * 2 + embed_dim + j] = 1e-6;
      }
    }
    kv.Wait(kv.Push(syn0_key, syn0));
    kv.Wait(kv.Push(syn1_key, syn1));
    ALOG("Model init finished: " << syn0[0] << " " << syn0[1] << " " << syn0[2] << " .. ");
  } else { // load syn0 and syn1 from a checkpoint
    ALOG("Initialize model from checkpoint:" << init_model);
    ALOG("Not implemented: initialize from checkpoint (does not support AdaGrad yet)");
    abort();
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
    auto num_read = fscanf(fo_syn0, "%lld %lld\n", &read_vocab_size, &read_embed_dim);
    CHECK ( num_read != EOF) << "Error: initialize_model()-fscanf encountered input failure or runtime constraint violation";

    assert(read_vocab_size == vocab_size);
    assert(read_embed_dim == embed_dim);

    // read word by word
    for (long a = 0; a < vocab_size; a++) {
      auto num_read = fscanf(fo_syn0, "%s", read_word);
      CHECK ( num_read != EOF) << "Error: initialize_model()-fscanf encountered input failure or runtime constraint violation";
      fgetc(fo_syn0); // read space
      assert(strcmp(read_word, vocab[a].word) == 0);

      auto syn0_num_read = fread(&syn0[a*embed_dim], sizeof(ValT), embed_dim, fo_syn0);
      auto syn1_num_read = fread(&syn1[a*embed_dim], sizeof(ValT), embed_dim, fo_syn1);

      CHECK (embed_dim == static_cast<long long int>(syn0_num_read) && syn0_num_read == syn1_num_read) << "Error: Number of objects read successfully differs from number of objects "
                                                                                   "intended to read. \n Intended to read " << embed_dim << " syn0-read: "
                                                                                   << syn0_num_read << ", syn1-read: " << syn1_num_read ;

      fgetc(fo_syn0); // read line break
    }
    fclose(fo_syn0);
    fclose(fo_syn1);

    // push parameters into the server
    kv.Wait(kv.Push(syn0_keys, syn0), kv.Push(syn1_keys, syn1));
  }
  kv.WaitSync();
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
    ALOG("Clustered Input: Inputfile will be loaded as:: '" << in_file << "' " << endl);
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

}

// sort a collection of pairs by the first element, in descending order
bool sortdesc(const pair<double, Key>& a, const pair<double, Key>& b) {
  return (a.first > b.first);
}


void RunWorker(int customer_id, ServerT *server = nullptr) {

  std::unordered_map <std::string, util::Stopwatch> sw{};
  WorkerT kv(customer_id, *server);

  int worker_id = ps::MyRank() * num_threads + customer_id; // a unique id for this worker thread

  // initialize the model with random parameter values
  kv.BeginSetup();
  if (customer_id == 0 && ps::MyRank() == ps::NumServers()-1) { // last rank initializes
    initialize_model(kv);
  }
  kv.EndSetup();

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

  training_thread(kv, customer_id, worker_id);

  kv.Finalize();
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
    ("enforce_random_keys", po::value<bool>(&enforce_random_keys)->default_value(false), "enforce that keys are assigned randomly")
    ("enforce_full_replication", po::value<bool>(&enforce_full_replication)->default_value(false), "manually enforce full model replication")
    ("debug_mode,d", po::value<int>(&debug_mode)->default_value(2), "disables debug mode")
    ("window,w", po::value<int>(&window)->default_value(5), "adjusts sizing of word-window, default is 5")
    ("embed_dim,v", po::value<long long int>(&embed_dim)->default_value(200),
     "number of values per key; so embed_dim")
    ("num_keys,k", po::value<Key>(&num_keys)->default_value(10), "number of parameters")
    ("negative", po::value<int>(&negative)->default_value(25),
     "number of negative samples per context word pair")
    ("clustered_input", po::value<bool>(&clustered_input)->default_value(false),
     "flag to utilize separate files for each server in a distributed setting")
    ("write_results", po::value<bool>(&write_results)->default_value(false), "write out results")
    ("sync_push", po::value<bool>(&sync_push)->default_value(false), "use synchronous pushes? (default: no)")
    ("data_words", po::value<long long int>(&data_words)->default_value(numeric_limits<long long int>::max()), "use synchronous pushes? (default: yes)")
    ("min_count", po::value<int>(&min_count)->default_value(5), "learn embeddings for all words with count larger than min_cout")
    ("neg_power", po::value<double>(&neg_power)->default_value(0.75), "power for negative sampling")
    ("starting_alpha", po::value<ValT>(&alpha)->default_value(0.025), "starting alpha (learning rate)")
    ("subsample", po::value<ValT>(&sample)->default_value(1e-4), "subsample frequent words")
    ("read_sentences_ahead", po::value<size_t>(&read_sentences_ahead)->default_value(1000), "How far to read ahead? This is relevant for intent signaling. 0: do not read ahead, N>0: read N sentences ahead")
    ("signal_intent", po::value<bool>(&signal_intent)->default_value(true), "whether to signal intent (default: yes)")
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

  Setup(num_keys, num_threads);
  std::string role = std::string(getenv("DMLC_ROLE"));

  ALOG("Word2vec: Starting " << role << ": running " << num_iterations << " iterations on " << num_keys
       << " keys in " << num_threads << " threads,\n"
       << "embed_dim " << embed_dim << ", sync_push: " << sync_push << ", min_count: " << min_count << ", neg_power " << neg_power << ",\n"
       << "file " << in_file << ", vocab: " << vocab_retrieve << ", data_words " << data_words << ",\n"
       << "window " << window << ", negative " << negative << ","
       << "read ahead " << read_sentences_ahead << ", signal_intent " << signal_intent << "\n"
       << (SyncManager<ValT,HandleT>::PrintOptions()));


  if (role.compare("scheduler") == 0) {
    Scheduler();
  } else if (role.compare("server") == 0) {

    // initialize data structures that are shared among threads
    init_shared_datastructures();
    starting_alpha = alpha;

    // construct unigram table (for efficient negative sampling)
    InitUnigramTable();

    // Start the server system
    auto server = new ServerT(embed_dim*2);
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
      workers.push_back(std::thread(RunWorker, i, server));
      std::string name = std::to_string(ps::MyRank())+"-worker-"+std::to_string(ps::MyRank()*num_threads + i);
      SET_THREAD_NAME((&workers[workers.size()-1]), name.c_str());
    }

    // wait for the workers to finish
    for (auto &w : workers)
      w.join();

    // stop the server
    server->shutdown();
  }
}
