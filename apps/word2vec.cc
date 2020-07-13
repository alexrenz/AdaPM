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
int binary = 0, debug_mode = 2, window = 5, min_reduce = 1, min_count,hs = 0, negative = 25,localize_next;
int *vocab_hash,*table;
long long vocab_max_size = 1000, vocab_size = 0;
long long train_words = 0, word_count_actual = 0, file_size = 0;
ValT alpha = 0.025, starting_alpha, sample;
int negative_list_size = 1000;
long long int embed_dim,data_words;

bool shuffle_b, write_results, sync_push,clustered_input = false, only_local_negatives, localize_positives, localize_negatives;
ValT *expTable;
double neg_power;

//shuffled vector acts as a scramble mapping of words - > keys
vector<unsigned long long> forwards;


using namespace ps;
using namespace std;


//syn0 and syn1neg alternate on the actual keyspace(to uniformly distribute key access over keyspace ); these functions calculate the offset/location of a given word.
inline Key syn0KeyResolver(long long word) {

  if (shuffle_b)return (forwards[word] * 2);

  return (word * 2);
}
inline Key syn1KeyResolver(long long word) {

  if (shuffle_b) return (forwards[word] * 2 + 1);

  return (word * 2 + 1);
}


typedef DefaultColoServerHandle <ValT> HandleT;
typedef ColoKVServer <ValT, HandleT> ServerT;
typedef ColoKVWorker <ValT, HandleT> WorkerT;

// Config
uint num_workers = -1;
size_t num_iterations = 0;
size_t num_threads = 0;
Key num_keys = 0;

// words are are put on randomized keys as apposed to "word 1 having key 1"; the function below creates this mapping
void Init_shuffling_maps() {

  forwards.resize(num_keys / 2 - 2);
  iota(forwards.begin(), forwards.end(), 1);

  unsigned seed = 2;
  shuffle(forwards.begin(), forwards.end(), default_random_engine(seed));


}

void InitUnigramTable() {
  int a, i;
  double train_words_pow = 0;
  double d1;
  table = (int *) malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, neg_power);
  i = 0;
  d1 = pow(vocab[i].cn, neg_power) / train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (double) table_size > d1) {
      if (i == 1 || i == 10 || i == 100 || i == 1000 || i == 10000) {
        ADLOG(setw(5) << setfill(' ') << i << " most frequent words have " << d1 << " sampling probability");
      }
      i++;
      d1 += pow(vocab[i].cn, neg_power) / train_words_pow;
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
void write_current_embeddings(string output, WorkerT &kv) {
  long a;
  vector <ValT> syn_vec(embed_dim);
  vector <Key> syn_key(1);
  util::Stopwatch sw; sw.start();

  // localize entire syn0 for faster write in distributed setting
  vector<Key> keys{};
  for (a = 0; a < vocab_size; a++) {
    keys.push_back(syn0KeyResolver(a));
  }
  kv.Localize(keys);

  if (binary == 1) { //adopted from the original implementation
    FILE *fo;
    fo = fopen(output.c_str(), "wb");
    long b;

    fprintf(fo, "%lld %lld\n", vocab_size, embed_dim);
    for (a = 0; a < vocab_size; a++) {
      fprintf(fo, "%s ", vocab[a].word);
      syn_key[0] = syn0KeyResolver(a);
      kv.Wait(kv.Pull(syn_key, &syn_vec));

      for (b = 0; b < embed_dim; b++) fwrite(&syn_vec[b], sizeof(ValT), 1, fo);
      fprintf(fo, "\n");
    }
    fclose(fo);

  } else {
    // write human-readable file
    ofstream file;
    file.open(output.c_str());

    file << vocab_size << " " << embed_dim << endl;
    for (a = 0; a < vocab_size; a++) {

      file << vocab[a].word;
      syn_key[0] = syn0KeyResolver(a);
      kv.Wait(kv.Pull(syn_key, &syn_vec));
      file << syn_vec << endl;
    }
    file.close();
  }
  sw.stop();
  ADLOG("Current embeddings have been written to '" << output << "' (" << sw << ")");
}

// localize the parameters for a given word
inline void preload_word(WorkerT &kv, long long &word) { //
  vector <Key> keys {syn0KeyResolver(word), syn1KeyResolver(word)};
  kv.Localize(keys);
}

// Generates a fresh list of negative samples
unsigned long long generate_negative_samples(vector <long long> &vec, vector <Key> &to_localize, unsigned long long next_random) {
  unsigned long long target = 0;
  vec.resize(negative_list_size);
  to_localize.resize(negative_list_size);
  for (unsigned int e = 0; e < vec.size(); e++) {
    next_random = next_random * (unsigned long long) 25214903917 + 11;
    target = table[(next_random >> 16) % table_size];
    if (target == 0) target = next_random % (vocab_size - 1) + 1;
    vec[e] = target;
    to_localize[e] = syn1KeyResolver(target);
  }
  // localize negative samples onto this node exactly once
  std::sort(to_localize.begin(), to_localize.end());
  to_localize.erase(unique(to_localize.begin(), to_localize.end()), to_localize.end());
  return next_random;
}

//training/computation happens here, equivalent to the original training-thread.
void training_thread(WorkerT &kv, int customer_id, int worker_id) {
  long long a, b, d, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long c, target, label, local_iter = num_iterations;
  unsigned long long next_random = (long long) worker_id;
  char eof = 0;
  int listpos = 0;
  bool have_neg_s = false;


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

  // for negative sample list
  vector <long long> current_list (negative_list_size);
  vector <Key>       to_localize  (negative_list_size);
  vector <long long> next_list    (negative_list_size);
  long long neg_checks = 0;
  long long neg_gone   = 0;
  long long num_neg_lists = 0;

  // generate and localize a list of negative samples
  next_random = generate_negative_samples(current_list, to_localize, next_random);
  if(localize_negatives)kv.Wait(kv.Localize(to_localize));

  FILE *fi = fopen(in_file.c_str(), "rb");

  //partitions file for threads
  if (clustered_input) {
    fseek(fi, (file_size / (long long) num_threads) * (long long) (customer_id - 1), SEEK_SET);

  } else {
    auto pos = (file_size / (long long) num_workers) * (long long) worker_id;
    ADLOG("Worker " << worker_id << ": Start at position " << pos << " of " << file_size);
    fseek(fi, pos, SEEK_SET);
  }


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

    //builds sentence
    if (sentence_length == 0) {

      while (1) {

        word = ReadWordIndex(fi, &eof);  // gets the position in the vocab; common words come first, uncommon words last

        if (eof) break;
        if (word == -1) continue; // word not in vocab
        word_count++;
        if (word == 0) break;
        // The subsampling randomly discards frequent words while keeping the ranking same
        if (sample > 0) {
          ValT ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
          next_random = next_random * (unsigned long long) 25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (ValT) 65536) continue;
        }

        // localizes the syn0-portion
        if(localize_positives)preload_word(kv, word);

        sen[sentence_length] = word;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }


    //finish iteration/epoch when either eof is reached or the # word_count
    if (eof || (word_count > min(train_words,data_words) / num_workers)) {
      word_count_actual += word_count - last_word_count;
      local_iter--;
      auto fepoch = num_iterations - local_iter; //finished epochs

      sw_epoch.stop();
      ADLOG("Worker " << worker_id << " finished epoch " << fepoch << " (" << sw_epoch << "). Negative lists: " << num_neg_lists << " (" << sw_epoch.elapsed_us() / num_neg_lists << " per list)" );
      kv.Barrier();
      sw_epoch_all.stop();
      if (worker_id == 0) {
        ADLOG("All workers finished epoch " << fepoch << " (" << sw_epoch_all << ")");
      }

      if (write_results) {
        kv.WaitAll();
        kv.Barrier();
        if (customer_id == 1 && ps::MyRank() == ps::NumServers()-1) {// last rank saves (usually this is the first node)
          util::Stopwatch sw_write; sw_write.start();
          ADLOG("Write epoch " << fepoch << " embeddings");
          write_current_embeddings(out_file + ".epoch." + to_string(fepoch), kv);
        }
      }
      kv.Barrier();
      if (worker_id == 0) ADLOG("");

      if (local_iter == 0) {
        ADLOG("[w" << worker_id << "] Gone: " << neg_gone << " / " << neg_checks << " (" << 1.0 * neg_gone / neg_checks << ")");
        ADLOG("[w" << worker_id << "] Wait syn0: " << sw_wait_syn0 << "  Wait syn1 positive: " << sw_wait_syn1_pos << "  Find negative: " << sw_find_neg);
        break;
      }
      sw_epoch.start();
      sw_epoch_all.start();
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


    next_random = next_random * (unsigned long long) 25214903917 + 11;
    b = next_random % window;

    // training by iterating through context.
    for (a = b; a < window * 2 + 1 - b; a++)
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

          if (d == 0) {
            label = 1;
            target = word;
            syn1neg_key[0] = syn1KeyResolver(target); // precomputed in preload neg_samples

            sw_wait_syn1_pos.resume();
            kv.Wait(kv.Pull(syn1neg_key, &syn1neg_vec));
            sw_wait_syn1_pos.stop();
          } else {
            label = 0;

            // Retrieve a negative sample
            sw_find_neg.resume();
            have_neg_s = false;
            while (!have_neg_s) {
              ++neg_checks;

              target = current_list[listpos];
              syn1neg_key[0] = syn1KeyResolver(target);

              // try to retrieve the parameter for this negative sample
              if (only_local_negatives) {
                have_neg_s = kv.PullIfLocal(syn1neg_key[0], &syn1neg_vec);
              } else { //normal pull
                have_neg_s = true;
                kv.Wait(kv.Pull(syn1neg_key, &syn1neg_vec));
              }

              //TODO git diff zum master auf word2vec.cc beschraenken

              if (!have_neg_s) {
                // this negative sample is not local anymore. use another one and don't try this one again in future passes
                ++neg_gone;
              }
              // move on to the next negative sample
              listpos++;

              // end of negative sample list: retrieve new negatives
              if (listpos == negative_list_size) {
                ++num_neg_lists;
                listpos = 0;
                std::swap(current_list,next_list);
              }

              // if we reached localize_next, we dispatch request for new samples to
              if (listpos == localize_next) {
                next_random = generate_negative_samples(next_list, to_localize, next_random);
                if (localize_negatives)kv.Localize(to_localize);
              }
            }
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
    sentence_position++;


    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }

  }


  sw_worker.stop();
  ADLOG("[w" << worker_id << "] finished training (" << sw_worker << "). Processed " << word_count << " words. \n");

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
void init_keyvals(WorkerT &kv) {

  vector <ValT> syn_vec(embed_dim * vocab_size);
  vector <Key> syn_key(vocab_size);
  unsigned long long next_random = 1;

  for (int e = 0; e < vocab_size; ++e) {
    // syn0 on even keys
    syn_key[e] = 2 * e;
    for (int j = 0; j < embed_dim; ++j) {
      next_random = next_random * (unsigned long long) 25214903917 + 11; // taken from original w2v
      syn_vec[e * embed_dim + j] =
        (((next_random & 0xFFFF) / (ValT) 65536) - 0.5) / embed_dim;
    }

  }
  kv.Push(syn_key, syn_vec);
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

//initializes datastructures found in original w2v-code
void initial_w2v_setup(WorkerT &kv, int worker_id) {

  if (clustered_input) { //full paths containing slashes are required for clustered inputs

    string slash = "/";
    std::size_t found = in_file.find(slash);
    CHECK(found != std::string::npos) << " full path needed for clustered input" << endl;
    in_file = clustered_ingest(in_file);

    ADLOG("[w" << worker_id << "] Clustered Input:: Inputfile will be loaded as:: '" << in_file << "' " << endl);

  }


  vocab = (struct vocab_word *) calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *) calloc(vocab_hash_size, sizeof(int));
  expTable = (ValT *) malloc((EXP_TABLE_SIZE + 1) * sizeof(ValT));
  for (int i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (ValT) EXP_TABLE_SIZE * 2 - 1) *
                      MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1); // Precompute f(x) = x / (x + 1)
  }

  load_vocab(0);

  starting_alpha = alpha;
  InitUnigramTable();
  // creates shuffling tables.
  Init_shuffling_maps();

  unsigned int min_key = vocab_size * 2 + 5, max_key = vocab_size * 2 + 10;
  //  +10 offset kept for possible keys being used as debug flags
  CHECK(num_keys >= min_key && num_keys <= max_key) << "number of keys should be set between " << min_key << " and "
                                                    << max_key << " (vocab size " << vocab_size << ")";


  ADLOG("[w" << worker_id << "]" << "______Global datastructures have been set up.______" << endl);
}

void RunWorker(int customer_id, ServerT *server = nullptr) {

  Start(customer_id);
  std::unordered_map <std::string, util::Stopwatch> sw{};
  WorkerT kv(0, customer_id, *server);

  int worker_id = ps::MyRank() * num_threads + customer_id - 1; // a unique id for this worker thread

  if (customer_id == 1) {
    //first worker in each node creates all needed datastructures for their node
    initial_w2v_setup(kv, worker_id);
  }

  if (worker_id == 0) {// sets up initial parameter-server key values
    init_keyvals(kv);
    ADLOG("Shuffled keys are turned on:: " << shuffle_b);
  }

  // halts every thread to ensure global-datastructure existence
  kv.ResetStats();
  kv.Barrier();

  training_thread(kv, customer_id, worker_id);

  // make sure all workers finished
  kv.Barrier();

  if (customer_id != 0) {
    Finalize(customer_id, false); // if this is not the main thread, we shut down the system for this thread here
  }
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
     "negative-sampling parameter, # of negative sampled words for each context")
    ("nls", po::value<int>(&negative_list_size)->default_value(1000),
     "negative-sampling list size, # of negative sampled words to localize at once ")
    ("clustered_input", po::bool_switch(&clustered_input)->default_value(false),
     "flag to utilize separate files for each server in a distributed setting")
    ("only_local_neg", po::value<bool>(&only_local_negatives)->default_value(false), "toggle to use neg samples which are localized on that node and skip those who are not. ")
    ("write_results", po::value<bool>(&write_results)->default_value(false), "write out results")
    ("localize_pos", po::value<bool>(&localize_positives)->default_value(true), "localize contextual data beforehand (default: yes)")
    ("localize_neg", po::value<bool>(&localize_negatives)->default_value(true), "localize negative samples beforehand (default: yes)")
    ("sync_push", po::value<bool>(&sync_push)->default_value(true), "use synchronous pushes? (default: yes)")
    ("data_words", po::value<long long int>(&data_words)->default_value(numeric_limits<long long int>::max()), "use synchronous pushes? (default: yes)")
    ("min_count", po::value<int>(&min_count)->default_value(5), "learn embeddings for all words with count larger than min_cout")
    ("neg_power", po::value<double>(&neg_power)->default_value(0.75), "power for negative sampling")
    ("subsample", po::value<ValT>(&sample)->default_value(1e-4), "subsample frequent words")
    ("localize_next",po::value<int>(&localize_next)->default_value(0)," determines the start of localization for the next negative sampling list ")
    ("binary", po::value<int>(&binary)->default_value(0), "output in binary human-readable format(default)");


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

  Postoffice::Get()->enable_dynamic_allocation(num_keys, num_threads);
  std::string role = std::string(getenv("DMLC_ROLE"));

  std::cout << "Word2vec: Starting " << role << ": running " << num_iterations << " iterations on " << num_keys
            << " keys in " << num_threads << " threads\n"
            << "embed_dim: " << embed_dim << ", sync_push: " << sync_push << ", min_count: " << min_count << "\n";


  if (role.compare("scheduler") == 0) {
    Start(0);
    Finalize(0, true);
  } else if (role.compare("server") == 0) {

    // Start the server system
    int server_customer_id = 0; // server gets customer_id=0, workers 1..n
    Start(server_customer_id);
    HandleT handle(num_keys, embed_dim); // the handle specifies how the server handles incoming Push() and Pull() calls
    auto server = new ServerT(server_customer_id, handle);
    RegisterExitCallback([server]() { delete server; });

    num_workers = ps::NumServers() * num_threads;
    // run worker(s)
    std::vector <std::thread> workers{};

    // generate vocab out of the clustered files first, else nobody is on the same page
    CHECK(!clustered_input || (clustered_input && vocab_retrieve.size() > 0))
      << "ERROR_________clustered computing with several files requires a pregenerated vocab created out of those files,"
      << endl << " so create one first and relaunch accordingly" << endl;

    // localize_next-trigger indice needs to be located inside nls
    CHECK(negative_list_size > localize_next) << "ERROR___________localize_next-trigger (" << localize_next <<") indice needs to be located inside nls("<< negative_list_size<<"); " << endl << "therefore adjust localize_next hyperparameter to be smaller than nls (negative list size)" << endl;

    for (size_t i = 0; i != num_threads; ++i)
      workers.push_back(std::thread(RunWorker, i + 1, server));


    // wait for the workers to finish
    for (auto &w : workers)
      w.join();

    // stop the server
    Finalize(server_customer_id, true);
  }
}
