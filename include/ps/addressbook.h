/**
 *  Copyright (c) 2018 by Contributors
 */
#ifndef PS_ADDRESS_BOOK_
#define PS_ADDRESS_BOOK_

#include "ps/base.h"
#include <iostream>
#include <sstream>
#include <cassert>

typedef short OwnershipT;

#define NOT_CACHED -1

namespace ps {

  /**
   * \brief Address book: manages the ownership of keys
   *
   * This class implements a simple way to manage responsibilities:
   * - For each key, there is two roles: a manager and an owner.
   * - The owner holds the current value for the key, the manager knows which node is the owner.
   * - To assign a manager to each key, we range partition the parameter space.
   *   A node is the manager for all keys in its range.
   * - The manager stores who is the owner for each of its keys.
   *
   * Optionally, it caches the locations of keys. Responses to pull requests and outgoing pins update the cache
   */
  class Addressbook {
  public:

    Addressbook(): ranges(po.GetServerKeyRanges()) {
      ownership = std::vector<OwnershipT>(myRange().size(), po.my_rank());

      if (Postoffice::Get()->use_location_caches()) {
        locationCache = std::vector<OwnershipT>(Postoffice::Get()->max_key(), NOT_CACHED);
        useCache = true;
      }
    }

    /**
     * \brief get directions for a given key
     *
     * 1) returns the current owner of the given key if this PS is the mgr for the given key
     * 2) otherwise, (if desired with `checkCache`) checks whether the location of this key
          is cached, returns the cached location if it is
     * 3) otherwise, returns the manager for the key
     */
    unsigned int getDirections(Key key, const bool checkCache = false) {
      // assert that the key is not local (should be checked before)

      if(isManagedHere(key)) {
        return ownerOfKey(key);
      } else {
        if (checkCache && useCache) {
          auto locCache = getCache(key);
          if (locCache != NOT_CACHED) {
            return locationCache[key];
          }
        }
        return getManager(key);
      }
    }

    /**
     * \brief update the ownership for a key that is managed by this PS
     *        Returns the previous owner.
     */
    OwnershipT updateResidence(Key key, OwnershipT new_owner) {
      CHECK(isManagedHere(key)); // can update ownership only for the parameters that this PS manages
      CHECK(new_owner < po.num_servers()); // new_owner has to be a valid node

      std::lock_guard<std::mutex> lck(mu_);
      return updateResidenceUnsafe(key, new_owner);
    }

    /**
     * \brief update the ownership for a key that is managed by this PS
     *        Returns the previous owner.
     *
     *        Warning: method is not thread safe. The caller needs to ensure safety manually via lock() and unlock()
     */
    inline OwnershipT updateResidenceUnsafe(Key key, OwnershipT new_owner) {
      auto old_owner = ownership[key-myRange().begin()];
      ownership[key-myRange().begin()] = new_owner;
      return old_owner;
    }

    // returns true if this node is the manager of this key
    bool isManagedHere(Key key) {
      return key >= myRange().begin() && key < myRange().end();
    }

    // returns the manager of a key
    inline unsigned int getManager(Key key) {
      for(uint i=0; i!=ranges.size(); ++i) {
        if(key < ranges[i].end()) {
          return i;
        }
      }
      CHECK(false) << "Fatal: key " << key << " is not in configured key range [0:" << Postoffice::Get()->max_key() << "). Check the configured maximum key.\n";; // should be found
      return -1;
    }

    inline void updateCache(Key key, OwnershipT current_owner) {
      std::lock_guard<std::mutex> lck(mu_);
      updateCacheUnsafe(key, current_owner);
    };

    inline void updateCacheUnsafe(Key key, OwnershipT current_owner) {
      locationCache[key] = current_owner;
    };

    inline OwnershipT getCache(Key key) {
      std::lock_guard<std::mutex> lck(mu_);
      return locationCache[key];
    }

    // lock externally
    inline void lock() {
      mu_.lock();
    }

    // unlock externally
    inline void unlock() {
      mu_.unlock();
    }

  private:

    // returns the owner of a key that is managed by this PS
    inline unsigned int ownerOfKey(Key key) {
      // assert key is actually managed by this PS
      CHECK(isManagedHere(key));

      std::lock_guard<std::mutex> lck(mu_);
      return ownership[key-myRange().begin()];
    }

    inline const Range& myRange() const {
      return ranges[po.my_rank()];
    }

    // Ownership storage: for each managed key, store the current owner of that key
    std::vector<OwnershipT> ownership;
    std::mutex mu_;

    // reference to post office (for more readable code)
    Postoffice& po = *Postoffice::Get();

    // reference to the Postoffice server ranges (initialized in constructor)
    const std::vector<Range>& ranges;

    // location cache
    bool useCache = false;
    std::vector<OwnershipT> locationCache;


    // ---------------------------------------
    // debug functions
    void debugStr() {
        printRanges();
        printOwnership();
    }
    void printRanges() {
      for(uint i=0; i!=ranges.size(); ++i) {
        std::cout << "Range " << i << ": " << ranges[i].begin() << ":" << ranges[i].end() << "\n";
      }
    }
    void printOwnership() {
      std::cout << "Ownership:" << "\n";
      for(auto i=myRange().begin(); i!=myRange().end(); ++i) {
        std::cout << "Key " << i << " at " << ownerOfKey(i) << "\n";
      }
    }
  }; // class Addressbook
}  // namespace ps
#endif  // PS_ADDRESS_BOOK_
