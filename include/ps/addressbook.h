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
   * - To assign a manager to each key, we use hash partitioning.
   * - The manager stores who is the owner for each of its keys.
   *
   * Optionally, it caches the locations of keys. Responses to pull requests and outgoing pins update the cache
   */
  template<typename Handle>
  class Addressbook {
  public:

    Addressbook(Handle& h): handle(h) {
      ownership = std::vector<OwnershipT>(std::ceil(1.0*Postoffice::Get()->num_keys()/Postoffice::Get()->num_servers()), po.my_rank());
      relocation_counters = std::vector<Version>(ownership.size(), 0);

      if (Postoffice::Get()->use_location_caches()) {
        locationCache = std::vector<OwnershipT>(Postoffice::Get()->num_keys(), NOT_CACHED);
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
    unsigned int getDirectionsUnsafe(Key key, const bool tryToUseLocationCache) {
      if (isManagedHere(key)) {
        return ownerOfKeyUnsafe(key);
      } else {
        if (tryToUseLocationCache && Postoffice::Get()->use_location_caches()) {
          auto locCache = getLocationCacheUnsafe(key);
          if (locCache != NOT_CACHED) {
            return locationCache[key];
          }
        }
        return getManager(key);
      }
    }

    // safe wrapper
    unsigned int getDirections(Key key, const bool tryToUseLocationCache) {
      handle.lockSingle(key);
      auto directions = getDirectionsUnsafe(key, tryToUseLocationCache);
      handle.unlockSingle(key);
      return directions;
    }

    /**
     * \brief update the ownership for a key that is managed by this PS
     *        Returns the previous owner.
     */
    OwnershipT updateResidence(Key key, OwnershipT new_owner, Version relocation_counter) {
      CHECK(isManagedHere(key)); // can update ownership only for the parameters that this PS manages
      CHECK(new_owner < po.num_servers()); // new_owner has to be a valid node

      handle.lockSingle(key);
      auto ret = updateResidenceUnsafe(key, new_owner, relocation_counter);
      handle.unlockSingle(key);
      return ret;
    }

    /**
     * \brief update the ownership for a key that is managed by this PS
     *        Returns the previous owner.
     *
     *        Warning: method is not thread safe
     */
    inline OwnershipT updateResidenceUnsafe(Key key, OwnershipT new_owner, Version relocation_counter) {
      const auto key_pos = key_position(key);
      auto old_owner = ownership[key_pos];
      // we write the new residence only if it is newer than the one that we already have
      // (in the current setup, it can happen that ownership updates overtake others)
      if (relocation_counter > relocation_counters[key_pos]) {
        ownership[key_pos] = new_owner;
        relocation_counters[key_pos] = relocation_counter;
      }
      return old_owner;
    }

    // is the node that we are on the manager of `key`?
    inline bool isManagedHere (const Key key) const {
      return static_cast<int>(getManager(key)) == po.my_rank();
    }

    // returns the manager of a key
    inline unsigned int getManager(const Key key) const {
      return key % po.num_servers();
    }

    inline void updateLocationCache(const Key key, const OwnershipT current_owner) {
      handle.lockSingle(key);
      updateLocationCacheUnsafe(key, current_owner);
      handle.unlockSingle(key);
    };

    inline void updateLocationCacheUnsafe(const Key key, const OwnershipT current_owner) {
      if (current_owner == po.my_rank()) {
        // Ignore any update that tells us that this node is the owner.
        // We might get updates like that when messages got forwarded a couple of times.
        // In these cases, the location cache is probably stale, so let's reset it.
        locationCache[key] = NOT_CACHED;
      } else {
        locationCache[key] = current_owner;
      }
    };

    inline OwnershipT getLocationCacheUnsafe(const Key key) const {
      return locationCache[key];
    }

  private:

    // returns the owner of a key that is managed by this PS
    inline unsigned int ownerOfKeyUnsafe (const Key key) const {
      // assert key is actually managed by this PS
      CHECK(isManagedHere(key));

      return ownership[key_position(key)];
    }

    // at which position do we store the ownership and relocation counter for a key?
    size_t key_position (const Key key) const {
      return key / po.num_servers();
    }

    // Ownership storage: for each managed key, store the current owner of that key
    std::vector<OwnershipT> ownership;

    // store to which relocation the residence information belongs
    std::vector<Version> relocation_counters;

    // reference to post office (for more readable code)
    Postoffice& po = *Postoffice::Get();

    // location cache
    std::vector<OwnershipT> locationCache;

    Handle& handle;

    // ---------------------------------------
    // debug functions
    void debugStr() {
      std::cout << "Ownership:" << "\n";
      for(auto i=0; i!=Postoffice::Get()->num_keys(); ++i) {
        if (isManagedHere(i)) {
          handle.lockSingle(i);
          std::cout << "Key " << i << " at " << ownerOfKeyUnsafe(i) << "\n";
          handle.unlockSingle(i);
        }
      }
    }
  }; // class Addressbook
}  // namespace ps
#endif  // PS_ADDRESS_BOOK_
