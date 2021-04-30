/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef PS_RANGE_H_
#define PS_RANGE_H_
#include "ps/internal/utils.h"
#include "ps/base.h"
namespace ps {

/**
 * \brief a range [begin, end)
 */
class Range {
 public:
  Range() : Range(0, 0) {}
  Range(Key begin, Key end) : begin_(begin), end_(end) { }

  Key begin() const { return begin_; }
  Key end() const { return end_; }
  Key size() const { return end_ - begin_; }
 private:
  Key begin_;
  Key end_;
};

}  // namespace ps
#endif  // PS_RANGE_H_
