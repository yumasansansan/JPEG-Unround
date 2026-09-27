// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The shapes of the library's arrays. What reads and writes an array is a
// std::mdspan of its shape, its entries row after row (layout_right) unless it
// says otherwise:
//
//   Grid<T>     rows x columns samples, [i, j]: a canvas, a field;
//   Blocks<T>   block rows x block columns of 8 x 8 entries, [by, bx, v, u]: the
//               levels or the coefficients of a component, each block's 64 in
//               natural order (entry 8 v + u of the vertical frequency v and
//               the horizontal u);
//   Block8x8<T> the 8 x 8 entries of one block, [v, u];
//   Picture<T>  rows x columns x channels, [i, j, c]: samples to write, those of
//               a pixel together.
//
// What holds the entries is a std::vector in an MdArray, which keeps it with its
// shape and gives views of it: the owning counterpart of std::mdspan (the mdarray
// of P1684, which no standard library has).
#ifndef UNROUND_ARRAYS_HPP
#define UNROUND_ARRAYS_HPP

#include <cstddef>
#include <mdspan>
#include <span>
#include <vector>

namespace unround {

using GridExtents = std::dextents<std::size_t, 2>;
using BlockExtents = std::extents<std::size_t, std::dynamic_extent, std::dynamic_extent, 8, 8>;
using PictureExtents = std::dextents<std::size_t, 3>;

template <typename T>
using Grid = std::mdspan<T, GridExtents>;
template <typename T>
using Blocks = std::mdspan<T, BlockExtents>;
template <typename T>
using Block8x8 = std::mdspan<T, std::extents<std::size_t, 8, 8>>;
template <typename T>
using Picture = std::mdspan<T, PictureExtents>;

// The entries of a view in the order of its layout, which leaves no gaps (an
// exhaustive one, as layout_right is).
template <typename T, typename Extents>
[[nodiscard]] std::span<T> entries(std::mdspan<T, Extents> view) noexcept {
  return std::span<T>(view.data_handle(), view.size());
}

// Block (by, bx) of blocks: its 64 entries in natural order, and its view [v, u].
template <typename T>
[[nodiscard]] std::span<T, 64> block_entries(Blocks<T> blocks, std::size_t by, std::size_t bx) noexcept {
  return std::span<T, 64>(&blocks[by, bx, 0, 0], 64);
}
template <typename T>
[[nodiscard]] Block8x8<T> block_of(Blocks<T> blocks, std::size_t by, std::size_t bx) noexcept {
  return Block8x8<T>(&blocks[by, bx, 0, 0]);
}

// An array of a shape that owns its entries, row after row, all T{} or the value
// given at first. Its views are std::mdspans; its entries, one after another, a
// std::span.
template <typename T, typename Extents>
class MdArray {
 public:
  using mapping_type = std::layout_right::mapping<Extents>;

  MdArray(void) = default;
  explicit MdArray(const Extents& extents, const T& value = T{})
      : mapping_(extents), entries_(mapping_.required_span_size(), value) {}

  [[nodiscard]] std::mdspan<T, Extents> view(void) noexcept { return {entries_.data(), mapping_}; }
  [[nodiscard]] std::mdspan<const T, Extents> view(void) const noexcept { return {entries_.data(), mapping_}; }
  [[nodiscard]] std::span<T> entries(void) noexcept { return entries_; }
  [[nodiscard]] std::span<const T> entries(void) const noexcept { return entries_; }
  [[nodiscard]] const Extents& extents(void) const noexcept { return mapping_.extents(); }
  [[nodiscard]] std::size_t extent(std::size_t rank) const noexcept { return mapping_.extents().extent(rank); }
  [[nodiscard]] std::size_t size(void) const noexcept { return entries_.size(); }

  friend bool operator==(const MdArray& a, const MdArray& b) noexcept {
    return a.mapping_ == b.mapping_ && a.entries_ == b.entries_;
  }

 private:
  mapping_type mapping_{};
  std::vector<T> entries_;
};

template <typename T>
using GridArray = MdArray<T, GridExtents>;
template <typename T>
using BlocksArray = MdArray<T, BlockExtents>;

// The shape of a grid, and of blocks, rows by columns.
[[nodiscard]] inline GridExtents grid_extents(std::size_t rows, std::size_t columns) noexcept {
  return GridExtents{rows, columns};
}
[[nodiscard]] inline BlockExtents block_extents(std::size_t rows, std::size_t columns) noexcept {
  return BlockExtents{rows, columns};
}

}  // namespace unround

#endif  // UNROUND_ARRAYS_HPP
