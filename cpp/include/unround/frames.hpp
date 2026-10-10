// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The components of a picture on one canvas (docs/math.md, 1.3 and 4.4). Each
// component is the Problem of its blocks, and each of its samples stands for a cell
// of samples of the canvas, which is at the resolution of the picture. A channel of
// the canvas is a Grid of rows x columns samples; the component's coefficients are
// A x, the DCT of the means of the cells over its blocks, and the canvas moves only
// within the range of A's adjoint: the deviations within the cells, and the samples
// beyond the blocks, are free.
#ifndef UNROUND_FRAMES_HPP
#define UNROUND_FRAMES_HPP

#include "unround/arrays.hpp"
#include "unround/error.hpp"
#include "unround/model.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <vector>

namespace unround {

// The colour space of a picture: one component, or Y, Cb and Cr, or R, G and B.
enum class ColourSpace : std::int32_t { grayscale, ycbcr, rgb };

// What a picture's samples beyond the blocks of a component, and the deviations
// within its cells, are bounded by in the gap of 6.6: the box of radius R about 128
// beyond the blocks and about 0 within them.
inline constexpr double beyond_blocks = 128.0;

// A component as a file gives it: the levels of its blocks, at least those that the
// picture covers (more complete the file's MCUs, and are left out), its quantization
// table in natural order, and its sampling factors across (h) and down (v), 1 to 4.
struct ComponentInput {
  Blocks<const std::int16_t> levels;
  std::array<std::uint16_t, 64> table{};
  std::uint32_t h = 1;
  std::uint32_t v = 1;
};

// One component on the canvas: the problem of the blocks it keeps, and the cell of
// rows x columns samples of the canvas that each of its samples stands for.
struct Component {
  Problem problem;
  std::size_t cell_rows = 1;
  std::size_t cell_columns = 1;

  // n_c, the samples of a cell; nu_c = 1 / n_c (1.3).
  [[nodiscard]] std::size_t cells(void) const noexcept { return cell_rows * cell_columns; }
  [[nodiscard]] std::size_t block_rows(void) const noexcept { return problem.rows(); }
  [[nodiscard]] std::size_t block_columns(void) const noexcept { return problem.columns(); }
};

// What the maps of a component need besides their inputs and outputs, kept from one
// call to the next so that the iterations of a solver allocate nothing: the
// coefficients, the means of the cells and their change, and a row of the canvas.
struct Room {
  BlocksArray<double> coefficients;
  GridArray<double> means;
  GridArray<double> change;
  std::vector<double> row;
};

class Frame {
 public:
  // The frame of a picture of height x width samples in a colour space, of its
  // components with the options of their data terms: one for every component, or one
  // for each. Cb and Cr of a file in YCbCr take their data term's of_chroma. A
  // component of a file with several is kept to the blocks that the picture covers,
  // ceil(height / (8 r^v)) by ceil(width / (8 r^h)); the canvas is the picture rounded
  // up to whole MCUs. With one component, the canvas is its blocks. Sampling factors
  // whose ratios to the largest are not integers, and options out of their ranges, are
  // refused.
  [[nodiscard]] static std::expected<Frame, Error> make(std::size_t height, std::size_t width, ColourSpace space,
                                                        std::span<const ComponentInput> components,
                                                        std::span<const DataTerm> data);

  [[nodiscard]] std::size_t height(void) const noexcept { return height_; }
  [[nodiscard]] std::size_t width(void) const noexcept { return width_; }
  [[nodiscard]] ColourSpace colour_space(void) const noexcept { return space_; }
  // The canvas: rows x columns samples of each channel, one channel to a component.
  [[nodiscard]] std::size_t rows(void) const noexcept { return rows_; }
  [[nodiscard]] std::size_t columns(void) const noexcept { return columns_; }
  [[nodiscard]] std::size_t channels(void) const noexcept { return components_.size(); }
  [[nodiscard]] GridExtents canvas_extents(void) const noexcept { return grid_extents(rows_, columns_); }
  [[nodiscard]] const Component& component(std::size_t c) const noexcept { return components_[c]; }
  // Whether a sample of channel c is free: the cells have several samples, or the
  // blocks leave part of the canvas out.
  [[nodiscard]] bool free(std::size_t c) const noexcept;
  // Whether any is.
  [[nodiscard]] bool free(void) const noexcept;

  // The coefficients of channel c: A x, the DCT of the means of x's cells over the
  // component's blocks.
  void coefficients(std::size_t c, Grid<const double> x, Blocks<double> out) const;

  // The room of component c's maps.
  [[nodiscard]] Room room(std::size_t c) const;

  // The proximal map of tau G_c (4.4): out = v + A^T (zeta - A v) / nu, zeta the
  // proximal map of (tau / n_c) g_c at A v, which goes to `kept`. Where the cells have
  // one sample, the blocks of out are the inverse DCT of zeta, and the samples beyond
  // them those of v. out may be v.
  void prox(std::size_t c, Grid<const double> v, double tau, Grid<double> out, Blocks<double> kept, Room& room) const;

  // The projection onto the set, alike with zeta = clip(A v).
  void project(std::size_t c, Grid<const double> v, Grid<double> out, Blocks<double> kept, Room& room) const;

  // The canvas of coefficients: their inverse DCT, each sample repeated over its cell,
  // and `beyond` in the samples beyond the blocks.
  void canvas(std::size_t c, Blocks<const double> coefficients, double beyond, Grid<double> out) const;

  // The start of the solvers (5): the canvas of the centres clipped to their intervals,
  // 128 beyond the blocks; their coefficients go to `kept`.
  void start(std::size_t c, Grid<double> out, Blocks<double> kept) const;

  // The conjugate of G_c at the field xi, with the free samples of 6.6 bounded by the
  // box of radius `radius`: g_c*(A xi / nu), the DCT of the sums of xi over the cells of
  // the blocks, plus <zeta, m> + radius |zeta|_1, zeta = xi - Pi xi (the deviations of
  // xi from the means of its cells, and xi beyond the blocks), m 128 beyond the blocks
  // and 0 within them. Where nothing is free, zeta is 0 and only g_c* is taken.
  [[nodiscard]] double conjugate(std::size_t c, Grid<const double> xi, double radius) const;

 private:
  Frame(void) = default;

  // The map of prox or project: from the coefficients at A v in room.coefficients,
  // `map` writes zeta into kept; then out.
  template <typename Map>
  void map(std::size_t c, Grid<const double> v, Grid<double> out, Blocks<double> kept, Room& room,
           const Map& zeta_of) const;
  // out = v + the change of each cell repeated over it, and v beyond the blocks.
  void moved(std::size_t c, Grid<const double> v, Grid<const double> change, Grid<double> out,
             std::vector<double>& row) const;

  std::size_t height_ = 0;
  std::size_t width_ = 0;
  ColourSpace space_ = ColourSpace::grayscale;
  std::size_t rows_ = 0;
  std::size_t columns_ = 0;
  std::vector<Component> components_;
};

}  // namespace unround

#endif  // UNROUND_FRAMES_HPP
