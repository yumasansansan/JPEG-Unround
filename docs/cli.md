<!-- SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com> -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->

# The command line of JPEG-Unround

```
unround [OPTIONS] INPUT [OUTPUT]
```

`unround` reads the JPEG file `INPUT`, reconstructs it within its quantization
intervals (`docs/math.md`), and writes the result to `OUTPUT`. The options are
those of the library, under the same names: the Rust crate, the C interface and
the Python package take every one of them, and every value the model and the
solvers use can be given. A value not given takes its default, which this
document states; the defaults of the solvers are those of `docs/math.md`, 6.5.

## Input and output

`INPUT` is a path, or `-` for standard input. `OUTPUT` is a path, or `-` for
standard output; without it, the result goes next to `INPUT`, with its extension
replaced by that of the format (`.tif` for TIFF). A file that exists is not
overwritten unless `--overwrite` is given.

`--format FORMAT` chooses the format; without it, the extension of `OUTPUT` does
(`.tif` and `.tiff` for TIFF, `.png` for PNG, `.pgm`, `.ppm` and `.pnm` for PNM),
and without either, TIFF.

- `tiff`: the result in binary64, bit for bit: TIFF 6.0, little-endian, one strip,
  64-bit IEEE floating-point samples, neither rounded nor clamped. A greyscale file
  gives one sample to a pixel, a colour one R, G and B, or with `--ycbcr` the Y, Cb
  and Cr of the solution, which the file declares (`PhotometricInterpretation`
  YCbCr, JFIF's coefficients, full range, no subsampling). The ICC profile is
  embedded as `InterColorProfile` (34675).
- `png`: 8-bit or 16-bit samples, greyscale or RGB, not interlaced, rounded as
  PNM's are (below), with the ICC profile in an `iCCP` chunk named `ICC profile`.
  `--compression N` is zlib's level of the samples and of the profile, 0 (none)
  to 9 (the most); without it, zlib's default, 6. The file is written by libpng
  and zlib-ng. A profile that compresses into an `iCCP` chunk shorter than
  libpng's reading takes (92 bytes) is stored uncompressed instead, and a profile
  of more than 8000000 bytes, which libpng's readers leave out unless they allow
  more, is written with a warning.
- `pnm`: 8-bit or 16-bit samples, `P5` (greyscale) or `P6` (colour). `--bits 8`
  (the default) rounds each sample to the nearest integer, halves away from 0, and
  clamps it to 0–255; `--bits 16` scales it by 257 first and clamps it to 0–65535,
  so that 255 is white in both. PNM holds no ICC profile: where the file has one,
  a warning says that it is not written.

`--bits` is for PNG and PNM, `--compression` for PNG, and `--ycbcr` for TIFF and
colour files in YCbCr.

### Orientation

`--orientation apply` (the default) turns the result upright as the file's EXIF
orientation (the tag `Orientation`, 274, of its APP1 marker) says; `--orientation
keep` writes the picture as the file stores it. A file without the tag, or with a
value other than 1 to 8, is written as stored. Value `o` says where the stored
picture's first row and first column are to be seen; pixel `(r, c)` of the result,
of the stored picture `S` of `h` rows and `w` columns, is:

| `o` | First row, first column | Result | Size |
|---|---|---|---|
| 1 | top, left | `S[r][c]` | `h x w` |
| 2 | top, right | `S[r][w-1-c]` | `h x w` |
| 3 | bottom, right | `S[h-1-r][w-1-c]` | `h x w` |
| 4 | bottom, left | `S[h-1-r][c]` | `h x w` |
| 5 | left, top | `S[c][r]` | `w x h` |
| 6 | right, top | `S[h-1-c][r]` | `w x h` |
| 7 | right, bottom | `S[h-1-c][w-1-r]` | `w x h` |
| 8 | left, bottom | `S[c][w-1-r]` | `w x h` |

The pixels move and none changes: TIFF's samples are still the result's, bit for
bit, and `--ycbcr` turns the Y, Cb and Cr alike. The orientation is applied to the
result only; the reconstruction is of the picture as the file stores it.

### ICC profile

The file's ICC profile (its APP2 markers) goes into TIFF and PNG where it goes
with the picture, as libpng takes the profile of an `iCCP` chunk when it reads
one: at least 132 bytes, the length its header gives, a multiple of 4 from version
4 on, the signature `acsp`, its tags within it, a rendering intent below 0xFFFF, a
class that is neither abstract (`abst`) nor a device link (`link`), the data colour
space of the picture (`GRAY` for greyscale, `RGB ` for colour, and for the Y, Cb
and Cr of `--ycbcr`, whose colours are those of the RGB that JFIF's conversion
gives, as in the JPEG file), and XYZ or Lab as its connection space. A profile
that fails is left out, with a warning that says why. `--no-icc` writes no
profile.

Warnings go to standard error, after the file is written (not with `--quiet`), and
into the report.

## The method and the model

| Option | Values | Default | |
|---|---|---|---|
| `--method` | `mmse`, `tv`, `tgv`, `subgradient` | `tgv` | The decoder of the centres (2.3), TV or TGV by the primal–dual method (5), or TV by the subgradient method (7; greyscale files only) |
| `--alpha` | > 0 | 1 | TV's weight (4.2) |
| `--alpha1` | > 0 | 1 | TGV's weight of the first-order term (4.3) |
| `--alpha0` | > 0 | 2 | TGV's weight of the second-order term |
| `--channel-weights` | > 0, one for each component | 1 each | `gamma` of each channel's differences (4.4) |
| `--channels` | `coupled`, `apart` | `coupled` | Whether the regularizer takes the channels together, pixel by pixel (4.4) |

## The data term

Each of these takes one value for every component, or a comma-separated list of
one value for each (4.1, 4.4).

| Option | Values | Default | |
|---|---|---|---|
| `--mu` | ≥ 0, or `rule` | 0.001 | The weight of the data term; `rule` makes it `mu-scale` times the mean of the component's 64 steps to the power `mu-power` |
| `--mu-scale` | ≥ 0 | 1 | The scale of the rule of `mu` |
| `--mu-power` | finite | 1 | The power of the rule of `mu` |
| `--weight-power` | ≥ 0 | 2 | `p`: the weight of a coefficient is `mu / Q^p` |
| `--dc-weight` | ≥ 0 | 0 | The weight of DC, a factor of that of AC |
| `--centres` | `mmse`, `midpoint` | `mmse` | The MMSE centres of the Laplace model (2.2), or the middles of the intervals |
| `--slack` | ≥ 0 | 0 | Widens every interval by that many steps on each side (1.2) |
| `--slack-cost` | ≥ 0 | 0 | What leaving the file's own interval costs, per step, within the slack |

## The primal–dual method

| Option | Values | Default | |
|---|---|---|---|
| `--iterations` | ≥ 0 | TV 20000, TGV 10000 | The most iterations |
| `--tolerance` | ≥ 0 | TV `2e-4` times `alpha`, TGV `1e-2` times `alpha1` | Stops where the duality gap per sample is at most this; 0 turns it off (6.4) |
| `--relative-tolerance` | ≥ 0 | 0 | Stops where the gap is at most this times the primal value; 0 turns it off |
| `--partial-tolerance` | ≥ 0 | 0 | TGV: stops where the partial gap per sample is at most this; needs `--partial-radius` |
| `--partial-radius` | ≥ 0 | none | TGV: the radius of the partial gap (6.3), which is then recorded |
| `--step-ratio` | > 0 | TV 30 over `alpha` squared, TGV 10 over `alpha1` squared | `tau / sigma` (5) |
| `--relaxation` | in (0, 2) | 1.9 | `rho` |
| `--step-product` | in (0, 1) | 0.99 | `sigma tau L^2` |
| `--norm-squared` | > 0 | TV 8, TGV `(17 + sqrt 33) / 2`, times the largest channel weight squared | `L^2`, a bound of the operator's norm squared |
| `--no-weight-scaling` | | | Keeps the defaults of the ratio and the tolerance as they are for a weight of 1 (6.5) |
| `--record-every` | ≥ 0 | 10 | Iterations between records, where the gaps are taken and the tolerances checked; 0: only the last |
| `--free-radius` | ≥ 0 | 255 | `R` of the box of free samples (6.6) |

## The subgradient method

| Option | Values | Default | |
|---|---|---|---|
| `--subgradient-iterations` | ≥ 0 | 50 | The iterations |
| `--subgradient-step` | > 0 | 0.5 | `eta` (7) |
| `--subgradient-decay` | ≥ 0 | 0.5 | `beta` |
| `--no-momentum` | | | Turns FISTA's extrapolation off |
| `--subgradient-record-every` | ≥ 0 | 1 | Iterations between records |

## Reading

| Option | Values | Default | |
|---|---|---|---|
| `--max-pixels` | ≥ 1 | 268435456 | The largest width times height a file may declare |
| `--max-scans` | ≥ 1 | 500 | The most scans a progressive file may have |
| `--warnings-are-errors` | | | A corrupt-data warning of libjpeg fails the read |

## Reports

- `-v`, `--verbose`: each record on standard error: the iteration, the gap per
  sample, the primal and dual values, and the solver's seconds.
- `-q`, `--quiet`: nothing but errors.
- `--report PATH`: a JSON object with what was done: the version, the input, the
  size as the file stores it, the colour space, the file's EXIF orientation
  (`exif_orientation`, 0 where it has none), whether the result was turned
  (`oriented`), whether the output holds the ICC profile (`icc_profile`), the
  warnings of the writing (`warnings`), the method, the settings, the solver's
  iterations, why it stopped (`converged`, `iterations`, `observer`,
  `stationary`), and every record (`iterations`, `seconds`, `primal`, `dual`,
  `scaling`, `partial_gap`, the last two `null` where not taken). `-` writes it to
  standard output.
- `--version`: the version and the implementation, and nothing else.
- `-h`, `--help`: the options.

## Builds

On x86-64 the program needs a processor with AVX2 and FMA (x86-64-v3): its
iterations are vector code of four doubles, a product fused with the sum that
takes it. A second build, for processors with AVX-512 (x86-64-v4), computes with
vectors of eight; the two give the same results up to the last bits of what
is computed in floating point (docs/math.md, Arithmetic). On AArch64 the vectors
are NEON's, of two doubles.

## Syntax and exit status

An option takes its value as the next argument or after `=` (`--mu 0.01`,
`--mu=0.01`). Numbers are decimal, as `0.5`, `2e-4` or `20000`. An option given
twice, a value out of its range, or one that the method does not take (a partial
gap for TV, say) is refused, with every such error named at once.

| Status | |
|---|---|
| 0 | The result is written. |
| 1 | The file is not read, or its layout is not taken (sampling factors that are not whole multiples, say). |
| 2 | The command line is refused. |
| 3 | The result could not be written. |
