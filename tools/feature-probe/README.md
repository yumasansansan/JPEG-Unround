<!--
SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Feature probes

JPEG-Unround uses a language or library feature only when it works on every
system the project supports, with the toolchain the project builds with:
Clang 23 and LLD, and the system's own C++ library (the MSVC STL on Windows,
libstdc++ on Linux, the SDK's libc++ on macOS). These probes are how that is
known rather than guessed. Each probe is a small program that exercises one
feature; it passes when it compiles, links and exits with status 0.

| Path | What it measures |
|---|---|
| `run.py` | Compiles, links and runs every probe of `cxx/*.cpp` (in `-std=c++23` and `-std=c++26`) and `c/*.c` (`-std=c23`), builds `import std` by hand, and writes `reports/report-<system>.{md,json}` |
| `cxx/language.cpp` | C++23 and C++26 core language |
| `cxx/library23.cpp` | the C++23 library, and the C++17/20 parts whose support differs between libraries |
| `cxx/library26.cpp` | the C++26 library |
| `cxx/extensions.cpp` | Clang's vector extensions, multiversioning, half precision, OpenMP, and which shared libraries a program ends up needing |
| `c/c23.c` | C23 language and C library |
| `cmake-modules/` | C++ modules through CMake and Ninja: a named module, `import std`, and headers mixed with `import std` |
| `rust-link/` | Rust linked by LLD against a C23 static library in which `setjmp`/`longjmp` stays inside C |
| `reports/` | the results measured on 2026-09-23: Windows 11 with the MSVC STL and Ubuntu 26.04 on WSL with libstdc++ 15 (Clang 23.1.3 of apt.llvm.org) on a desktop, and macOS 26.6.2 with the Xcode 26.6 SDK's libc++ in CI (run 35851566476), with that run's module and Rust results. CI's Windows and Linux gave the same results as the desktop, except OpenMP, which CI's Linux does not install |

## Running

```sh
# Windows (Git Bash or PowerShell), with LLVM 23 on PATH
python tools/feature-probe/run.py

# Ubuntu (WSL), where the default clang++ is Ubuntu's older one
python3 tools/feature-probe/run.py --cxx clang++-23 --cc clang-23

# everything CI runs, on the current system
bash ci/probe.sh probe-reports
```

On macOS, `run.py` compiles against the SDK (`xcrun --show-sdk-path`) for the
deployment target given with `--macos-min`, 26.0 by default: the SDK's libc++
marks some features as available only from a given version (floating-point
`std::from_chars` from 26.0). `ci/probe.sh` measures every deployment target
that `UNROUND_MACOS_TARGETS` names (26.0 when unset) and names its reports after
the SDK and the target, such as `report-macos-arm64-sdk27.0-min26.0.md`: what a
newer SDK's headers bring shows for the older target already, and what needs
the newer system's own libc++ shows only for the newer target. The std module
for `import std` has to be the one of the SDK's libc++, so it is looked for in
Xcode and the SDK, never taken from the LLVM toolchain, whose `std.cppm` needs
newer headers than the SDK's. CI runs the whole set on all three systems, macOS
with Xcode 26 and with the Xcode 27 preview (the `probe` job of
`.github/workflows/ci.yml`), and keeps the reports as an artifact.

## Writing a probe

A bundle holds many probes. Each starts with `//=== probe: <id>` and a few
`//--- <key>: <value>` lines (see the top of `run.py` for the keys), followed by
a complete program. A probe that exercises several features and fails says
nothing about which one is missing: split it until the failure names one. The
probes follow the project's C and C++ rules: an empty parameter list is written
`(void)`, and no conversion is left implicit.
