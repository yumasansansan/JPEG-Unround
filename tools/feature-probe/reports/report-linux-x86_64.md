# Feature probes: linux-x86_64

- Date: 2026-09-23T10:49:27+00:00
- System: Linux-6.18.40.1-microsoft-standard-WSL2-x86_64-with-glibc2.43 (x86_64)
- Compiler: Ubuntu clang version 23.1.3 (++20260922084409+67f4a076a097-1~exp1~20260922084419.77) (target x86_64-pc-linux-gnu)
- C++ library: libstdc++ 15 (20260321)
- C library: glibc 2.43
- Compile flags: `-O2 -pthread`
- Link flags: `-fuse-ld=lld -pthread`
- Standard flags: c++20 → `-std=c++20`, c++23 → `-std=c++23`, c++26 → `-std=c++26`, c23 → `-std=c23`

## C++ probes

| Probe | Feature | Paper | c++23 | c++26 | First diagnostic or output |
|---|---|---|---|---|---|
| `clang_vector_builtins` | Clang ext_vector_type with __builtin_elementwise_* (portable SIMD) | Clang extension | ✅ | ✅ |  |
| `clang_masked_load_store` | __builtin_masked_load / __builtin_masked_store (vector tails) | Clang extension | ✅ | ✅ |  |
| `target_clones` | function multiversioning with target_clones (x86-64) | Clang extension | ✅ | ✅ |  |
| `target_attribute_dispatch` | target("avx2,fma") functions chosen with __builtin_cpu_supports (x86-64) | Clang extension | ✅ | ✅ |  |
| `aarch64_fmv` | function multiversioning with target_version (AArch64) | Clang extension (ACLE FMV) | ➖ | ➖ |  |
| `float16_extension` | _Float16 arithmetic (Clang) | Clang extension (ISO/IEC TS 18661-3 type) | ✅ | ✅ |  |
| `bfloat16_extension` | __bf16 arithmetic (Clang) | Clang extension | ✅ | ✅ |  |
| `openmp` | OpenMP parallel regions (-fopenmp); prints the threads it ran | OpenMP 5.x | ✅ (`-fopenmp`) | ✅ (`-fopenmp`) | c++23: threads=16 max=16; needs libstdc++.so.6, libm.so.6, libomp.so.5, libgcc_s.so.1, libc.so.6; c++26: threads=16 max=16 |
| `openmp_simd` | #pragma omp simd without the OpenMP runtime (-fopenmp-simd) | OpenMP 5.x | ✅ (`-fopenmp-simd`) | ✅ (`-fopenmp-simd`) | needs libstdc++.so.6, libm.so.6, libgcc_s.so.1, libc.so.6 |
| `cxx_runtime_linkage` | the C++ library a program links (should be the system's, as a shared library) | - | ✅ | ✅ | c++23: x 1; needs libstdc++.so.6, libm.so.6, libgcc_s.so.1, libc.so.6; c++26: x 1 |
| `deducing_this` | explicit object parameter (deducing this) | P0847R7 | ✅ | ✅ |  |
| `multidimensional_subscript` | multidimensional subscript operator a[i, j] | P2128R6 | ✅ | ✅ |  |
| `if_consteval` | if consteval | P1938R3 | ✅ | ✅ |  |
| `static_call_operator` | static operator() and static lambdas | P1169R4 | ✅ | ✅ |  |
| `auto_cast` | auto(x) decay-copy | P0849R8 | ✅ | ✅ |  |
| `assume_attribute` | [[assume(expr)]] | P1774R8 | ✅ | ✅ |  |
| `size_t_literal` | uz / z integer literal suffixes | P0330R8 | ✅ | ✅ |  |
| `constexpr_static_local` | static constexpr variables in constexpr functions | P2647R1 | ✅ | ✅ |  |
| `elifdef` | #elifdef / #elifndef | P2334R1 | ✅ | ✅ |  |
| `consteval_escalation` | consteval propagates up (immediate-escalating functions) | P2564R3 | ✅ | ✅ |  |
| `float16_t` | std::float16_t (<stdfloat>) | P1467R9 | ❌ compile | ❌ compile | c++23: float16_t.cxx23.cpp:8:2: error: "std::float16_t is not provided"; c++26: float16_t.cxx26.cpp:8:2: error: "std::float16_t is not provided" |
| `bfloat16_t` | std::bfloat16_t (<stdfloat>) | P1467R9 | ❌ compile | ❌ compile | c++23: bfloat16_t.cxx23.cpp:8:2: error: "std::bfloat16_t is not provided"; c++26: bfloat16_t.cxx26.cpp:8:2: error: "std::bfloat16_t is not provided" |
| `builtin_overflow` | __builtin_mul_overflow / __builtin_add_overflow (checked sizes) | Clang builtin | ✅ | ✅ |  |
| `pack_indexing` | pack indexing Ts...[I] | P2662R3 | — | ✅ |  |
| `deleted_with_reason` | = delete("reason") | P2573R2 | — | ✅ |  |
| `placeholder_variables` | placeholder variables named _ | P2169R4 | — | ✅ |  |
| `structured_binding_pack` | structured bindings that introduce a pack | P1061R10 | — | ✅ |  |
| `structured_binding_condition` | structured binding declaration as a condition | P0963R3 | — | ✅ |  |
| `constexpr_placement_new` | placement new in constant expressions | P2747R2 | — | ✅ |  |
| `constexpr_void_cast` | cast from void* in constant expressions | P2738R1 | — | ✅ |  |
| `static_assert_message` | user-generated static_assert messages | P2741R3 | — | ✅ |  |
| `embed_cxx` | #embed in C++ | P1967R14 | — | ✅ |  |
| `contracts` | contracts: pre, post, contract_assert | P2900R14 | — | ❌ compile | c++26: contracts.cxx26.cpp:8:24: error: expected function body after function declarator |
| `reflection_core` | the reflection operator ^^ (compiler side of P2996, without <meta>) | P2996R13 | — | ❌ compile | c++26: reflection_core.cxx26.cpp:8:29: error: type name requires a specifier or qualifier |
| `reflection` | static reflection with std::meta | P2996R13 | — | ❌ compile | c++26: reflection.cxx26.cpp:8:10: fatal error: 'meta' file not found |
| `expansion_statements` | expansion statements (template for) | P1306R5 | — | ✅ |  |
| `trivial_unions` | unions of non-trivial members are trivially constructible | P3074R7 | — | ❌ compile | c++26: trivial_unions.cxx26.cpp:13:8: error: call to implicitly-deleted default constructor of 'Slot' |
| `mdspan` | std::mdspan, extents, layout_right / layout_stride | P0009R18 | ❌ compile | ❌ compile | c++23: mdspan.cxx23.cpp:7:10: fatal error: 'mdspan' file not found; c++26: mdspan.cxx26.cpp:7:10: fatal error: 'mdspan' file not found |
| `expected` | std::expected with monadic operations | P0323R12, P2505R5 | ✅ | ✅ |  |
| `print` | std::print / std::println | P2093R14 | ✅ | ✅ | c++23: probe 1 2.500; c++26: probe 1 2.500 |
| `format_ranges` | formatting ranges with std::format | P2286R8 | ✅ | ✅ |  |
| `ranges_to` | std::ranges::to and container append_range | P1206R7 | ✅ | ✅ |  |
| `views_zip` | views::zip | P2321R2 | ✅ | ✅ |  |
| `views_zip_transform` | views::zip_transform | P2321R2 | ✅ | ✅ |  |
| `views_adjacent` | views::adjacent | P2321R2 | ✅ | ✅ |  |
| `views_pairwise_transform` | views::pairwise_transform (adjacent_transform) | P2321R2 | ✅ | ✅ |  |
| `views_enumerate` | views::enumerate | P2164R9 | ✅ | ✅ |  |
| `views_cartesian_product` | views::cartesian_product | P2374R4 | ✅ | ✅ |  |
| `views_chunk` | views::chunk | P2442R1 | ✅ | ✅ |  |
| `views_slide` | views::slide | P2442R1 | ✅ | ✅ |  |
| `views_stride` | views::stride | P1899R3 | ✅ | ✅ |  |
| `views_chunk_by` | views::chunk_by | P2443R1 | ✅ | ✅ |  |
| `views_join_with` | views::join_with | P2441R2 | ✅ | ✅ |  |
| `views_repeat` | views::repeat | P2474R2 | ✅ | ✅ |  |
| `views_as_rvalue` | views::as_rvalue | P2446R2 | ✅ | ✅ |  |
| `views_as_const` | views::as_const | P2278R4 | ✅ | ✅ |  |
| `ranges_fold_left` | ranges::fold_left | P2322R6 | ✅ | ✅ |  |
| `ranges_fold_left_first` | ranges::fold_left_first | P2322R6 | ✅ | ✅ |  |
| `ranges_fold_right` | ranges::fold_right | P2322R6 | ✅ | ✅ |  |
| `ranges_contains` | ranges::contains, contains_subrange | P2302R4 | ✅ | ✅ |  |
| `ranges_starts_ends_with` | ranges::starts_with, ends_with | P1659R3 | ❌ compile | ❌ compile | c++23: ranges_starts_ends_with.cxx23.cpp:12:23: error: no member named 'starts_with' in namespace 'std::ranges'; c++26: ranges_starts_ends_with.cxx26.cpp:12:23: error: no member named 'starts_with' in namespace 'std::ranges' |
| `ranges_find_last` | ranges::find_last | P1223R5 | ✅ | ✅ |  |
| `ranges_iota` | ranges::iota | P2440R1 | ✅ | ✅ |  |
| `optional_monadic` | std::optional and_then / transform / or_else | P0798R8 | ✅ | ✅ |  |
| `move_only_function` | std::move_only_function | P0288R9 | ✅ | ✅ |  |
| `out_ptr` | std::out_ptr / inout_ptr for C APIs | P1132R8 | ✅ | ✅ |  |
| `stacktrace` | std::stacktrace | P0881R7 | ✅ (`-lstdc++exp`) | ✅ (`-lstdc++exp`) |  |
| `generator` | std::generator | P2502R2 | ✅ | ✅ |  |
| `flat_map` | std::flat_map / std::flat_set | P0429R9, P1222R4 | ✅ | ✅ |  |
| `spanstream` | std::spanstream | P0448R4 | ✅ | ✅ |  |
| `utility23` | unreachable, to_underlying, byteswap, forward_like, invoke_r | P0627R6, P1682R3, P1272R4, P2445R1, P2136R3 | ✅ | ✅ |  |
| `constexpr_cmath` | constexpr <cmath> basics (fabs, floor, fmax, copysign, isnan) | P0533R9 | ❌ compile | ❌ compile | c++23: constexpr_cmath.cxx23.cpp:7:15: error: static assertion expression is not an integral constant expression; c++26: constexpr_cmath.cxx26.cpp:7:15: error: static assertion expression is not an integral constant expression |
| `string23` | string::contains, resize_and_overwrite | P1679R3, P1072R10 | ✅ | ✅ |  |
| `start_lifetime_as` | std::start_lifetime_as | P2590R2 | ❌ compile | ❌ compile | c++23: start_lifetime_as.cxx23.cpp:16:17: error: no member named 'start_lifetime_as' in namespace 'std'; c++26: start_lifetime_as.cxx26.cpp:16:17: error: no member named 'start_lifetime_as' in namespace 'std' |
| `bind_back` | std::bind_back | P2387R3 | ✅ | ✅ |  |
| `constexpr_unique_ptr` | constexpr std::unique_ptr | P2273R3 | ✅ | ✅ |  |
| `threads20` | jthread, stop_token, barrier, latch, counting_semaphore, atomic wait | P0660R10, P1135R6 | ✅ | ✅ |  |
| `format_float` | std::format of floating point (needs to_chars in the library) | P0645R10 | ✅ | ✅ |  |
| `from_chars_double` | std::from_chars for double | P0067R5 | ✅ | ✅ |  |
| `from_chars_float` | std::from_chars for float | P0067R5 | ✅ | ✅ |  |
| `to_chars_double` | std::to_chars for double | P0067R5 | ✅ | ✅ |  |
| `to_chars_float` | std::to_chars for float | P0067R5 | ✅ | ✅ |  |
| `filesystem_utf8` | std::filesystem with UTF-8 (u8) file names | P0218R1, P0482R6 | ✅ | ✅ |  |
| `parallel_algorithms` | parallel algorithms (execution::par / par_unseq); prints the threads used | P0024R2 | ✅ | ✅ | c++23: threads=1; needs libstdc++.so.6, libm.so.6, libgcc_s.so.1, libc.so.6; c++26: threads=1 |
| `hardware_interference_size` | std::hardware_destructive_interference_size | P0154R1 | ✅ | ✅ |  |
| `assume_aligned` | std::assume_aligned | P1007R3 | ✅ | ✅ |  |
| `source_location` | std::source_location | P1208R6 | ✅ | ✅ |  |
| `hardening` | library hardening stops vector[] out of range (vendor switches) | P3471R4 (vendor modes) | ✅ (`-D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_FAST -D_GLIBCXX_ASSERTIONS -D_MSVC_STL_HARDENING=1`) | ✅ (`-D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_FAST -D_GLIBCXX_ASSERTIONS -D_MSVC_STL_HARDENING=1`) |  |
| `simd_vec` | std::simd::vec (<simd>, C++26 names) | P1928R15, P3287R3 | — | ❌ compile | c++26: simd_vec.cxx26.cpp:8:10: fatal error: 'simd' file not found |
| `simd_draft_names` | std::simd<float> (<simd>, names of P1928 before P3287) | P1928R15 | — | ❌ compile | c++26: simd_draft_names.cxx26.cpp:6:10: fatal error: 'simd' file not found |
| `experimental_simd` | std::experimental::simd (Parallelism TS 2) | N4808 | ✅ | ✅ |  |
| `submdspan` | std::submdspan, full_extent, strided_slice | P2630R4 | — | ❌ compile | c++26: submdspan.cxx26.cpp:7:10: fatal error: 'mdspan' file not found |
| `mdspan_padded` | std::layout_right_padded / layout_left_padded | P2642R6 | — | ❌ compile | c++26: mdspan_padded.cxx26.cpp:6:10: fatal error: 'mdspan' file not found |
| `aligned_accessor` | std::aligned_accessor for mdspan | P2897R7 | — | ❌ compile | c++26: aligned_accessor.cxx26.cpp:8:10: fatal error: 'mdspan' file not found |
| `dims` | std::dims | P2389R2 | — | ❌ compile | c++26: dims.cxx26.cpp:7:10: fatal error: 'mdspan' file not found |
| `saturation_arithmetic` | add_sat, sub_sat, saturate_cast | P0543R3 | — | ✅ |  |
| `inplace_vector` | std::inplace_vector | P0843R14 | — | ❌ compile | c++26: inplace_vector.cxx26.cpp:7:10: fatal error: 'inplace_vector' file not found |
| `linalg` | std::linalg (BLAS-like on mdspan) | P1673R13 | — | ❌ compile | c++26: linalg.cxx26.cpp:7:10: fatal error: 'linalg' file not found |
| `views_concat` | views::concat | P2542R8 | — | ✅ |  |
| `views_indices` | views::indices | P3060R3 | — | ❌ compile | c++26: views_indices.cxx26.cpp:10:28: error: no member named 'indices' in namespace 'std::ranges::views' |
| `function_ref` | std::function_ref | P0792R14 | — | ❌ compile | c++26: function_ref.cxx26.cpp:8:23: error: no template named 'function_ref' in namespace 'std'; did you mean 'function'? |
| `copyable_function` | std::copyable_function | P2548R6 | — | ❌ compile | c++26: copyable_function.cxx26.cpp:9:8: error: no member named 'copyable_function' in namespace 'std' |
| `optional_ref` | std::optional<T&> | P2988R12 | — | ❌ compile | c++26: optional:246:8: error: union member '_M_value' has reference type 'int &' |
| `optional_range` | std::optional as a range | P3168R2 | — | ❌ compile | c++26: optional_range.cxx26.cpp:11:14: error: invalid range expression of type 'std::optional<int>'; no viable 'begin' function available |
| `runtime_format` | std::runtime_format | P2918R2 | — | ✅ |  |
| `println_blank` | std::println() with no arguments | P3142R0 | — | ✅ |  |
| `cxx_stdckdint` | <stdckdint.h> ckd_add / ckd_mul in C++ | P3370R1 | — | ✅ |  |
| `text_encoding` | std::text_encoding | P1885R12 | — | ✅ |  |
| `debugging` | std::is_debugger_present / breakpoint | P2546R5 | — | ❌ compile | c++26: debugging.cxx26.cpp:7:10: fatal error: 'debugging' file not found |
| `execution_senders` | std::execution senders and receivers | P2300R10 | — | ❌ compile | c++26: execution_senders.cxx26.cpp:11:19: error: no member named 'just' in namespace 'std::execution' |
| `hive` | std::hive | P0447R28 | — | ❌ compile | c++26: hive.cxx26.cpp:7:10: fatal error: 'hive' file not found |
| `philox_engine` | std::philox_engine (counter-based random numbers) | P2075R6 | — | ❌ compile | c++26: philox_engine.cxx26.cpp:9:8: error: no member named 'philox4x32' in namespace 'std' |
| `is_within_lifetime` | std::is_within_lifetime | P2641R4 | — | ❌ compile | c++26: is_within_lifetime.cxx26.cpp:10:15: error: no member named 'is_within_lifetime' in namespace 'std' |
| `span_at` | std::span::at | P2821R5 | — | ✅ |  |
| `atomic_fetch_max` | atomic fetch_max / fetch_min | P0493R5 | — | ❌ compile | c++26: atomic_fetch_max.cxx26.cpp:10:5: error: no member named 'fetch_max' in 'std::atomic<int>' |

## C23 probes

| Probe | Feature | Paper | c23 | First diagnostic or output |
|---|---|---|---|---|
| `c23_version` | __STDC_VERSION__ is 202311L |  | ✅ |  |
| `c23_nullptr` | nullptr and nullptr_t |  | ✅ |  |
| `c23_constexpr` | constexpr objects |  | ✅ |  |
| `c23_typeof` | typeof and typeof_unqual |  | ✅ |  |
| `c23_keywords` | bool/true/false, static_assert, alignas, alignof, thread_local as keywords |  | ✅ |  |
| `c23_auto` | auto type inference |  | ✅ |  |
| `c23_attributes` | [[nodiscard("...")]], [[maybe_unused]], [[deprecated]], [[fallthrough]], [[noreturn]] |  | ✅ |  |
| `c23_empty_initializer` | empty initializer = {} |  | ✅ |  |
| `c23_literals` | binary literals and digit separators |  | ✅ |  |
| `c23_enum_underlying` | enumerations with a fixed underlying type |  | ✅ |  |
| `c23_bitint` | _BitInt(N) |  | ✅ |  |
| `c23_embed` | #embed |  | ✅ |  |
| `c23_stdckdint` | <stdckdint.h> ckd_add / ckd_mul (checked sizes) |  | ✅ |  |
| `c23_stdbit` | <stdbit.h> bit utilities |  | ✅ |  |
| `c23_unreachable` | unreachable() from <stddef.h> |  | ✅ |  |
| `c23_memset_explicit` | memset_explicit |  | ✅ |  |
| `c23_strdup` | strdup / strndup in <string.h> |  | ✅ |  |
| `c23_va_start` | va_start with one argument, variadic without a named parameter |  | ✅ |  |
| `c23_unnamed_parameters` | unnamed parameters in function definitions |  | ✅ |  |
| `c23_labels` | labels before declarations and at the end of blocks |  | ✅ |  |
| `c23_elifdef` | #elifdef / #elifndef |  | ✅ |  |
| `c23_free_sized` | free_sized / free_aligned_sized |  | ✅ |  |
| `c23_strfromd` | strfromd / strfromf |  | ✅ |  |
| `c11_threads` | <threads.h> (C11 threads, optional in C23) |  | ✅ |  |
| `c_setjmp_longjmp` | setjmp / longjmp (how libjpeg reports errors) |  | ✅ |  |

## import std

| Standard | Result | Detail |
|---|---|---|
| c++23 | ✅ | /usr/include/c++/15/bits/std.cc |
| c++26 | ✅ | /usr/include/c++/15/bits/std.cc |

## Feature-test macros

| Macro | c++23 | c++26 |
|---|---|---|
| `__cpp_explicit_this_parameter` | 202110 | 202110 |
| `__cpp_multidimensional_subscript` | 202211 | 202211 |
| `__cpp_if_consteval` | 202106 | 202106 |
| `__cpp_static_call_operator` | 202207 | 202207 |
| `__cpp_auto_cast` | 202110 | 202110 |
| `__has_cpp_attribute(assume)` | 202207 | 202207 |
| `__cpp_size_t_suffix` | 202011 | 202011 |
| `__cpp_constexpr` | 202211 | 202406 |
| `__cpp_consteval` | 202211 | 202211 |
| `__STDCPP_FLOAT16_T__` | - | - |
| `__STDCPP_BFLOAT16_T__` | - | - |
| `__cpp_pack_indexing` | 202311 | 202311 |
| `__cpp_deleted_function` | 202403 | 202403 |
| `__cpp_placeholder_variables` | 202306 | 202306 |
| `__cpp_structured_bindings` | 202411 | 202411 |
| `__cpp_static_assert` | 202306 | 202306 |
| `__cpp_pp_embed` | - | - |
| `__cpp_contracts` | - | - |
| `__cpp_impl_reflection` | - | - |
| `__cpp_lib_reflection` | - | - |
| `__cpp_expansion_statements` | - | - |
| `__cpp_trivial_union` | - | - |
| `__cpp_lib_mdspan` | - | - |
| `__cpp_lib_expected` | 202211 | 202211 |
| `__cpp_lib_print` | 202211 | 202211 |
| `__cpp_lib_format_ranges` | 202207 | 202207 |
| `__cpp_lib_ranges_to_container` | 202202 | 202202 |
| `__cpp_lib_containers_ranges` | 202202 | 202202 |
| `__cpp_lib_ranges_zip` | 202110 | 202110 |
| `__cpp_lib_ranges_enumerate` | 202302 | 202302 |
| `__cpp_lib_ranges_cartesian_product` | 202207 | 202207 |
| `__cpp_lib_ranges_chunk` | 202202 | 202202 |
| `__cpp_lib_ranges_slide` | 202202 | 202202 |
| `__cpp_lib_ranges_stride` | 202207 | 202207 |
| `__cpp_lib_ranges_chunk_by` | 202202 | 202202 |
| `__cpp_lib_ranges_join_with` | 202202 | 202202 |
| `__cpp_lib_ranges_repeat` | 202207 | 202207 |
| `__cpp_lib_ranges_as_rvalue` | 202207 | 202207 |
| `__cpp_lib_ranges_as_const` | 202311 | 202311 |
| `__cpp_lib_ranges_fold` | 202207 | 202207 |
| `__cpp_lib_ranges_contains` | 202207 | 202207 |
| `__cpp_lib_ranges_starts_ends_with` | - | - |
| `__cpp_lib_ranges_find_last` | 202207 | 202207 |
| `__cpp_lib_ranges_iota` | 202202 | 202202 |
| `__cpp_lib_optional` | 202110 | 202110 |
| `__cpp_lib_move_only_function` | 202110 | 202110 |
| `__cpp_lib_out_ptr` | 202311 | 202311 |
| `__cpp_lib_stacktrace` | 202011 | 202011 |
| `__cpp_lib_generator` | 202207 | 202207 |
| `__cpp_lib_flat_map` | 202207 | 202207 |
| `__cpp_lib_flat_set` | 202207 | 202207 |
| `__cpp_lib_spanstream` | 202106 | 202106 |
| `__cpp_lib_unreachable` | 202202 | 202202 |
| `__cpp_lib_to_underlying` | 202102 | 202102 |
| `__cpp_lib_byteswap` | 202110 | 202110 |
| `__cpp_lib_forward_like` | 202207 | 202207 |
| `__cpp_lib_invoke_r` | 202106 | 202106 |
| `__cpp_lib_constexpr_cmath` | - | - |
| `__cpp_lib_string_contains` | 202011 | 202011 |
| `__cpp_lib_string_resize_and_overwrite` | 202110 | 202110 |
| `__cpp_lib_start_lifetime_as` | - | - |
| `__cpp_lib_bind_back` | 202202 | 202202 |
| `__cpp_lib_constexpr_memory` | 202202 | 202202 |
| `__cpp_lib_jthread` | 201911 | 201911 |
| `__cpp_lib_barrier` | 201907 | 201907 |
| `__cpp_lib_latch` | 201907 | 201907 |
| `__cpp_lib_semaphore` | 201907 | 201907 |
| `__cpp_lib_atomic_wait` | 201907 | 201907 |
| `__cpp_lib_format` | 202304 | 202311 |
| `__cpp_lib_to_chars` | 201611 | 202306 |
| `__cpp_lib_filesystem` | 201703 | 201703 |
| `__cpp_lib_char8_t` | 201907 | 201907 |
| `__cpp_lib_execution` | 201902 | 201902 |
| `__cpp_lib_parallel_algorithm` | 201603 | 201603 |
| `__cpp_lib_hardware_interference_size` | 201703 | 201703 |
| `__cpp_lib_assume_aligned` | 201811 | 201811 |
| `__cpp_lib_source_location` | 201907 | 201907 |
| `__cpp_lib_simd` | - | - |
| `__cpp_lib_experimental_parallel_simd` | - | - |
| `__cpp_lib_submdspan` | - | - |
| `__cpp_lib_aligned_accessor` | - | - |
| `__cpp_lib_saturation_arithmetic` | - | 202311 |
| `__cpp_lib_inplace_vector` | - | - |
| `__cpp_lib_linalg` | - | - |
| `__cpp_lib_ranges_concat` | - | 202403 |
| `__cpp_lib_ranges_indices` | - | - |
| `__cpp_lib_function_ref` | - | - |
| `__cpp_lib_copyable_function` | - | - |
| `__cpp_lib_optional_range_support` | - | - |
| `__cpp_lib_stdckdint` | - | - |
| `__cpp_lib_text_encoding` | - | 202306 |
| `__cpp_lib_debugging` | - | - |
| `__cpp_lib_senders` | - | - |
| `__cpp_lib_hive` | - | - |
| `__cpp_lib_philox_engine` | - | - |
| `__cpp_lib_is_within_lifetime` | - | - |
| `__cpp_lib_span` | 202002 | 202311 |
| `__cpp_lib_atomic_min_max` | - | - |
| `__cplusplus` | 202302 | 202400 |
| `__cpp_modules` | 1 | 1 |
| `__cpp_lib_modules` | 202207 | 202207 |
| `__cpp_concepts` | 202002 | 202002 |
| `__cpp_lib_ranges` | 202302 | 202406 |
| `__cpp_lib_bit_cast` | 201806 | 201806 |
| `__cpp_lib_math_constants` | 201907 | 201907 |
