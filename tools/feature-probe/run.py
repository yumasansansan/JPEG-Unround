#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Feature probes of JPEG-Unround.

Compiles, links and runs small programs that each exercise one language or
library feature, with the toolchain the project builds with, and writes a
report of what works. JPEG-Unround uses a feature only when its probe passes on
every system it supports: Windows (MSVC STL), Linux (libstdc++) and macOS (the
SDK's libc++). Guessing from documentation is not enough -- a feature a table
says is missing may work, and one it says is there may not link.

A bundle (cxx/*.cpp, c/*.c) holds many probes. Each starts with a line
"//=== probe: <id>" followed by "//--- <key>: <value>" lines:
  title    what the probe exercises          paper   the proposal
  std      the oldest standard to try (c++23 when absent; c23 for C)
  macro    feature-test macros to report, comma-separated
  flags    compile-and-link flags; alternatives separated by "|" are tried in order
  libs     link-only flags; alternatives as for flags
  only     systems or architectures it applies to (windows, linux, macos, x86_64, arm64)
  expect   "abnormal" when passing means the program is stopped (hardening checks)
  inspect  "imports" to list the shared libraries the program needs
A probe passes when it compiles, links and exits with status 0.

Usage:
  python tools/feature-probe/run.py [--cxx clang++] [--cc clang] [--out DIR]
      [--std c++23,c++26] [--label NAME] [--macos-min 26.0] [--jobs N]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import dataclasses
import datetime
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROBE_RE = re.compile(r"^//=== probe: (\S+)[ \t]*$", re.M)
META_RE = re.compile(r"^//--- ([a-z][a-z0-9_-]*):[ \t]?(.*)$", re.M)
CXX_STDS = ["c++20", "c++23", "c++26"]

# Macros reported for every run besides those the probes name.
EXTRA_MACROS = [
    "__cplusplus", "__cpp_modules", "__cpp_lib_modules", "__cpp_concepts",
    "__cpp_lib_ranges", "__cpp_lib_format", "__cpp_lib_execution",
    "__cpp_lib_parallel_algorithm", "__cpp_lib_jthread", "__cpp_lib_barrier",
    "__cpp_lib_atomic_wait", "__cpp_lib_to_chars", "__cpp_lib_filesystem",
    "__cpp_lib_hardware_interference_size", "__cpp_lib_assume_aligned",
    "__cpp_lib_source_location", "__cpp_lib_span", "__cpp_lib_bit_cast",
    "__cpp_lib_math_constants", "__cpp_lib_char8_t",
]


@dataclasses.dataclass
class Probe:
    id: str
    lang: str  # "c++" or "c"
    bundle: str
    body: str
    meta: dict[str, str]

    def alternatives(self, key: str) -> list[str]:
        value = self.meta.get(key)
        if value is None:
            return [""]
        return [a.strip() for a in value.split("|")]

    @property
    def min_std(self) -> str:
        return self.meta.get("std", "c++23" if self.lang == "c++" else "c23")


@dataclasses.dataclass
class Toolchain:
    system: str  # windows, linux, macos
    arch: str  # x86_64, arm64
    cxx: str
    cc: str
    cflags: list[str]
    ldflags: list[str]
    std_flag: dict[str, str]
    obj: str
    exe: str
    tools_dir: Path | None


def load_probes() -> list[Probe]:
    probes = []
    for lang, pattern in (("c++", "cxx/*.cpp"), ("c", "c/*.c")):
        for bundle in sorted(HERE.glob(pattern)):
            text = bundle.read_text(encoding="utf-8")
            marks = list(PROBE_RE.finditer(text))
            for i, mark in enumerate(marks):
                end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
                body = text[mark.start():end]
                meta = {k: v.strip() for k, v in META_RE.findall(body)}
                probes.append(Probe(mark.group(1), lang, bundle.name, body, meta))
    ids = [p.id for p in probes]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        sys.exit(f"error: probes defined twice: {', '.join(sorted(duplicates))}")
    return probes


def run(cmd: list[str], cwd: Path | None = None, timeout: float = 300) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout)


def first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def diagnostic(text: str) -> str:
    """The first error of a compiler or linker output, without the directory of its file."""
    for line in text.splitlines():
        if re.search(r"\b(error|undefined symbol|undefined reference)\b", line):
            line = re.sub(r"^(?:[A-Za-z]:)?[^:]*[\\/]([^\\/:]+:\d+)", r"\1", line.strip())
            return line[:300]
    return first_line(text)[:300]


def detect_toolchain(args) -> Toolchain:
    system = {"win32": "windows", "linux": "linux", "darwin": "macos"}.get(sys.platform, sys.platform)
    machine = platform.machine().lower()
    arch = {"amd64": "x86_64", "x86_64": "x86_64", "arm64": "arm64", "aarch64": "arm64"}.get(machine, machine)
    cflags, ldflags = ["-O2"], ["-fuse-ld=lld"]
    if system == "windows":
        # The MSVC STL and the C runtime as DLLs (/MD), as CMake links them.
        cflags += ["-fms-runtime-lib=dll", "-D_CRT_SECURE_NO_WARNINGS"]
        ldflags += ["-fms-runtime-lib=dll"]
    elif system == "linux":
        cflags += ["-pthread"]
        ldflags += ["-pthread"]
    elif system == "macos":
        sdk = run(["xcrun", "--show-sdk-path"]).stdout.strip()
        target = [f"-mmacosx-version-min={args.macos_min}"]
        cflags += ["-isysroot", sdk] + target
        ldflags += ["-isysroot", sdk] + target
    cflags += shlex.split(args.extra_cflags)
    ldflags += shlex.split(args.extra_ldflags)

    resolved = shutil.which(args.cxx)
    if not resolved:
        sys.exit(f"error: {args.cxx} is not found")
    tools_dir = Path(os.path.realpath(resolved)).parent

    tc = Toolchain(system, arch, args.cxx, args.cc, cflags, ldflags, {},
                   ".obj" if system == "windows" else ".o",
                   ".exe" if system == "windows" else "", tools_dir)
    # The spelling of each standard that this compiler accepts.
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "empty.cpp"
        src.write_text("int main(void) { return 0; }\n")
        for std in CXX_STDS:
            spellings = [std] + ({"c++26": ["c++2c"], "c++23": ["c++2b"], "c++20": ["c++2a"]}[std])
            for spelling in spellings:
                result = run([tc.cxx, f"-std={spelling}", "-fsyntax-only", str(src)])
                if result.returncode == 0:
                    tc.std_flag[std] = f"-std={spelling}"
                    break
        csrc = Path(tmp) / "empty.c"
        csrc.write_text("int main(void) { return 0; }\n")
        for spelling in ["c23", "c2x"]:
            if run([tc.cc, f"-std={spelling}", "-fsyntax-only", str(csrc)]).returncode == 0:
                tc.std_flag["c23"] = f"-std={spelling}"
                break
    return tc


def applies(probe: Probe, tc: Toolchain) -> bool:
    only = probe.meta.get("only")
    if not only:
        return True
    wanted = {w.strip() for w in only.split(",")}
    systems = wanted & {"windows", "linux", "macos"}
    arches = wanted & {"x86_64", "arm64"}
    return (not systems or tc.system in systems) and (not arches or tc.arch in arches)


def find_tool(tc: Toolchain, names: list[str]) -> str | None:
    for name in names:
        for candidate in ([str(tc.tools_dir / name)] if tc.tools_dir else []) + [name]:
            found = shutil.which(candidate)
            if found:
                return found
    return None


def imports_of(tc: Toolchain, exe: Path) -> list[str]:
    major = re.search(r"-(\d+)$", Path(tc.cxx).stem)
    suffix = [f"-{major.group(1)}"] if major else []
    if tc.system == "windows":
        tool = find_tool(tc, ["llvm-readobj"])
        if not tool:
            return ["(llvm-readobj not found)"]
        out = run([tool, "--coff-imports", str(exe)]).stdout
        return sorted(set(re.findall(r"Name: (\S+\.dll)", out, re.I)), key=str.lower)
    if tc.system == "linux":
        tool = find_tool(tc, ["llvm-readelf"] + [f"llvm-readelf{s}" for s in suffix])
        if not tool:
            return ["(llvm-readelf not found)"]
        out = run([tool, "--dynamic-table", str(exe)]).stdout
        return re.findall(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]", out)
    out = run(["otool", "-L", str(exe)]).stdout
    return [line.split()[0] for line in out.splitlines()[1:] if line.strip()]


def build_and_run(probe: Probe, std: str, tc: Toolchain, workdir: Path) -> dict:
    if not applies(probe, tc):
        return {"status": "skip", "note": f"only {probe.meta['only']}"}
    std_flag = tc.std_flag.get(std)
    if not std_flag:
        return {"status": "skip", "note": f"the compiler has no {std} mode"}
    ext = ".cpp" if probe.lang == "c++" else ".c"
    compiler = tc.cxx if probe.lang == "c++" else tc.cc
    stem = f"{probe.id}.{std.replace('+', 'x')}"
    src = workdir / f"{stem}{ext}"
    src.write_text(f"// probe {probe.id} from {probe.bundle}\n" + probe.body, encoding="utf-8")

    first_failure = None
    for flags in probe.alternatives("flags"):
        for libs in probe.alternatives("libs"):
            variant = " ".join(x for x in (flags, libs) if x)
            obj = workdir / f"{stem}{tc.obj}"
            exe = workdir / f"{stem}{tc.exe}"
            compile_cmd = [compiler, std_flag, *tc.cflags, *shlex.split(flags), "-c", str(src), "-o", str(obj)]
            result = run(compile_cmd, cwd=workdir)
            if result.returncode != 0:
                failure = {"status": "compile", "variant": variant, "diag": diagnostic(result.stderr + result.stdout)}
                first_failure = first_failure or failure
                break  # the link flags cannot help a compile error
            link_cmd = [compiler, *tc.ldflags, *shlex.split(flags), str(obj), "-o", str(exe), *shlex.split(libs)]
            result = run(link_cmd, cwd=workdir)
            if result.returncode != 0:
                failure = {"status": "link", "variant": variant, "diag": diagnostic(result.stderr + result.stdout)}
                first_failure = first_failure or failure
                continue
            try:
                result = run([str(exe)], cwd=workdir, timeout=60)
            except subprocess.TimeoutExpired:
                failure = {"status": "timeout", "variant": variant, "diag": "no exit within 60 s"}
                first_failure = first_failure or failure
                continue
            abnormal = probe.meta.get("expect") == "abnormal"
            passed = (result.returncode != 0) if abnormal else (result.returncode == 0)
            outcome = {
                "status": "pass" if passed else "run",
                "variant": variant,
                "exit": result.returncode,
                "stdout": first_line(result.stdout)[:200],
            }
            if not passed:
                outcome["diag"] = f"exit status {result.returncode}" + (
                    f": {first_line(result.stderr)[:200]}" if result.stderr.strip() else "")
            if passed and probe.meta.get("inspect") == "imports":
                outcome["imports"] = imports_of(tc, exe)
            if passed:
                return outcome
            first_failure = first_failure or outcome
    return first_failure


def macro_program(names: list[str]) -> str:
    lines = [
        "#include <version>",
        "#include <cstdio>",
        "#define PROBE_TEXT_OF(...) #__VA_ARGS__",
        "#define PROBE_TEXT(...) PROBE_TEXT_OF(__VA_ARGS__)",
        "int main(void) {",
    ]
    for name in names:
        attribute = re.fullmatch(r"__has_cpp_attribute\((\w+)\)", name)
        if attribute:
            cond = f"defined(__has_cpp_attribute) && __has_cpp_attribute({attribute.group(1)})"
            lines += [f"#if {cond}", f'  std::puts("{name}=" PROBE_TEXT({name}));', "#else",
                      f'  std::puts("{name}=-");', "#endif"]
        else:
            lines += [f"#ifdef {name}", f'  std::puts("{name}=" PROBE_TEXT({name}));', "#else",
                      f'  std::puts("{name}=-");', "#endif"]
    lines += ["  return 0;", "}", ""]
    return "\n".join(lines)


ENV_PROGRAM = r"""
#include <version>
#include <cstdio>
#if defined(__GLIBC__)
#include <gnu/libc-version.h>
#endif
#define PROBE_TEXT_OF(x) #x
#define PROBE_TEXT(x) PROBE_TEXT_OF(x)
int main(void) {
#if defined(_LIBCPP_VERSION)
  std::printf("stdlib=libc++ %d\n", static_cast<int>(_LIBCPP_VERSION));
#elif defined(__GLIBCXX__)
  std::printf("stdlib=libstdc++ %d (%ld)\n", static_cast<int>(_GLIBCXX_RELEASE), static_cast<long>(__GLIBCXX__));
#elif defined(_MSVC_STL_UPDATE)
  std::printf("stdlib=MSVC STL %d (%ld)\n", static_cast<int>(_MSVC_STL_VERSION), static_cast<long>(_MSVC_STL_UPDATE));
#else
  std::printf("stdlib=unknown\n");
#endif
#if defined(__GLIBC__)
  std::printf("libc=glibc %s\n", gnu_get_libc_version());
#endif
#if defined(_MSC_VER)
  std::printf("msvc_compat=%d\n", static_cast<int>(_MSC_FULL_VER));
#endif
#if defined(__apple_build_version__)
  std::printf("apple_clang=%d\n", static_cast<int>(__apple_build_version__));
#endif
  std::printf("clang=%s\n", __clang_version__);
  return 0;
}
"""


def run_program(tc: Toolchain, source: str, std: str, workdir: Path, name: str) -> tuple[bool, str]:
    src = workdir / f"{name}.cpp"
    exe = workdir / f"{name}{tc.exe}"
    src.write_text(source, encoding="utf-8")
    result = run([tc.cxx, tc.std_flag[std], *tc.cflags, *tc.ldflags, str(src), "-o", str(exe)])
    if result.returncode != 0:
        return False, diagnostic(result.stderr)
    result = run([str(exe)])
    return result.returncode == 0, result.stdout


def import_std(tc: Toolchain, std: str, stdlib: str, workdir: Path) -> dict:
    """Builds the standard library module the way a build system would, and imports it."""
    std_flag = tc.std_flag.get(std)
    if not std_flag:
        return {"status": "skip", "note": f"no {std} mode"}
    work = workdir / f"import_std.{std.replace('+', 'x')}"
    work.mkdir(exist_ok=True)
    extra = []
    if stdlib.startswith("MSVC STL"):
        verbose = run([tc.cxx, "-###", "-fms-runtime-lib=dll", "-x", "c++", "-c", os.devnull, "-o", os.devnull])
        match = re.search(r'"([^"]*[\\/]VC[\\/]+Tools[\\/]+MSVC[\\/]+[0-9.]+)[\\/]+include"', verbose.stderr)
        if not match:
            return {"status": "skip", "note": "the MSVC tools directory was not found"}
        source = Path(match.group(1).replace("\\\\", "\\")) / "modules" / "std.ixx"
        extra = ["-Wno-include-angled-in-module-purview", "-Wno-reserved-module-identifier"]
    elif tc.system == "macos":
        # A macOS build compiles against the SDK's libc++, so its std module has to be that
        # libc++'s. The manifest the compiler finds is the LLVM toolchain's, whose std.cppm needs
        # headers newer than the SDK's, so only Xcode's and the SDK's own are looked for.
        sdk = Path(tc.cflags[tc.cflags.index("-isysroot") + 1])
        developer = Path(run(["xcode-select", "-p"]).stdout.strip())
        candidates = [developer / "Toolchains" / "XcodeDefault.xctoolchain" / "usr" / "lib" / "libc++.modules.json",
                      sdk / "usr" / "lib" / "libc++.modules.json"]
        manifest = next((c for c in candidates if c.exists()), None)
        if manifest is None:
            return {"status": "skip",
                    "note": "no libc++.modules.json for the SDK's libc++ in " + ", ".join(str(c.parent) for c in candidates)}
    else:
        manifest_name = "libc++.modules.json" if stdlib.startswith("libc++") else "libstdc++.modules.json"
        manifest = Path(run([tc.cxx, *tc.cflags, f"-print-file-name={manifest_name}"]).stdout.strip())
        if not manifest.is_absolute() or not manifest.exists():
            return {"status": "skip", "note": f"{manifest_name} is not found by the compiler"}
    if not stdlib.startswith("MSVC STL"):
        data = json.loads(manifest.read_text(encoding="utf-8"))
        entry = next((m for m in data.get("modules", []) if m.get("logical-name") == "std"), None)
        if not entry:
            return {"status": "skip", "note": f"{manifest} names no std module"}
        source = (manifest.parent / entry["source-path"]).resolve()
        for directory in entry.get("local-arguments", {}).get("system-include-directories", []):
            extra += ["-isystem", str((manifest.parent / directory).resolve())]
        extra += ["-Wno-reserved-module-identifier"]
    if not source.exists():
        return {"status": "skip", "note": f"{source} does not exist"}

    pcm, std_obj = work / "std.pcm", work / f"std{tc.obj}"
    steps = [
        ("module", [tc.cxx, std_flag, *tc.cflags, *extra, "--precompile", "-x", "c++-module", str(source), "-o", str(pcm)]),
        ("module object", [tc.cxx, std_flag, *tc.cflags, "-Wno-unused-command-line-argument", "-c", str(pcm), "-o", str(std_obj)]),
    ]
    main_src, main_obj, exe = work / "main.cpp", work / f"main{tc.obj}", work / f"main{tc.exe}"
    main_src.write_text(
        "import std;\n"
        "int main(void) {\n"
        "  std::vector<int> v{1, 2, 3};\n"
        "  auto text = std::format(\"{}\", v.size());\n"
        "  return text == \"3\" && std::ranges::max(v) == 3 ? 0 : 1;\n"
        "}\n", encoding="utf-8")
    steps += [
        ("import", [tc.cxx, std_flag, *tc.cflags, f"-fmodule-file=std={pcm}", "-c", str(main_src), "-o", str(main_obj)]),
        ("link", [tc.cxx, *tc.ldflags, str(main_obj), str(std_obj), "-o", str(exe)]),
    ]
    for stage, cmd in steps:
        result = run(cmd, cwd=work, timeout=900)
        if result.returncode != 0:
            return {"status": "compile" if stage != "link" else "link",
                    "diag": f"{stage}: {diagnostic(result.stderr + result.stdout)}", "source": str(source)}
    result = run([str(exe)], cwd=work)
    return {"status": "pass" if result.returncode == 0 else "run", "source": str(source),
            "exit": result.returncode}


def status_cell(result: dict | None) -> str:
    if result is None:
        return "—"
    status = result["status"]
    mark = {"pass": "✅", "compile": "❌ compile", "link": "❌ link", "run": "❌ run",
            "timeout": "❌ timeout", "skip": "➖"}[status]
    if status == "pass" and result.get("variant"):
        mark += f" (`{result['variant']}`)"
    return mark


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "clang++"))
    parser.add_argument("--cc", default=os.environ.get("CC", "clang"))
    parser.add_argument("--std", default="c++23,c++26", help="C++ standards to try, comma-separated")
    parser.add_argument("--label", default=None, help="name of this run in the report (default: system)")
    parser.add_argument("--out", default=None, help="directory for report-<label>.{md,json}")
    parser.add_argument("--macos-min", default="26.0", help="macOS deployment target")
    parser.add_argument("--extra-cflags", default="")
    parser.add_argument("--extra-ldflags", default="")
    parser.add_argument("--only", default="", help="probe ids to run, comma-separated")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    args = parser.parse_args()

    tc = detect_toolchain(args)
    stds = [s.strip() for s in args.std.split(",") if s.strip()]
    probes = load_probes()
    if args.only:
        wanted = set(args.only.split(","))
        probes = [p for p in probes if p.id in wanted]
    label = args.label or f"{tc.system}-{tc.arch}"
    out_dir = Path(args.out) if args.out else HERE / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="unround-probe-") as tmp:
        workdir = Path(tmp)
        ok, env_text = run_program(tc, ENV_PROGRAM, "c++23", workdir, "env")
        env = dict(line.split("=", 1) for line in env_text.splitlines() if "=" in line) if ok else {"error": env_text}
        stdlib = env.get("stdlib", "unknown")

        jobs = []
        for probe in probes:
            if probe.lang == "c++":
                order = CXX_STDS.index(probe.min_std)
                for std in stds:
                    if CXX_STDS.index(std) >= order:
                        jobs.append((probe, std))
            else:
                jobs.append((probe, "c23"))
        results: dict[str, dict[str, dict]] = {}
        with cf.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(build_and_run, p, s, tc, workdir): (p, s) for p, s in jobs}
            for future in cf.as_completed(futures):
                probe, std = futures[future]
                results.setdefault(probe.id, {})[std] = future.result()

        macro_names = []
        for probe in probes:
            if probe.lang == "c++":
                for name in probe.meta.get("macro", "").split(","):
                    name = name.strip()
                    if name and name not in macro_names:
                        macro_names.append(name)
        macro_names += [m for m in EXTRA_MACROS if m not in macro_names]
        macros: dict[str, dict[str, str]] = {}
        for std in stds:
            ok, text = run_program(tc, macro_program(macro_names), std, workdir, f"macros.{std.replace('+', 'x')}")
            for line in text.splitlines() if ok else []:
                name, _, value = line.partition("=")
                macros.setdefault(name, {})[std] = value.rstrip("L")

        modules = {std: import_std(tc, std, stdlib, workdir) for std in stds} if not args.only else {}

    version = run([tc.cxx, "--version"]).stdout.splitlines()
    report = {
        "label": label,
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "system": tc.system, "arch": tc.arch, "os": platform.platform(),
        "cxx": tc.cxx, "cc": tc.cc, "compiler": version[0] if version else "?",
        "target": next((v.split(":", 1)[1].strip() for v in version if v.startswith("Target:")), "?"),
        "cflags": tc.cflags, "ldflags": tc.ldflags, "std_flags": tc.std_flag,
        "environment": env, "standards": stds,
        "probes": [{"id": p.id, "lang": p.lang, "bundle": p.bundle, "meta": p.meta,
                    "results": results.get(p.id, {})} for p in probes],
        "macros": macros, "import_std": modules,
    }
    (out_dir / f"report-{label}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / f"report-{label}.md").write_text(markdown(report), encoding="utf-8")

    passed = sum(1 for p in report["probes"] for r in p["results"].values() if r["status"] == "pass")
    total = sum(1 for p in report["probes"] for r in p["results"].values() if r["status"] != "skip")
    print(f"{label}: {passed} of {total} probe builds pass; report in {out_dir / f'report-{label}.md'}")
    return 0


def markdown(r: dict) -> str:
    env = r["environment"]
    out = [f"# Feature probes: {r['label']}", ""]
    out += [f"- Date: {r['date']}", f"- System: {r['os']} ({r['arch']})",
            f"- Compiler: {r['compiler']} (target {r['target']})",
            f"- C++ library: {env.get('stdlib', '?')}"]
    if "libc" in env:
        out.append(f"- C library: {env['libc']}")
    out += [f"- Compile flags: `{' '.join(r['cflags'])}`", f"- Link flags: `{' '.join(r['ldflags'])}`",
            f"- Standard flags: {', '.join(f'{k} → `{v}`' for k, v in r['std_flags'].items())}", ""]
    stds = r["standards"]
    for lang, title in (("c++", "C++"), ("c", "C23")):
        rows = [p for p in r["probes"] if p["lang"] == lang]
        if not rows:
            continue
        columns = stds if lang == "c++" else ["c23"]
        out += [f"## {title} probes", "",
                "| Probe | Feature | Paper | " + " | ".join(columns) + " | First diagnostic or output |",
                "|---|---|---|" + "---|" * len(columns) + "---|"]
        for p in rows:
            cells = [status_cell(p["results"].get(s)) for s in columns]
            notes = []
            for s in columns:
                res = p["results"].get(s)
                if not res:
                    continue
                if res.get("diag"):
                    notes.append(f"{s}: {res['diag']}")
                elif res.get("stdout"):
                    notes.append(f"{s}: {res['stdout']}")
                if res.get("imports"):
                    notes.append("needs " + ", ".join(res["imports"]))
            note = "; ".join(dict.fromkeys(notes)).replace("|", "\\|")
            out.append(f"| `{p['id']}` | {p['meta'].get('title', '')} | {p['meta'].get('paper', '')} | "
                       + " | ".join(cells) + f" | {note} |")
        out.append("")
    if r["import_std"]:
        out += ["## import std", "", "| Standard | Result | Detail |", "|---|---|---|"]
        for std, res in r["import_std"].items():
            detail = res.get("diag") or res.get("note") or res.get("source", "")
            out.append(f"| {std} | {status_cell(res)} | {str(detail).replace('|', '/')} |")
        out.append("")
    out += ["## Feature-test macros", "", "| Macro | " + " | ".join(stds) + " |", "|---|" + "---|" * len(stds)]
    for name, values in r["macros"].items():
        out.append(f"| `{name}` | " + " | ".join(values.get(s, "?") for s in stds) + " |")
    out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    sys.exit(main())
