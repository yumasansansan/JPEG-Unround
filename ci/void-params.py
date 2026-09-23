#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Checks that every empty parameter list of the project's C and C++ is written (void).

    python3 ci/void-params.py --build build/<preset> [<file>...]
    python3 ci/void-params.py --self-test

In this project a function, a member function, a lambda and a function type say
that they take nothing by writing (void), and never () -- a rule no compiler and no
check of clang-tidy has: C23 made f() mean f(void), C++ always did, and the syntax
tree gives both forms the same type, so only the text tells them apart. And the
text alone cannot tell a declaration from a call: `S s();` declares a function
while `s = f();` calls one, and which a line is depends on what its names are.

So the compiler is asked. Every empty pair of parentheses of a file -- but for the
name operator() -- and every lambda introducer that is followed by no parameter
list at all are candidates, and three compilations of copies of the file decide:

  A. With "void void" in every candidate, which is an error wherever it is read at
     all: in a parameter list and in an argument list alike. A candidate without
     an error here is in code that is not compiled (an inactive #if branch), and
     it is left alone.
  B. With "void" in every candidate. (void) is a parameter list and never an
     argument list, so a call, a construction or a value-initialization gains an
     error at the inserted void, and a parameter list gains none. A candidate
     that compiles here is a parameter list written ().
  C. Each such candidate alone, to be sure that no error elsewhere in B kept the
     compiler from reading it.

The flags of each file are those of the build's compile commands (build/<preset>/
compile_commands.json); a header, or a source file the preset does not compile,
takes those of the nearest file of its language that the preset does compile, and
is compiled as a file of its own. The copies go to a temporary directory, with the
file's own directory first on the quoted include path, so that its includes find
what the original's find. Preprocessor directives are left out, and so are macros.

With --self-test, the checker runs on ci/void-params-cases, whose lines that end
in the comment "want: void" are the places it has to report, and nothing else.
"""

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHECKED_DIRECTORIES = ("c", "cpp", "tests")
C_SUFFIXES = (".c", ".h")
CXX_SUFFIXES = (".cpp", ".hpp", ".cc", ".cxx", ".hh")
# What may follow a lambda introducer that has no parameter list.
LAMBDA_FOLLOWERS = {"{", "mutable", "constexpr", "consteval", "noexcept", "->", "[", "requires", "static"}


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class Candidate:
    """Where the inserted text goes, the span that encloses it, and which kind of place it is."""

    insert_at: int
    span_start: int
    span_end: int
    lambda_introducer: bool


TOKEN = re.compile(
    r"""
    (?P<space>\s+)
  | (?P<line_comment>//[^\n]*)
  | (?P<block_comment>/\*.*?\*/)
  | (?P<raw_string>(?:u8|[uUL])?R"(?P<delimiter>[^()\\\s]{0,16})\(.*?\)(?P=delimiter)")
  | (?P<string>(?:u8|[uUL])?"(?:[^"\\\n]|\\.)*")
  | (?P<char>(?:u8|[uUL])?'(?:[^'\\\n]|\\.)*')
  | (?P<number>\.?\d(?:[\w.]|[eEpP][+-]|'(?=\w))*)
  | (?P<identifier>[A-Za-z_]\w*)
  | (?P<punctuator>->\*?|::|\.\.\.|&&|\|\||\+\+|--|<<=?|>>=?|[-+*/%&|^!=<>]=?|[()\[\]{};,.?:~#])
  | (?P<other>.)
    """,
    re.VERBOSE | re.DOTALL,
)
DIRECTIVE = re.compile(r"[ \t]*#")


def directive_end(text: str, position: int) -> int:
    """Where a directive that starts at position ends: at its line's end, past escaped line ends."""
    end = position
    while True:
        newline = text.find("\n", end)
        if newline < 0:
            return len(text)
        escaped = text[max(newline - 1, 0)] == "\\" or text[max(newline - 2, 0) : newline] == "\\\r"
        end = newline + 1
        if not escaped:
            return end


def tokens_of(text: str) -> list[Token]:
    """The tokens of a file that the compiler sees, without comments and directives."""
    tokens: list[Token] = []
    position = 0
    line_start = True
    while position < len(text):
        if line_start and DIRECTIVE.match(text, position):
            position = directive_end(text, position)
            continue
        match = TOKEN.match(text, position)
        if match is None:
            position += 1
            continue
        kind = match.lastgroup
        if kind in {"space", "line_comment", "block_comment"}:
            # Comments are white space; a line that begins after one begins afresh.
            line_start = line_start or "\n" in match.group()
        else:
            tokens.append(Token(match.group(), match.start(), match.end()))
            line_start = False
        position = match.end()
    return tokens


def candidates_of(text: str, *, cxx: bool) -> list[Candidate]:
    tokens = tokens_of(text)
    found: list[Candidate] = []
    for index, token in enumerate(tokens):
        following = tokens[index + 1] if index + 1 < len(tokens) else None
        previous = tokens[index - 1] if index > 0 else None
        if token.text == "(" and following is not None and following.text == ")":
            if previous is not None and previous.text == "operator":
                continue
            found.append(Candidate(following.start, token.start, following.end, lambda_introducer=False))
        elif (
            cxx
            and token.text == "]"
            and following is not None
            and following.text in LAMBDA_FOLLOWERS
            and not (previous is not None and previous.text == "]")
        ):
            found.append(Candidate(token.end, token.start, token.end, lambda_introducer=True))
    return found


def insertion(candidate: Candidate, *, doubled: bool) -> str:
    inner = "void void" if doubled else "void"
    return f"({inner})" if candidate.lambda_introducer else inner


def modified(text: str, chosen: list[Candidate], *, doubled: bool) -> tuple[str, list[tuple[int, int]]]:
    """The text with the insertions made, and each candidate's span in it, in the order given."""
    pieces: list[str] = []
    spans: list[tuple[int, int]] = [(0, 0)] * len(chosen)
    shift = 0
    last = 0
    for index in sorted(range(len(chosen)), key=lambda item: chosen[item].insert_at):
        candidate = chosen[index]
        inserted = insertion(candidate, doubled=doubled)
        pieces.append(text[last : candidate.insert_at])
        pieces.append(inserted)
        last = candidate.insert_at
        spans[index] = (candidate.span_start + shift, candidate.span_end + shift + len(inserted))
        shift += len(inserted)
    pieces.append(text[last:])
    return "".join(pieces), spans


def split_command(command: str) -> list[str]:
    """A compile command, split as the shell of its generator would split it."""
    if os.name != "nt":
        return shlex.split(command)
    arguments: list[str] = []
    current: list[str] = []
    quoted = False
    has_argument = False
    index = 0
    while index < len(command):
        character = command[index]
        if character == "\\" and index + 1 < len(command) and command[index + 1] == '"':
            current.append('"')
            has_argument = True
            index += 2
            continue
        if character == '"':
            quoted = not quoted
            has_argument = True
        elif character in " \t" and not quoted:
            if has_argument:
                arguments.append("".join(current))
                current = []
                has_argument = False
        else:
            current.append(character)
            has_argument = True
        index += 1
    if has_argument:
        arguments.append("".join(current))
    return arguments


@dataclass(frozen=True)
class Command:
    directory: Path
    arguments: list[str]
    file: Path


def load_commands(build: Path) -> list[Command]:
    database = build / "compile_commands.json"
    if not database.is_file():
        message = f"{database} does not exist: configure the preset first"
        raise SystemExit(message)
    commands = []
    for entry in json.loads(database.read_text(encoding="utf-8")):
        arguments = entry["arguments"] if "arguments" in entry else split_command(entry["command"])
        directory = Path(entry["directory"])
        file = Path(entry["file"])
        if not file.is_absolute():
            file = directory / file
        commands.append(Command(directory, arguments, file.resolve()))
    return commands


def syntax_only(command: Command, source: Path, copy: Path, language: str | None) -> list[str]:
    """The arguments that compile copy, standing for source, for its syntax only."""
    arguments = [command.arguments[0]]
    skip_next = False
    for argument in command.arguments[1:]:
        if skip_next:
            skip_next = False
            continue
        if argument in {"-o", "-MF", "-MT", "-MQ"}:
            skip_next = True
            continue
        if argument in {"-c", "-MD", "-MMD", "--"}:
            continue
        if not argument.startswith("-") and (command.directory / argument).resolve() == command.file:
            continue
        arguments.append(argument)
    arguments += ["-fsyntax-only", "-w", "-ferror-limit=0", "-fno-color-diagnostics", "-fno-caret-diagnostics"]
    arguments += ["-iquote", str(source.parent)]
    if language is not None:
        arguments += ["-x", language]
    arguments.append(str(copy))
    return arguments


ERROR = re.compile(r"^(?P<path>.*?):(?P<line>\d+):(?P<column>\d+): (?:fatal )?error: ", re.MULTILINE)


def error_offsets(output: str, copy: Path, text: str) -> list[int]:
    """The offsets in text of the errors that output reports in copy."""
    line_starts = [0] + [match.end() for match in re.finditer("\n", text)]
    offsets = []
    for match in ERROR.finditer(output):
        try:
            same = Path(match.group("path")).resolve() == copy.resolve()
        except OSError:
            same = False
        line = int(match.group("line")) - 1
        if not same or line >= len(line_starts):
            continue
        start = line_starts[line]
        end = line_starts[line + 1] if line + 1 < len(line_starts) else len(text)
        # Clang counts columns in bytes of UTF-8.
        column = int(match.group("column")) - 1
        prefix = text[start:end].encode("utf-8")[:column].decode("utf-8", errors="ignore")
        offsets.append(start + len(prefix))
    return offsets


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    column: int
    lambda_introducer: bool


@dataclass(frozen=True)
class Source:
    """A file under check, with the compile command that checks it and the language to name, if any."""

    path: Path
    text: str
    command: Command
    language: str | None


class Checker:
    def __init__(self, commands: list[Command], scratch: Path) -> None:
        self.commands = commands
        self.scratch = scratch
        self.compilations = 0

    def command_for(self, path: Path) -> tuple[Command, str | None]:
        for command in self.commands:
            if command.file == path:
                return command, None
        cxx = path.suffix in CXX_SUFFIXES
        suffixes = CXX_SUFFIXES if cxx else C_SUFFIXES
        same_language = [command for command in self.commands if command.file.suffix in suffixes]
        if not same_language:
            message = f"no compile command of {'C++' if cxx else 'C'} to borrow for {path}"
            raise SystemExit(message)

        def shared(command: Command) -> int:
            return len(os.path.commonpath([command.file, path]))

        return max(same_language, key=shared), ("c++" if cxx else "c")

    def compile(self, source: Source, text: str) -> tuple[str, Path]:
        """Compiles text in place of the source, and returns what the compiler said and where the copy was."""
        directory = Path(tempfile.mkdtemp(dir=self.scratch))
        copy = directory / source.path.name
        copy.write_text(text, encoding="utf-8", newline="")
        result = subprocess.run(
            syntax_only(source.command, source.path, copy, source.language),
            cwd=source.command.directory,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        self.compilations += 1
        return result.stdout + result.stderr, copy

    def erred(self, source: Source, chosen: list[Candidate], *, doubled: bool) -> list[bool]:
        """Whether the compiler reports an error inside each chosen candidate once it is changed."""
        changed, spans = modified(source.text, chosen, doubled=doubled)
        output, copy = self.compile(source, changed)
        offsets = error_offsets(output, copy, changed)
        return [any(start <= offset <= end for offset in offsets) for start, end in spans]

    def check(self, path: Path) -> list[Finding]:
        text = path.read_text(encoding="utf-8")
        candidates = candidates_of(text, cxx=path.suffix in CXX_SUFFIXES)
        command, language = self.command_for(path)
        source = Source(path, text, command, language)

        output, _ = self.compile(source, text)
        if ERROR.search(output):
            message = f"{path} does not compile as it is, so nothing can be told from it:\n{output}"
            raise SystemExit(message)
        if not candidates:
            return []

        read = self.erred(source, candidates, doubled=True)
        active = [candidate for candidate, error in zip(candidates, read, strict=True) if error]
        if not active:
            return []
        called = self.erred(source, active, doubled=False)
        findings = []
        for candidate in (candidate for candidate, error in zip(active, called, strict=True) if not error):
            if not self.erred(source, [candidate], doubled=False)[0]:
                line = text.count("\n", 0, candidate.span_start) + 1
                column = candidate.span_start - (text.rfind("\n", 0, candidate.span_start) + 1) + 1
                findings.append(Finding(path, line, column, candidate.lambda_introducer))
        return findings


def project_files() -> list[Path]:
    listed = subprocess.run(
        ["git", "ls-files", "--", *CHECKED_DIRECTORIES], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    return sorted((ROOT / name).resolve() for name in listed if Path(name).suffix in C_SUFFIXES + CXX_SUFFIXES)


def describe(finding: Finding) -> str:
    try:
        shown: Path = finding.path.relative_to(ROOT)
    except ValueError:
        shown = finding.path
    what = "a lambda without a parameter list" if finding.lambda_introducer else "an empty parameter list"
    return f"{shown.as_posix()}:{finding.line}:{finding.column}: {what}; write (void)"


def self_test(scratch: Path) -> int:
    cases = ROOT / "ci" / "void-params-cases"
    compilers = {".c": ("clang", ["-std=c23"]), ".cpp": ("clang++", ["-std=c++26"])}
    failures = 0
    for path in sorted(cases.iterdir()):
        compiler, flags = compilers[path.suffix]
        command = Command(cases, [compiler, *flags, "-c", str(path)], path.resolve())
        checker = Checker([command], scratch)
        found = {finding.line for finding in checker.check(path.resolve())}
        lines = path.read_text(encoding="utf-8").splitlines()
        wanted = {number for number, line in enumerate(lines, start=1) if line.rstrip().endswith("// want: void")}
        for number in sorted(wanted - found):
            print(f"{path.name}:{number}: not reported: {lines[number - 1].strip()}")
            failures += 1
        for number in sorted(found - wanted):
            print(f"{path.name}:{number}: reported, and it should not be: {lines[number - 1].strip()}")
            failures += 1
        print(f"== {path.name}: {len(found & wanted)} of {len(wanted)} reported, {checker.compilations} compilations")
    if failures:
        print(f"error: the checker is wrong in {failures} places")
        return 1
    print("The checker reports what it has to, and nothing else.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--build", type=Path, help="the build directory whose compile commands to use")
    parser.add_argument("--self-test", action="store_true", help="check the checker on ci/void-params-cases")
    parser.add_argument("files", nargs="*", type=Path)
    options = parser.parse_args()

    scratch = Path(tempfile.mkdtemp(prefix="void-params-"))
    try:
        if options.self_test:
            return self_test(scratch)
        if options.build is None:
            parser.error("--build is required")
        checker = Checker(load_commands(options.build.resolve()), scratch)
        files = [path.resolve() for path in options.files] or project_files()
        print(f"== the parameter lists of {len(files)} files of C and C++")
        findings = []
        for path in files:
            findings += checker.check(path)
        for finding in findings:
            print(describe(finding))
        print(f"== {checker.compilations} compilations")
        if findings:
            print("error: a function, a lambda or a function type that takes nothing is written with (void)")
            return 1
        print("Every function says what it takes.")
        return 0
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
