# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Checks that the Rust code computes in vectors of the processor's full width.

    python ci/vectorization.py kernels LISTING --width {128,256,512} [--neon]
    python ci/vectorization.py library LISTING --width {128,256,512} [--neon] --allowed FILE
    python ci/vectorization.py self-test

LISTING is assembly that rustc wrote (--emit asm, with -C debuginfo=1 for the source
lines), built as ci/vectorization.sh builds it: for x86-64, with the vectorization of
the remainders of loops turned off, which LLVM otherwise does at half the width, so
that what is narrower than the full width is code that the compiler kept from it.
The widths are those of the registers, xmm 128, ymm 256 and zmm 512; on AArch64
(--neon) a vector is two doubles, 128 bits, in either syntax of NEON (`fadd v0.2d,
v1.2d, v2.2d`, or Apple's `fadd.2d v0, v1, v2`), and narrower is scalar.

The loops are the natural loops of each function's control flow graph: an edge to a
block that dominates where it comes from closes a loop, of the blocks that reach it
without passing that block. A loop's own instructions are those of its blocks that
no loop within it holds. A loop whose own instructions compute in doubles is
  vector     if they are arithmetic on full vectors and none on narrower ones,
  remainder  if the source lines of its arithmetic (line 0, of no line, aside) are
             among those of one vector loop of the function: the loop after it for the
             last values, or the copy that runs where a check at run time keeps the
             vector loop from running,
  outer      if it holds other loops, its own arithmetic done once for each time
             around them, or if the source lines of its arithmetic are among those
             of such a loop's own (a copy of the outer loop for a case in which its
             inner loops do not run), and
  scalar     otherwise.

kernels: rust/unround/examples/vectorization.rs puts each kernel of the solvers in a
function of its own, named inspect_*. Each must hold a vector loop, no scalar loop,
and, on x86-64, no arithmetic on narrower vectors.

library: every scalar loop of the library must be in a function that FILE allows,
with the reason: FILE is TOML, its table `functions` the prefixes of the names,
demangled, of functions whose loops may be scalar, and its table `sums` those of
functions whose loops may be scalar where they only add, as sums in a fixed order do.
Arithmetic on narrower vectors outside loops is a few values computed side by side,
which no wider vector would hold, and is not counted.

self-test: checks the reading of small listings of each kind, and the demangling.
"""

import argparse
import contextlib
import io
import re
import sys
import tomllib
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

# Arithmetic in doubles, x86-64: packed (pd) or scalar (sd).
X86_ARITHMETIC = re.compile(
    r"^v?(add|sub|mul|div|sqrt|max|min|fmadd\d*|fmsub\d*|fnmadd\d*|fnmsub\d*|fmaddsub\d*|fmsubadd\d*"
    r"|cmp\w*|blendv|round|rndscale|range|getexp|getmant|scalef|cvt\w*2)(pd|sd)\b"
)
X86_JUMP = re.compile(r"^(j\w+)\s+(\S+)$")
X86_END = re.compile(r"^(ret\w*|ud2|int3|hlt)\b")
X86_REGISTER = re.compile(r"%([xyz])mm\d+")
X86_WIDTHS = {"x": 128, "y": 256, "z": 512}
# Arithmetic in doubles, AArch64: on vectors of two (v.2d, or .2d on the mnemonic in
# Apple's syntax), or scalar (d registers).
ARM_ARITHMETIC = re.compile(
    r"^(fadd|fsub|fmul|fdiv|fsqrt|fmax\w*|fmin\w*|fmla|fmls|fmadd|fmsub|fnmadd|fnmsub|fnmul|fneg|fabs"
    r"|fcm\w+|fcmp\w*|fcsel|frint\w|fcvt\w+|scvtf|ucvtf)(?:\.(\w+))?\s"
)
ARM_VECTOR = re.compile(r"\bv\d+\.2d\b")
ARM_SCALAR = re.compile(r"\bd\d+\b")
ARM_JUMP = re.compile(r"^(b|b\.\w+|cbz|cbnz|tbz|tbnz)\s+(?:.*,\s*)?(\S+)$")
ARM_END = re.compile(r"^(ret|br|brk|udf)\b")
UNCONDITIONAL = {"jmp", "jmpq", "b"}
# The operations of sums.
ADDITIONS = {"add", "sub", "fadd", "fsub"}
LABEL = re.compile(r"^([A-Za-z_$.][\w$.@]*):")
SYMBOL = re.compile(r"^_{1,2}(R|ZN)")
CV_FILE = re.compile(r'^\.cv_file\s+(\d+)\s+"([^"]+)"')
DWARF_FILE = re.compile(r'^\.file\s+(\d+)\s+"([^"]+)"(?:\s+"([^"]+)")?')
CV_LOC = re.compile(r"^\.cv_loc\s+\d+\s+(\d+)\s+(\d+)")
DWARF_LOC = re.compile(r"^\.loc\s+(\d+)\s+(\d+)")
# The basic types of the v0 scheme.
BASIC_TYPES = {
    "a": "i8", "b": "bool", "c": "char", "d": "f64", "e": "str", "f": "f32", "h": "u8",
    "i": "isize", "j": "usize", "l": "i32", "m": "u32", "n": "i128", "o": "u128",
    "s": "i16", "t": "u16", "u": "()", "v": "...", "x": "i64", "y": "u64", "z": "!", "p": "_",
}  # fmt: skip


class Demangler:
    """Rust's symbol names of the v0 scheme (RFC 2603), as paths: `crate::module::name`,
    an inherent method as `crate::Type::method`, a trait's as `<Type as Trait>::name`."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.at = 0

    def peek(self) -> str:
        return self.text[self.at] if self.at < len(self.text) else ""

    def take(self) -> str:
        character = self.peek()
        if not character:
            message = "the name ends early"
            raise ValueError(message)
        self.at += 1
        return character

    def skip(self, character: str) -> bool:
        """Whether the next character is this one, which is then passed."""
        if self.peek() == character:
            self.at += 1
            return True
        return False

    def base62(self) -> int:
        if self.skip("_"):
            return 0
        value = 0
        while (character := self.take()) != "_":
            if character.isdigit():
                digit = ord(character) - ord("0")
            elif character.islower():
                digit = ord(character) - ord("a") + 10
            elif character.isupper():
                digit = ord(character) - ord("A") + 36
            else:
                message = f"not base 62: {character}"
                raise ValueError(message)
            value = value * 62 + digit
        return value + 1

    def decimal(self) -> int:
        # 0, or digits from 1 to 9 first: a 0 ends the number.
        start = self.at
        if self.skip("0"):
            return 0
        while self.peek().isdigit():
            self.at += 1
        if start == self.at:
            message = "no number"
            raise ValueError(message)
        return int(self.text[start : self.at])

    def disambiguator(self) -> int:
        return self.base62() if self.skip("s") else 0

    def identifier(self) -> tuple[str, int]:
        number = self.disambiguator()
        self.skip("u")
        length = self.decimal()
        self.skip("_")
        name = self.text[self.at : self.at + length]
        self.at += length
        return name, number

    def back(self, parse: Callable[[], str]) -> str:
        position = self.base62()
        saved, self.at = self.at, position
        result = parse()
        self.at = saved
        return result

    def path(self) -> str:
        tag = self.take()
        table: dict[str, Callable[[], str]] = {
            "C": lambda: self.identifier()[0],
            "M": self.inherent,
            "X": self.implementation,
            "Y": self.definition,
            "N": self.nested,
            "I": self.instance,
            "B": lambda: self.back(self.path),
        }
        parse = table.get(tag)
        if parse is None:
            message = f"not a path: {tag}"
            raise ValueError(message)
        return parse()

    def inherent(self) -> str:
        self.disambiguator()
        self.path()
        return self.type()

    def implementation(self) -> str:
        self.disambiguator()
        self.path()
        return self.definition()

    def definition(self) -> str:
        own = self.type()
        return f"<{own} as {self.path()}>"

    def nested(self) -> str:
        namespace = self.take()
        inner = self.path()
        name, number = self.identifier()
        if namespace == "C":
            return f"{inner}::{{closure#{number}}}"
        if namespace == "S":
            return f"{inner}::{{shim#{number}}}"
        if namespace.isupper():
            return f"{inner}::{{{namespace}:{name}#{number}}}"
        return f"{inner}::{name}"

    def instance(self) -> str:
        inner = self.path()
        arguments = []
        while not self.skip("E"):
            arguments.append(self.generic())
        return f"{inner}::<{', '.join(arguments)}>"

    def generic(self) -> str:
        if self.skip("L"):
            self.base62()
            return "'_"
        if self.skip("K"):
            return self.constant()
        return self.type()

    def constant(self) -> str:
        if self.skip("B"):
            return self.back(self.constant)
        if self.skip("p"):
            return "_"
        kind = self.type()
        negative = self.skip("n")
        start = self.at
        while self.peek() != "_":
            self.take()
        digits = self.text[start : self.at]
        self.at += 1
        value = int(digits, 16) if digits else 0
        if kind == "bool":
            return "true" if value else "false"
        return f"{'-' if negative else ''}{value}"

    def type(self) -> str:
        tag = self.peek()
        if tag in BASIC_TYPES:
            self.at += 1
            return BASIC_TYPES[tag]
        if tag in "CMXYNI":
            return self.path()
        self.at += 1
        table: dict[str, Callable[[], str]] = {
            "B": lambda: self.back(self.type),
            "A": lambda: f"[{self.type()}; {self.constant()}]",
            "S": lambda: f"[{self.type()}]",
            "T": self.members,
            "R": lambda: "&" + self.referred(),
            "Q": lambda: "&mut " + self.referred(),
            "P": lambda: "*const " + self.type(),
            "O": lambda: "*mut " + self.type(),
            "F": self.function,
            "D": self.dynamic,
        }
        parse = table.get(tag)
        if parse is None:
            message = f"not a type: {tag}"
            raise ValueError(message)
        return parse()

    def members(self) -> str:
        members = []
        while not self.skip("E"):
            members.append(self.type())
        return f"({', '.join(members)})"

    def referred(self) -> str:
        if self.skip("L"):
            self.base62()
        return self.type()

    def binder(self) -> None:
        if self.skip("G"):
            self.base62()

    def function(self) -> str:
        self.binder()
        self.skip("U")
        if self.skip("K") and not self.skip("C"):
            self.identifier()
        parameters = []
        while not self.skip("E"):
            parameters.append(self.type())
        return f"fn({', '.join(parameters)}) -> {self.type()}"

    def dynamic(self) -> str:
        self.binder()
        traits = []
        while not self.skip("E"):
            traits.append(self.path())
            while self.skip("p"):
                self.identifier()
                self.type()
        if self.skip("L"):
            self.base62()
        return "dyn " + " + ".join(traits)


def demangled(symbol: str) -> str:
    """The path of a symbol of the v0 scheme, or of the legacy one; the symbol itself
    where it is neither."""
    name = symbol.split(".llvm.", maxsplit=1)[0]
    name = name[1:] if name.startswith("__") else name
    if name.startswith("_R"):
        demangler = Demangler(name[2:])
        try:
            if demangler.peek().isdigit():
                demangler.decimal()
            return demangler.path()
        except ValueError:
            return symbol
        except RecursionError:
            return symbol
    parts, rest = [], name[3:] if name.startswith("_ZN") else ""
    while (match := re.match(r"\d+", rest)) is not None:
        length, digits = int(match.group()), len(match.group())
        parts.append(rest[digits : digits + length])
        rest = rest[digits + length :]
    if parts and re.fullmatch(r"h[0-9a-f]{16}", parts[-1]):
        parts.pop()
    return "::".join(parts) if parts else symbol


@dataclass
class Item:
    """A label, or an instruction and the source line it comes from."""

    label: str | None
    text: str
    place: str | None


@dataclass
class Function:
    symbol: str
    items: list[Item] = field(default_factory=list)


@dataclass
class Loop:
    """A natural loop: its blocks, and the arithmetic in doubles of its own instructions."""

    blocks: set[int]
    packed: Counter[int] = field(default_factory=Counter)
    scalar: int = 0
    places: set[str] = field(default_factory=set)
    adds_only: bool = True
    holds_loops: bool = False
    kind: str = ""


def source_file(line: str, files: dict[str, str]) -> bool:
    """Keeps the name of a file of the source that the line of the listing numbers;
    whether it was such a line."""
    match = CV_FILE.match(line) or DWARF_FILE.match(line)
    if match is None:
        return False
    named = match.group(3) if match.re is DWARF_FILE and match.group(3) else match.group(2)
    files[match.group(1)] = re.split(r"[\\/]+", named)[-1]
    return True


def source_place(line: str, files: dict[str, str]) -> str | None:
    """The line of the source that the line of the listing says the code after it is
    for, if it says one."""
    match = CV_LOC.match(line) or DWARF_LOC.match(line)
    return f"{files.get(match.group(1), '?')}:{match.group(2)}" if match else None


def functions(text: str) -> list[Function]:
    """The functions of a listing, with their labels and instructions."""
    result: list[Function] = []
    files: dict[str, str] = {}
    current: Function | None = None
    place: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if source_file(line, files):
            continue
        if match := LABEL.match(line):
            if SYMBOL.match(match.group(1)):
                current = Function(match.group(1))
                result.append(current)
            elif current is not None:
                current.items.append(Item(match.group(1), "", None))
            continue
        if current is None or not line:
            continue
        if (new := source_place(line, files)) is not None:
            place = new
        elif not line.startswith((".", "#", ";", "//", "@")):
            current.items.append(Item(None, line, place))
    return result


def x86_arithmetic(text: str) -> tuple[int, bool, str] | None:
    match = X86_ARITHMETIC.match(text.split(maxsplit=1)[0])
    if not match:
        return None
    if match.group(2) == "sd":
        return 64, False, match.group(1)
    widths = [X86_WIDTHS[register] for register in X86_REGISTER.findall(text)]
    return max(widths, default=128), True, match.group(1)


def arm_arithmetic(text: str) -> tuple[int, bool, str] | None:
    match = ARM_ARITHMETIC.match(text)
    if not match:
        return None
    if match.group(2) == "2d" or ARM_VECTOR.search(text):
        return 128, True, match.group(1)
    return (64, False, match.group(1)) if ARM_SCALAR.search(text) else None


def arithmetic(text: str, *, neon: bool) -> tuple[int, bool, str] | None:
    """The width of an instruction's arithmetic in doubles, whether it is packed, and
    the operation; None for other instructions."""
    return arm_arithmetic(text) if neon else x86_arithmetic(text)


def successors_of(found: list[list[Item]], starts: dict[str, int], *, neon: bool) -> list[list[int]]:
    """The blocks that each block goes on to: after a jump, where it goes, and the
    next unless it always goes; none after a return; the next otherwise."""
    jump, end = (ARM_JUMP, ARM_END) if neon else (X86_JUMP, X86_END)
    successors: list[list[int]] = []
    for index, block in enumerate(found):
        following = [index + 1] if index + 1 < len(found) else []
        last = block[-1].text if block else ""
        if match := jump.match(last):
            target = [starts[match.group(2)]] if match.group(2) in starts else []
            successors.append(target if match.group(1) in UNCONDITIONAL else target + following)
        elif end.match(last):
            successors.append([])
        else:
            successors.append(following)
    return successors


def blocks(function: Function, *, neon: bool) -> tuple[list[list[Item]], list[list[int]]]:
    """The basic blocks of a function and the successors of each: a block begins at a
    label that a jump goes to, and ends at a jump or a return. A jump to what is not a
    label of the function (a tail call, or through a register) leaves it."""
    jump, end = (ARM_JUMP, ARM_END) if neon else (X86_JUMP, X86_END)
    labels = {item.label for item in function.items if item.label}
    targets = set()
    for item in function.items:
        match = jump.match(item.text) if item.label is None else None
        if match and match.group(2) in labels:
            targets.add(match.group(2))
    found: list[list[Item]] = [[]]
    starts: dict[str, int] = {}
    for item in function.items:
        if item.label is None:
            found[-1].append(item)
            if jump.match(item.text) or end.match(item.text):
                found.append([])
        elif item.label in targets:
            if found[-1]:
                found.append([])
            starts[item.label] = len(found) - 1
    return found, successors_of(found, starts, neon=neon)


def reverse_postorder(successors: list[list[int]]) -> list[int]:
    """The blocks reached from block 0, in the reverse of the order in which a search
    in depth leaves them."""
    order: list[int] = []
    seen = {0}
    stack = [(0, iter(successors[0]))]
    while stack:
        node, children = stack[-1]
        child = next(children, None)
        if child is None:
            order.append(node)
            stack.pop()
        elif child not in seen:
            seen.add(child)
            stack.append((child, iter(successors[child])))
    order.reverse()
    return order


def dominators(order: list[int], predecessors: dict[int, list[int]]) -> dict[int, int]:
    """The immediate dominator of each block, in reverse postorder (Cooper, Harvey and
    Kennedy's iteration)."""
    rank = {node: index for index, node in enumerate(order)}
    dominator: dict[int, int] = {order[0]: order[0]}

    def meet(a: int, b: int) -> int:
        while a != b:
            while rank[a] > rank[b]:
                a = dominator[a]
            while rank[b] > rank[a]:
                b = dominator[b]
        return a

    changed = True
    while changed:
        changed = False
        for node in order[1:]:
            done = [parent for parent in predecessors[node] if parent in dominator]
            new = done[0]
            for parent in done[1:]:
                new = meet(parent, new)
            if dominator.get(node) != new:
                dominator[node] = new
                changed = True
    return dominator


def natural_loops(successors: list[list[int]]) -> list[set[int]]:
    """The natural loops of a control flow graph whose entry is block 0, each as its
    blocks; loops with the same header are one loop."""
    order = reverse_postorder(successors)
    predecessors: dict[int, list[int]] = {node: [] for node in order}
    for node in order:
        for child in successors[node]:
            predecessors[child].append(node)
    dominator = dominators(order, predecessors)

    def dominates(head: int, node: int) -> bool:
        while node not in (head, 0):
            node = dominator[node]
        return node == head

    loops: dict[int, set[int]] = {}
    for node in order:
        for head in successors[node]:
            if head in dominator and dominates(head, node):
                body = loops.setdefault(head, {head})
                work = [node]
                while work:
                    member = work.pop()
                    if member not in body:
                        body.add(member)
                        work.extend(predecessors[member])
    return list(loops.values())


def count_arithmetic(loop: Loop, own: list[Item], *, neon: bool) -> None:
    """Counts the arithmetic in doubles of a loop's own instructions."""
    for item in own:
        kind = arithmetic(item.text, neon=neon)
        if kind is None:
            continue
        # Line 0 is code that no line of the source is for.
        if item.place and not item.place.endswith(":0"):
            loop.places.add(item.place)
        bits, is_packed, operation = kind
        loop.adds_only &= operation in ADDITIONS
        if is_packed:
            loop.packed[bits] += 1
        else:
            loop.scalar += 1


def loops(function: Function, width: int, *, neon: bool) -> list[Loop]:
    """The loops of a function whose own instructions compute in doubles, classified."""
    found_blocks, successors = blocks(function, neon=neon)
    found = [Loop(body) for body in natural_loops(successors)]
    for loop in found:
        inner = [other.blocks for other in found if other is not loop and other.blocks < loop.blocks]
        loop.holds_loops = bool(inner)
        own = [item for block in sorted(loop.blocks.difference(*inner)) for item in found_blocks[block]]
        count_arithmetic(loop, own, neon=neon)
    found = [loop for loop in found if loop.packed or loop.scalar]
    for loop in found:
        if loop.packed.get(width, 0) and all(bits >= width for bits in loop.packed):
            loop.kind = "vector"
    vectors = [loop for loop in found if loop.kind == "vector"]
    for loop in found:
        if loop.kind:
            continue
        if loop.places and any(loop.places <= vector.places for vector in vectors):
            loop.kind = "remainder"
        else:
            loop.kind = "outer" if loop.holds_loops else "scalar"
    # A copy of an outer loop, for a case in which its inner loops do not run, does no
    # more than the outer loop's own work.
    outers = [loop for loop in found if loop.kind == "outer"]
    for loop in found:
        if loop.kind == "scalar" and loop.places and any(loop.places <= outer.places for outer in outers):
            loop.kind = "outer"
    return found


def narrower(function: Function, width: int, *, neon: bool) -> int:
    """The instructions of a function that compute on vectors narrower than the width."""
    kinds = [arithmetic(item.text, neon=neon) for item in function.items if item.label is None]
    return sum(1 for kind in kinds if kind is not None and kind[1] and kind[0] < width)


def describe(loop: Loop) -> str:
    places = ", ".join(sorted(loop.places)[:6]) + (" ..." if len(loop.places) > 6 else "")
    packed = ", ".join(f"{count} of {bits} bits" for bits, count in sorted(loop.packed.items()))
    return f"{loop.scalar} scalar{', packed ' + packed if packed else ''} ({places})"


def check_kernels(listing: list[Function], width: int, *, neon: bool) -> int:
    kernels = {}
    for function in listing:
        match = re.search(r"inspect_(\w+)", demangled(function.symbol))
        if match:
            kernels[match.group(1)] = function
    if not kernels:
        print("error: no inspect_* function in the listing", file=sys.stderr)
        return 1
    failures = 0
    for name, function in sorted(kernels.items()):
        found = loops(function, width, neon=neon)
        kinds = Counter(loop.kind for loop in found)
        narrow = 0 if neon else narrower(function, width, neon=neon)
        ok = kinds["vector"] > 0 and kinds["scalar"] == 0 and narrow == 0
        detail = ", ".join(f"{count} {kind}" for kind, count in sorted(kinds.items()))
        print(f"{'ok' if ok else 'FAIL'}  {name}: loops {detail or 'none'}; narrower arithmetic {narrow}")
        for loop in found:
            if loop.kind == "scalar":
                print(f"        scalar loop: {describe(loop)}")
        failures += not ok
    if failures:
        print(f"error: {failures} kernels are not vector code of {width} bits", file=sys.stderr)
        return 1
    return 0


def check_library(listing: list[Function], width: int, allowed: dict[str, dict[str, str]], *, neon: bool) -> int:
    functions_allowed, sums_allowed = allowed.get("functions", {}), allowed.get("sums", {})
    used: set[str] = set()
    totals: Counter[str] = Counter()
    failures = 0
    for function in listing:
        name = demangled(function.symbol)
        found = loops(function, width, neon=neon)
        totals.update(loop.kind for loop in found)
        scalar = [loop for loop in found if loop.kind == "scalar"]
        if not scalar:
            continue
        prefix = next((prefix for prefix in functions_allowed if name.startswith(prefix)), None)
        if prefix is not None:
            used.add(prefix)
            continue
        prefix = next((prefix for prefix in sums_allowed if name.startswith(prefix)), None)
        if prefix is not None:
            used.add(prefix)
            scalar = [loop for loop in scalar if not loop.adds_only]
            if not scalar:
                continue
        failures += 1
        print(f"FAIL  {name}")
        for loop in scalar:
            print(f"        scalar loop: {describe(loop)}")
    for prefix in [*functions_allowed, *sums_allowed]:
        if prefix not in used:
            print(f"note: nothing needed the allowance of {prefix}")
    print("loops: " + ", ".join(f"{count} {kind}" for kind, count in sorted(totals.items())))
    if failures:
        print(f"error: {failures} functions have loops that are not vectorized", file=sys.stderr)
        return 1
    return 0


# Small listings for the self-test: a function of a kernel whose loop is vector code
# of 256 bits and whose remainder is scalar, one with a scalar loop that adds and
# multiplies, one with a scalar loop that only adds, and one whose loop over rows has
# a vector loop within it and a copy for rows where it does not run, in the forms of
# each system.
X86_COFF = """
\t.cv_file\t1 "C:\\\\src\\\\kernels.rs"
_RNvCslXYZ_4test14inspect_kernel:
\t.cv_loc\t0 1 10 5
\txorl\t%eax, %eax
.LBB0_1:
\t.cv_loc\t0 1 11 9
\tvmovupd\t(%rcx,%rax,8), %ymm0
\tvfmadd213pd\t(%rdx,%rax,8), %ymm1, %ymm0
\tvmovupd\t%ymm0, (%r8,%rax,8)
\taddq\t$4, %rax
\tcmpq\t%r9, %rax
\tjb\t.LBB0_1
.LBB0_2:
\t.cv_loc\t0 1 11 9
\tvmovsd\t(%rcx,%rax,8), %xmm0
\tvfmadd213sd\t(%rdx,%rax,8), %xmm1, %xmm0
\t.cv_loc\t0 1 0 0
\tvaddsd\t%xmm2, %xmm0, %xmm0
\tvmovsd\t%xmm0, (%r8,%rax,8)
\tincq\t%rax
\tcmpq\t%r10, %rax
\tjb\t.LBB0_2
\tretq
_RNvCslXYZ_4test6scalar:
.LBB1_1:
\t.cv_loc\t0 1 20 9
\tvmulsd\t(%rcx,%rax,8), %xmm1, %xmm0
\tvaddsd\t%xmm0, %xmm2, %xmm2
\tincq\t%rax
\tcmpq\t%r10, %rax
\tjne\t.LBB1_1
\tretq
_RNvCslXYZ_4test4sums:
.LBB2_1:
\t.cv_loc\t0 1 30 9
\tvaddsd\t(%rcx,%rax,8), %xmm2, %xmm2
\tincq\t%rax
\tcmpq\t%r10, %rax
\tjne\t.LBB2_1
\tretq
_RNvCslXYZ_4test4rows:
\ttestq\t%r9, %r9
\tje\t.LBB3_4
.LBB3_1:
\t.cv_loc\t0 1 41 9
\tvsubsd\t(%rcx), %xmm0, %xmm1
\txorl\t%eax, %eax
.LBB3_2:
\t.cv_loc\t0 1 40 9
\tvmovupd\t(%rcx,%rax,8), %ymm1
\tvsubpd\t8(%rcx,%rax,8), %ymm1, %ymm1
\tvmovupd\t%ymm1, (%r8,%rax,8)
\taddq\t$4, %rax
\tcmpq\t%r9, %rax
\tjb\t.LBB3_2
\tdecq\t%rdx
\tjne\t.LBB3_1
\tretq
.LBB3_4:
\t.cv_loc\t0 1 41 9
\tvsubsd\t(%rcx), %xmm0, %xmm1
\tdecq\t%r10
\tjne\t.LBB3_4
\tretq
"""
X86_ELF = X86_COFF.replace('\t.cv_file\t1 "C:\\\\src\\\\kernels.rs"', '\t.file\t1 "/src" "kernels.rs"').replace(
    ".cv_loc\t0 1 ", ".loc\t1 "
)
ARM_GENERIC = """
\t.file\t1 "/src" "kernels.rs"
_RNvCslXYZ_4test14inspect_kernel:
\t.loc\t1 10 5
\tmov\tx8, xzr
.LBB0_1:
\t.loc\t1 11 9
\tldr\tq0, [x0, x8]
\tfmla\tv0.2d, v1.2d, v2.2d
\tstr\tq0, [x1, x8]
\tadd\tx8, x8, #16
\tcmp\tx8, x9
\tb.ne\t.LBB0_1
.LBB0_2:
\t.loc\t1 11 9
\tldr\td0, [x0, x8]
\tfmadd\td0, d1, d2, d0
\tstr\td0, [x1, x8]
\tsubs\tx10, x10, #1
\tb.ne\t.LBB0_2
\tret
"""
ARM_APPLE = (
    ARM_GENERIC.replace("_RNv", "__RNv")
    .replace(".LBB", "LBB")
    .replace("fmla\tv0.2d, v1.2d, v2.2d", "fmla.2d\tv0, v1, v2")
)
DEMANGLED = {
    "_RNvMs3_NtCs8orh4KnCB2e_12jpeg_unround5modelNtB5_7Problem9data_term": ("jpeg_unround::model::Problem::data_term"),
    "_RINvMs0_NtCs8orh4KnCB2e_12jpeg_unround5sweepNtB6_6Primal4bandKb0_EB8_": (
        "jpeg_unround::sweep::Primal::band::<false>"
    ),
    "_RNCNCNvMs3_NtCs8orh4KnCB2e_12jpeg_unround5modelNtB9_7Problem3news0_00Bb_": (
        "jpeg_unround::model::Problem::new::{closure#1}::{closure#0}"
    ),
    "__RNvCslXYZ_4test6kernel": "test::kernel",
    "_ZN12jpeg_unround5exact3sum17h0123456789abcdefE": "jpeg_unround::exact::sum",
}


def self_test() -> int:
    """Checks the reading of the small listings and the demangling."""
    failures = []

    def expect(what: str, found: object, wanted: object) -> None:
        if found != wanted:
            failures.append(f"{what}: {found!r}, not {wanted!r}")

    def quietly(listing: list[Function], allowed: dict[str, dict[str, str]]) -> int:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return check_library(listing, 256, allowed, neon=False)

    for symbol, name in DEMANGLED.items():
        expect(f"the name of {symbol}", demangled(symbol), name)
    for label, listing, width, neon in [
        ("x86-64, COFF", X86_COFF, 256, False),
        ("x86-64, ELF", X86_ELF, 256, False),
        ("AArch64", ARM_GENERIC, 128, True),
        ("AArch64, Apple's syntax", ARM_APPLE, 128, True),
    ]:
        read = functions(listing)
        found = {demangled(function.symbol): function for function in read}
        kernel = found.get("test::inspect_kernel")
        if kernel is None:
            failures.append(f"{label}: no kernel in {sorted(found)}")
            continue
        kinds = sorted(loop.kind for loop in loops(kernel, width, neon=neon))
        expect(f"{label}: the kernel's loops", kinds, ["remainder", "vector"])
        if neon:
            continue
        expect(f"{label}: at 512 bits", sorted(loop.kind for loop in loops(kernel, 512, neon=neon)), ["scalar"] * 2)
        expect(f"{label}: the scalar loop", [loop.kind for loop in loops(read[1], width, neon=neon)], ["scalar"])
        sums = {"sums": {"test::sums": "sums"}}
        expect(f"{label}: sums allowed", quietly(read[2:], sums), 0)
        expect(f"{label}: a product not allowed", quietly(read[1:2], sums), 1)
        expect(f"{label}: a function allowed", quietly(read[1:2], {"functions": {"test::scalar": "scalar"}}), 0)
        rows = sorted(loop.kind for loop in loops(read[3], width, neon=neon))
        expect(f"{label}: the loops over rows", rows, ["outer", "outer", "vector"])
    for failure in failures:
        print(f"FAIL  {failure}")
    print(f"self-test: {'ok' if not failures else f'{len(failures)} failed'}")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("kernels", "library", "demangle", "self-test"))
    parser.add_argument("listing", type=Path, nargs="?")
    parser.add_argument("--width", type=int, choices=(128, 256, 512))
    parser.add_argument("--neon", action="store_true", help="the listing is of AArch64")
    parser.add_argument("--allowed", type=Path, help="the allowed scalar loops (library)")
    arguments = parser.parse_args()
    if arguments.mode == "self-test":
        return self_test()
    if arguments.listing is None:
        parser.error("a listing is needed")
    text = arguments.listing.read_text(encoding="utf-8", errors="replace")
    if arguments.mode == "demangle":
        for line in text.split():
            print(demangled(line))
        return 0
    if arguments.width is None:
        parser.error("--width is needed")
    listing = functions(text)
    if arguments.mode == "kernels":
        return check_kernels(listing, arguments.width, neon=arguments.neon)
    if arguments.allowed is None:
        parser.error("library needs --allowed")
    with arguments.allowed.open("rb") as file:
        allowed = tomllib.load(file)
    return check_library(listing, arguments.width, allowed, neon=arguments.neon)


if __name__ == "__main__":
    sys.exit(main())
