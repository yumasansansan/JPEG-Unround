#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fetch the photographs that JPEG-Unround is benchmarked on, into data/photos/.

    python scripts/fetch_datasets.py list
    python scripts/fetch_datasets.py fetch kodak [bsds500] [--record]

The synthetic test images (scripts/test_images.py) are this project's own; these
are not. Each set comes under terms of its own, which `list` shows with where
they are written, and which have to be read before a set is used, or a picture
of it shown: nothing fetched may be committed (data/ is ignored) or passed on.

Two sets are fetched: Kodak's Photo CD sampler and the test pictures of BSDS500.
LIVE1 is given out on request only, so it is put in place by hand, as `list`
says. Classic5 is not used: no source of it states terms, and one of its five
pictures is refused by several journals.

Every file fetched is checked against the SHA-256 that scripts/datasets.sha256
records. None of the sources publishes digests of its own, so the recorded ones
are those of a first fetch, which `fetch --record` writes, to be committed.
"""

import argparse
import dataclasses
import hashlib
import os
import sys
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Final

ROOT: Final = Path(__file__).resolve().parent.parent
DIGESTS: Final = ROOT / "scripts" / "datasets.sha256"
USER_AGENT: Final = "jpeg-unround-fetch-datasets"
CHUNK: Final = 1 << 20


@dataclasses.dataclass(frozen=True)
class Dataset:
    name: str
    title: str
    source: str
    terms: str
    terms_url: str
    files: list[tuple[str, str]]  # (URL, name under data/photos/<name>/); empty when fetched by hand
    by_hand: str = ""


KODAK: Final = Dataset(
    name="kodak",
    title="Kodak Photo CD sampler (PCD0992): 24 photographs, 768x512 or 512x768, PNG",
    source="https://r0k.us/graphics/kodak/",
    terms=(
        "Kodak grants holders of the sampler disc the use of its pictures for any purpose of electronic imaging; "
        "a picture shown has to credit its photographer (PhotoCD_credits.txt names them) unless it has been "
        "altered. The host's page says it understands them to be free for any use; that is its reading, not a "
        "licence. They were converted from Photo CD with a loss the host describes."
    ),
    terms_url="https://r0k.us/graphics/kodak/PhotoCD_credits.txt",
    files=[
        (f"https://r0k.us/graphics/kodak/kodak/kodim{number:02d}.png", f"kodim{number:02d}.png")
        for number in range(1, 25)
    ],
)

BSDS500: Final = Dataset(
    name="bsds500",
    title="Berkeley Segmentation Data Set 500: its 200 test pictures, 481x321 or 321x481, JPEG",
    source="https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html",
    terms=(
        "For non-commercial research and teaching; a use cites Martin et al., ICCV 2001. The pictures are Corel's. "
        "They are JPEG files themselves, so their blocking is in the reference, and neither side is a multiple of 8."
    ),
    terms_url="https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html",
    files=[
        (
            "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz",
            "BSR_bsds500.tgz",
        )
    ],
)

LIVE1: Final = Dataset(
    name="live1",
    title="LIVE Image Quality Assessment Database, release 2: its 29 reference pictures, BMP",
    source="https://live.ece.utexas.edu/research/quality/subjective.htm",
    terms=(
        "The database may be used, copied and passed on with its notice kept, crediting LIVE and citing its three "
        "papers. Nine of the pictures, Visual Delights', may not be used outside the scope of the database without "
        "their permission; the others are Kodak's Photo CD pictures."
    ),
    terms_url="https://live.ece.utexas.edu/research/quality/copyright2.htm",
    files=[],
    by_hand=(
        "Ask for the database through the form of its page (the password comes by e-mail), and copy the 29 "
        "pictures of its refimgs folder, as they are, into data/photos/live1/."
    ),
)

CLASSIC5: Final = Dataset(
    name="classic5",
    title="Classic5: baboon, barbara, boats, lena, peppers, 512x512, grey",
    source="(copies in the repositories of several papers)",
    terms=(
        "No source of these pictures states terms, and the copyright of many classic test pictures is unknown "
        "(USC-SIPI says as much). Lena is refused by several journals, and by IEEE since 2024."
    ),
    terms_url="https://sipi.usc.edu/database/copyright.php",
    files=[],
    by_hand="Not used by this project.",
)

DATASETS: Final = {dataset.name: dataset for dataset in (KODAK, BSDS500, LIVE1, CLASSIC5)}


def read_digests() -> dict[str, str]:
    digests: dict[str, str] = {}
    if DIGESTS.exists():
        for line in DIGESTS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                digest, name = line.split(maxsplit=1)
                digests[name.strip()] = digest
    return digests


def write_digests(digests: dict[str, str]) -> None:
    lines = [f"{digest}  {name}" for name, digest in sorted(digests.items())]
    DIGESTS.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path) -> None:
    """Writes what url answers into target, through a partial file that only a whole answer replaces it with."""
    if not url.startswith("https://"):
        message = f"not fetched over anything but HTTPS: {url}"
        raise ValueError(message)
    partial = target.with_name(target.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310 -- HTTPS, checked above
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as stream:  # noqa: S310 -- HTTPS
        expected = response.headers.get("Content-Length")
        written = 0
        while chunk := response.read(CHUNK):
            stream.write(chunk)
            written += len(chunk)
    if expected is not None and written != int(expected):
        partial.unlink()
        message = f"{url} answered {written} bytes of {expected}"
        raise OSError(message)
    partial.replace(target)


def extract_bsds500(archive: Path, directory: Path) -> int:
    """The test pictures of the archive, into directory/test/, by their own names: nothing else is written."""
    test = directory / "test"
    test.mkdir(exist_ok=True)
    count = 0
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            path = PurePosixPath(member.name)
            if (
                member.isfile()
                and path.parent == PurePosixPath("BSR/BSDS500/data/images/test")
                and path.suffix == ".jpg"
            ):
                source = tar.extractfile(member)
                if source is None:
                    continue
                with source, (test / path.name).open("wb") as stream:
                    stream.write(source.read())
                count += 1
    return count


def fetch(dataset: Dataset, digests: dict[str, str], *, record: bool) -> int:
    if not dataset.files:
        print(f"{dataset.name}: fetched by hand. {dataset.by_hand}")
        return 1
    directory = ROOT / "data" / "photos" / dataset.name
    directory.mkdir(parents=True, exist_ok=True)
    failures = 0
    for url, name in dataset.files:
        key = f"{dataset.name}/{name}"
        target = directory / name
        if key not in digests and not record:
            print(
                f"error: scripts/datasets.sha256 records no digest for {key}. If this is the first fetch, "
                "fetch with --record, and commit the digests it writes."
            )
            return 1
        if not (target.exists() and sha256_of(target) == digests.get(key)):
            print(f"fetching {url}")
            download(url, target)
        digest = sha256_of(target)
        if key not in digests:
            digests[key] = digest
            print(f"recorded {key}")
        elif digest != digests[key]:
            print(f"error: {key} has the SHA-256 {digest}, not the {digests[key]} recorded")
            failures += 1
    if dataset is BSDS500 and not failures:
        count = extract_bsds500(directory / "BSR_bsds500.tgz", directory)
        print(f"{count} test pictures in {(directory / 'test').as_posix()}")
        failures += count != 200
    if not failures:
        print(f"{dataset.name}: {len(dataset.files)} files in {directory.as_posix()}")
    return failures


def show(dataset: Dataset) -> None:
    print(f"{dataset.name}: {dataset.title}")
    print(f"  source: {dataset.source}")
    print(f"  terms:  {dataset.terms}")
    print(f"          ({dataset.terms_url})")
    if dataset.by_hand:
        print(f"  by hand: {dataset.by_hand}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="the sets, where they come from, and their terms")
    get = commands.add_parser("fetch", help="fetch sets into data/photos/")
    get.add_argument("names", nargs="+", choices=[name for name, dataset in DATASETS.items() if dataset.files])
    get.add_argument("--record", action="store_true", help="record the digests of files that have none")
    options = parser.parse_args()

    if options.command == "list":
        for dataset in DATASETS.values():
            show(dataset)
        return 0
    digests = read_digests()
    failures = 0
    for name in options.names:
        show(DATASETS[name])
        failures += fetch(DATASETS[name], digests, record=options.record)
    if options.record:
        write_digests(digests)
        print(f"the digests are in {os.path.relpath(DIGESTS, ROOT)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
