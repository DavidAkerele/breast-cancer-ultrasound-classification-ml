"""Normalise the retained Terms of Reference to the registered submission title."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path


CURRENT_TITLE = (
    "Auditable Deep Learning for Breast Ultrasound Classification: "
    "Reproducible Model Comparison and a Research Dashboard"
)
REGISTERED_TITLE = "Breast Cancer Ultrasound Classification Using Machine Learning"


def update_document(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tor-update-") as temporary:
        working = Path(temporary) / destination.name
        with zipfile.ZipFile(source, "r") as source_archive:
            infos = source_archive.infolist()
            document_xml = source_archive.read("word/document.xml").decode("utf-8")

            current_count = document_xml.count(CURRENT_TITLE)
            registered_count = document_xml.count(REGISTERED_TITLE)
            if current_count not in {0, 2} or (current_count == 0 and registered_count != 2):
                raise ValueError(
                    "Terms of Reference title must occur exactly twice as either "
                    f"the current title ({current_count}) or registered title ({registered_count})"
                )
            document_xml = document_xml.replace(CURRENT_TITLE, REGISTERED_TITLE)
            document_xml = document_xml.replace("7Z10SS Masters Project", "7V0007 MSc Project")
            document_xml = document_xml.replace("6G7Z102", "6G7V0007")

            with zipfile.ZipFile(working, "w") as destination_archive:
                for info in infos:
                    data = (
                        document_xml.encode("utf-8")
                        if info.filename == "word/document.xml"
                        else source_archive.read(info.filename)
                    )
                    destination_archive.writestr(info, data)

        shutil.copy2(working, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.source.resolve() == args.destination.resolve():
        raise ValueError("The retained source must not be overwritten")
    update_document(args.source, args.destination)


if __name__ == "__main__":
    main()
