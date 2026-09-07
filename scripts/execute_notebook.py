"""Execute the dissertation notebook and fail immediately on a cell error."""
import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "notebook",
        nargs="?",
        type=Path,
        default=Path("Breast_Cancer_Ultrasound_Classification_Dissertation.ipynb"),
    )
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    notebook = nbformat.read(args.notebook, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=args.timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(args.notebook.resolve().parent)}},
    )
    client.execute()
    nbformat.validate(notebook)
    nbformat.write(notebook, args.notebook)
    print(f"Executed and validated {args.notebook}")


if __name__ == "__main__":
    main()
