
"""Generate HTML API documentation for the MMA3001 Python package.

The documentation is generated using pdoc from the shared
mma3001 modules and their Google-style function docstrings.
"""

import argparse
import os
import subprocess
import sys

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"
PACKAGE_ROOT = SOURCE_ROOT / "mma3001"

DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "api"


def parse_args() -> argparse.Namespace:
    """Parse arguments for HTML documentation generation.

    Returns:
        Namespace containing the documentation output directory.
    """
    parser = argparse.ArgumentParser(
        description="Generate MMA3001 HTML API documentation."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for generated HTML API pages.",
    )

    return parser.parse_args()


def build_documentation(output_dir: Path) -> None:
    """Generate navigable HTML documentation using pdoc.

    The Python package is loaded from src/mma3001 and all
    its submodules are included in the generated API reference.

    Args:
        output_dir: Destination directory for generated HTML pages.

    Returns:
        None. Writes HTML documentation to the output directory.

    Raises:
        FileNotFoundError: If the project package is missing.
        subprocess.CalledProcessError: If pdoc generation fails.
    """
    if not PACKAGE_ROOT.is_dir():
        raise FileNotFoundError(
            f"MMA3001 package not found: {PACKAGE_ROOT}"
        )

    environment = os.environ.copy()

    previous_pythonpath = environment.get("PYTHONPATH", "")

    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in [
            str(SOURCE_ROOT),
            previous_pythonpath,
        ]
        if value
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        sys.executable,
        "-m",
        "pdoc",
        "--output-directory",
        str(output_dir),
        "mma3001",
    ]

    print("MMA3001 HTML DOCUMENTATION")
    print("Source package:", PACKAGE_ROOT)
    print("Output directory:", output_dir)

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )

    print("\nDocumentation generated successfully.")
    print("Open:", output_dir / "mma3001.html")


def main() -> None:
    """Build HTML documentation using command-line configuration.

    Returns:
        None. Generates documentation under docs/api by default.

    Raises:
        FileNotFoundError: If the source package is missing.
        subprocess.CalledProcessError: If pdoc generation fails.
    """
    args = parse_args()

    build_documentation(
        args.output_dir.expanduser().resolve()
    )


if __name__ == "__main__":
    main()
