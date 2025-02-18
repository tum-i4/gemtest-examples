import re
import sys
from pathlib import Path
from subprocess import CalledProcessError, run  # nosec


def run_tests() -> None:
    try:
        run(
            [
                "poetry", "run", "pytest", "tests", "--capture=tee-sys",
                *sys.argv[1:]
            ],
            check=True
        )
    except CalledProcessError:
        print("Tests failed")
        sys.exit(1)


def run_example() -> None:
    if len(sys.argv) <= 1:
        print("No example specified", file=sys.stderr)
        sys.exit(1)
    example = sys.argv[1]
    example_path = Path("examples") / example
    if not example_path.exists():
        print(f"Example {example} does not exist", file=sys.stderr)
        sys.exit(1)
    try:
        run(
            [
                "poetry", "run", "pytest",
                str(example_path),
                "--capture=tee-sys",
                "--tb=no",
                "--html=report.html", "--self-contained-html",
                *sys.argv[2:]
            ],
            check=True
        )
    except CalledProcessError:
        print("Tests failed")
        sys.exit(1)


def run_example_fail() -> None:
    if len(sys.argv) <= 1:
        print("No example specified", file=sys.stderr)
        sys.exit(1)
    example = sys.argv[1]
    example_path = Path("examples") / example
    if not example_path.exists():
        print(f"Example {example} does not exist", file=sys.stderr)
        sys.exit(1)

    result = run(
        [
            "poetry", "run", "pytest",
            str(example_path),
            "--capture=tee-sys",
            "--tb=no",
            "--html=report.html", "--self-contained-html",
            *sys.argv[2:]
        ],
        capture_output=True,
        text=True
    )
    output = result.stderr + result.stdout
    print(output)
    regexp = re.compile(r'\s\d+\s(error)s?[\n\s]')
    if regexp.search(output):
        print('Error found. Pipeline failed.')
        sys.exit(1)

    else:
        print('Failures expected. Pipeline succeeded.')
        sys.exit(0)
