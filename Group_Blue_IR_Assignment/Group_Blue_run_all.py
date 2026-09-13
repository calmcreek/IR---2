
import os
import subprocess
import sys


# Project directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "Group_Blue_src")


def run_program(filename):
    """Run a Python program from Group_Blue_src."""
    filepath = os.path.join(SRC_DIR, filename)

    print("\n" + "=" * 70)
    print(f"Running: {filename}")
    print("=" * 70)

    result = subprocess.run(
        [sys.executable, filepath],
        cwd=SRC_DIR
    )

    if result.returncode != 0:
        print(f"\nERROR: {filename} failed.")
        sys.exit(result.returncode)

    print(f"\nCompleted: {filename}")


def main():
    print("=" * 70)
    print("GROUP BLUE - INFORMATION RETRIEVAL ASSIGNMENT II")
    print("=" * 70)

    # Step 1: Build the index
    run_program("Group_Blue_index.py")

    # Step 2: Run retrieval models and experiments
    run_program("Group_Blue_experiments.py")

    print("\n" + "=" * 70)
    print("ALL PROGRAMS COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("\nCheck the following directory for generated results:")
    print(os.path.join(BASE_DIR, "Group_Blue_results"))


if __name__ == "__main__":
    main()

