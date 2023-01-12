"""Console script for ani_ext."""
import argparse
import sys


def main():
    """Console script for ani_ext."""
    parser = argparse.ArgumentParser()
    parser.add_argument('_', nargs='*')
    args = parser.parse_args()

    print("Arguments: " + str(args._))
    print("Replace this message by putting your code into "
          "ani_ext.cli.main")
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
