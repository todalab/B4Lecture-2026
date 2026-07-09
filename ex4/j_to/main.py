"""H.M.M. evaluation entry point."""

import os

from evaluate import FIG_DIR, evaluate


def main():
    """Run H.M.M. evaluation across datasets."""
    datasets = [
        "data/data1.pickle",
        "data/data2.pickle",
        "data/data3.pickle",
        "data/data4.pickle",
    ]
    os.makedirs(FIG_DIR, exist_ok=True)
    for ds in datasets:
        evaluate(ds)


if __name__ == "__main__":
    main()
