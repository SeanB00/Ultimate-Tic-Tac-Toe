"""Compatibility wrapper for the filtered-dataset CNN training job."""

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import uttt.ml.cnn_core as cnn_core
import uttt.ml.train_cnn as train_cnn


def main():
    """Run the filtered-dataset training job."""
    cnn_core.TRAINING_NAME = "filtered"
    train_cnn.main()


if __name__ == "__main__":
    main()
