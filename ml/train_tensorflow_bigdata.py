"""Compatibility shim for the relocated TensorFlow big-data trainer."""

from pipelines.bigdata.tensorflow.train_tensorflow_bigdata import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
