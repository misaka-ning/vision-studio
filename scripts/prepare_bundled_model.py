#!/usr/bin/env python3
"""Prepare Ultralytics YOLOv5 v6.0 Nano for OpenCV's static ONNX importer.

Build-time helper only. The C++ application does not use Python.
Verified with onnx==1.17.0, onnxsim==0.4.36, onnxruntime==1.28.0
and numpy==2.4.6.
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxsim import simplify


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--size", type=int, default=640)
    args = parser.parse_args()
    if args.source.resolve() == args.destination.resolve():
        parser.error("Source and destination must differ to preserve the original model.")
    if args.size < 32 or args.size % 32:
        parser.error("YOLO input size must be a positive multiple of 32.")
    shape = [1, 3, args.size, args.size]
    source = onnx.load(str(args.source))
    input_name = source.graph.input[0].name
    simplified, equivalent = simplify(
        source, overwrite_input_shapes={input_name: shape}, check_n=3
    )
    if not equivalent:
        raise RuntimeError("ONNX simplification did not preserve model outputs.")
    onnx.checker.check_model(simplified)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(simplified, str(args.destination))

    # Verify every output against the unmodified official network with a
    # deterministic input, beyond the simplifier's three built-in checks.
    sample = np.random.default_rng(42).random(shape, dtype=np.float32)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    original_session = ort.InferenceSession(str(args.source), options, providers=["CPUExecutionProvider"])
    static_session = ort.InferenceSession(str(args.destination), options, providers=["CPUExecutionProvider"])
    expected = original_session.run(None, {input_name: sample})
    actual = static_session.run(None, {input_name: sample})
    if len(actual) != len(expected):
        raise RuntimeError("Output count changed during simplification.")
    maximum_error = 0.0
    for reference, output in zip(expected, actual):
        np.testing.assert_allclose(output, reference, rtol=1e-4, atol=2e-4)
        maximum_error = max(maximum_error, float(np.max(np.abs(output - reference))))
    print(f"Numerical comparison passed; maximum absolute error: {maximum_error:.8g}")
    print(f"Source SHA256: {sha256(args.source)}")
    print(f"Prepared SHA256: {sha256(args.destination)}")
    print(f"Input: {input_name} {shape}")
    for output in simplified.graph.output:
        print(f"Output: {output.name} {[dim.dim_value for dim in output.type.tensor_type.shape.dim]}")


if __name__ == "__main__":
    main()
