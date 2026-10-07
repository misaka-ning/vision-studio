#!/usr/bin/env python3
"""Create small, real model files used only by the Qt model-viewer tests.

The minimal ONNX protobuf writer follows ONNX ModelProto/GraphProto schemas;
it avoids introducing an extra ONNX Python dependency into the runtime.
"""

import argparse
from pathlib import Path
import struct


def varint(value):
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def integer(field, value):
    return varint(field << 3) + varint(value)


def blob(field, value):
    if isinstance(value, str):
        value = value.encode("utf-8")
    return varint((field << 3) | 2) + varint(len(value)) + value


def value_info(name, dtype, dimensions):
    shape = b"".join(blob(1, integer(1, dimension) if isinstance(dimension, int)
                            else blob(2, dimension)) for dimension in dimensions)
    tensor_type = integer(1, dtype) + blob(2, shape)
    return blob(1, name) + blob(2, blob(1, tensor_type))


def tensor(name, dimensions):
    count = 1
    for dimension in dimensions:
        count *= dimension
    return (b"".join(integer(1, dimension) for dimension in dimensions)
            + integer(2, 1) + blob(8, name) + blob(9, struct.pack("<" + "f" * count, *([0.0] * count))))


def if_model():
    def branch(name, dimensions):
        node = blob(1, "W") + blob(2, "branch_output") + blob(3, "take_W") + blob(4, "Identity")
        return (blob(1, node) + blob(2, name) + blob(5, tensor("W", dimensions))
                + blob(12, value_info("branch_output", 1, dimensions)))

    then_graph = branch("then_scope", [2, 2])
    else_graph = branch("else_scope", [3, 2])
    then_attribute = blob(1, "then_branch") + blob(6, then_graph) + integer(20, 5)
    else_attribute = blob(1, "else_branch") + blob(6, else_graph) + integer(20, 5)
    node = (blob(1, "condition") + blob(2, "Y") + blob(3, "choose_branch") + blob(4, "If")
            + blob(5, then_attribute) + blob(5, else_attribute))
    graph = (blob(1, node) + blob(2, "scoped_parameters")
             + blob(11, value_info("condition", 9, []))
             + blob(12, value_info("Y", 1, ["rows", 2])))
    return integer(1, 8) + blob(2, "VisionStudio model-viewer test") + blob(7, graph) + blob(8, integer(2, 13))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    import torch
    torch.save({"backbone.block.conv.weight": torch.zeros(4, 3, 3, 3),
                "backbone.block.bn.bias": torch.ones(4),
                "head.fc.weight": torch.zeros(2, 4)}, args.output / "weights-only.pt")
    module = torch.nn.Linear(4, 3).eval()
    torch.jit.trace(module, torch.zeros(1, 4)).save(str(args.output / "linear-script.pt"))
    (args.output / "scoped-if.onnx").write_bytes(if_model())


if __name__ == "__main__":
    main()
