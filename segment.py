import sys
import numpy as np

from src.segment_v2 import segment_v2
from src import utils


if __name__ == "__main__":
    try:
        dataset = sys.argv[1]
    except IndexError:
        print("Usage: segment.py <path:dataset> (optional: nx=<int>) (optional: ny=<int>) (optional: nz=<int>) (optional: out=<path:output>) (optional: slices_from=<int>) (optional: slices_to=<int>)")
        sys.exit(1)

    out = "segmentation.npy"
    slices_from = None
    slices_to = None
    nx, ny, nz = None, None, None
    for arg in sys.argv:
        if "out=" in arg:
            out = arg.split("=")[1]
        if "slices_from=" in arg:
            slices_from = int(arg.split("=")[1])
        if "slices_to=" in arg:
            slices_to = int(arg.split("=")[1])
        if "nx=" in arg:
            nx = int(arg.split("=")[1])
        if "ny=" in arg:
            ny = int(arg.split("=")[1])
        if "nz=" in arg:
            nz = int(arg.split("=")[1])

    if (nx or ny or nz) and not (nx and ny and nz):
        print("nx, ny, nz must be set together or left all empty")
        sys.exit(1)

    if dataset.endswith(".raw") and not nx:
        print("nx, ny, nz must be set for .raw files")
        sys.exit(1)

    if nx:
        try:
            stack = utils.load_raw_volume_without_json(dataset, nx, ny, nz)
        except FileNotFoundError:
            print("File not found")
            sys.exit(1)
    else:
        try:
            stack = utils.load_raw_volume(dataset)
        except FileNotFoundError:
            print("File not found")
            sys.exit(1)

    if slices_to or slices_from:
        slices_to = slices_to or len(stack)
        slices_from = slices_from or 0
        stack = stack[slices_from:slices_to]

    segmentation = segment_v2(stack)
    np.save(out, segmentation)
