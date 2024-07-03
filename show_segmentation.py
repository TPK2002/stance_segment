import sys

import napari
import numpy as np

if __name__ == "__main__":

    try:
        segmentation_path = sys.argv[1]
        nx = int(sys.argv[2])
        ny = int(sys.argv[3])
        nz = int(sys.argv[4])
    except IndexError:
        print("Usage: show_dataset.py <path:segmentation> <nx> <ny> <nz>")
        sys.exit(1)

    try:
        segmentation = np.fromfile(segmentation_path, dtype=np.uint8)
        segmentation = segmentation.reshape(nz, nx, ny)
    except FileNotFoundError:
        print("File not found")
        sys.exit(1)
    except ValueError:
        print("File is not a numpy array")
        sys.exit(1)

    viewer = napari.Viewer()
    viewer.add_image(segmentation)
    napari.run()
