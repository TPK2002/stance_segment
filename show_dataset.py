import sys

import napari
from src import utils

if __name__ == "__main__":

    try:
        dataset = sys.argv[1]
        nx = int(sys.argv[2])
        ny = int(sys.argv[3])
        nz = int(sys.argv[4])
    except IndexError:
        print("Usage: show_dataset.py <path:dataset> <int:nx> <int:ny> <int:nz>")
        sys.exit(1)
    except ValueError:
        print("Usage: show_dataset.py <path:dataset> <int:nx> <int:ny> <int:nz>")
        sys.exit(1)

    try:
        stack = utils.load_raw_volume_without_json(dataset, nx, ny, nz)
    except FileNotFoundError:
        print("File not found")
        sys.exit(1)

    viewer = napari.Viewer()
    viewer.add_image(stack)
    napari.run()
