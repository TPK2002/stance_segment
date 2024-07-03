import json
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import skimage.morphology as morph
import napari

def load_volume(stack_folder):
    """Load a volume from a path."""

    ## Get all files in folder
    files = {}
    for file in os.listdir(stack_folder):
        if file.endswith('.tif'):
            slice_number = int(file.split('.')[0].replace('slice', ''))
            files[slice_number] = file

    ## Sort files by slice number
    sorted_files = np.asarray(sorted(files.items()))[:, 1]

    ## Load stack
    stack = [imread(stack_folder + item) for item in sorted_files]
    return np.asarray(stack)

def load_raw_volume(dataset):
    """Load a volume from a path."""

    file = open(dataset + ".raw", "rb")
    array = np.fromfile(file, dtype=">H")

    json_file = open(dataset + ".json", "r")
    infos = json.loads(json_file.read())

    nx = infos['volume']['nx']
    ny = infos['volume']['ny']
    nz = infos['volume']['nz']

    array = array.reshape((nz, nx, ny))
    return array


def load_raw_volume_without_json(dataset, nx, ny, nz):
    """Load a volume from a path."""

    file = open(dataset, "rb")
    array = np.fromfile(file, dtype=">H")

    array = array.reshape((nz, nx, ny))
    return array

def showSlice(stack, slice_number):
    """Show a slice of a stack."""
    plt.imshow(stack[slice_number], cmap='gray')
    plt.show()

def showComparison(*slices, cmap='gray'):
    """Show a comparison of two slices."""

    fig, axes = plt.subplots(1, len(slices), figsize=(8, 4))
    ax = axes.ravel()
    for i, slice in enumerate(slices):
        ax[i].imshow(slice, cmap=cmap)
        ax[i].set_title("Slice {}".format(i))
    fig.tight_layout()
    plt.show()

def plot_ball(radius):
    ball = morph.ball(radius)

    # Prepare the figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Prepare the coordinates
    x, y, z = np.indices((2*radius+2, 2*radius+2, 2*radius+2)) - radius

    # Plot the voxels
    ax.voxels(x, y, z, ball, facecolors='blue', edgecolor='k')

    # Set the labels and title
    ax.set_title('3D Visualization of a Spherical Structuring (ball) Element')

    plt.show()

def run_napari(stack):
    napari.imshow(stack)
    napari.run()
