import numpy as np
import skimage.morphology as morph
import time
from scipy.ndimage import label

def close_volume(stack, dilationParameterEdges=4):
    start_time = time.time()
    #### Dilation und Erosion auf Volumen anwenden ####

    str_elem_dil = morph.cube(dilationParameterEdges // 2)
    str_elem_ero = morph.ball(dilationParameterEdges) # matlab: "sphere"
    # str_elem_ero = morph.cube(dilationParameterEdges)

    ## Automatic Approach using skimage, but same str_elem for erosion and dilation
    ## Automatic Approach is worse than manual approach and slower
    # autoClosing = morph.closing(stack, str_elem_ero)
    # print(f"Volumen mit ball closing={dilationParameterEdges} bearbeitet")

    ## Manual Approach using skimage
    morph.dilation(stack, str_elem_dil, out=stack)
    end_time = time.time()
    print(f"Volume Dilation with cube={dilationParameterEdges // 2} finished in {end_time-start_time}s")

    start_time = time.time()
    morph.erosion(stack, str_elem_dil, out=stack)
    end_time = time.time()
    print(f"Volumen mit erosion={dilationParameterEdges} in {end_time-start_time}s bearbeitet")

    return stack

def threshold_mean(stack, threshold_modifier=1):
    threshold = stack.mean()

    # Man könnte denken, stacks[stack < threshold] = 0 mache das gleiche
    # allerdings haben wir dann eine riesige Differenz zwischen den 0 und anderen Pixeln
    # subtrahieren wir erst, dann wird diese Differenz um threshold verringert
    # damit haben wir einen viel besseren Kontrast
    stack = stack - (threshold * threshold_modifier)
    stack[stack < 0] = 0


def remove_air(stack, threshold_modifier=1):
    start_time = time.time()

    # in Matlab wird ein statischer Wert genommen: 20.000
    # Mittelwert stand dahinter auskommentiert
    threshold = stack.mean()

    # Man könnte denken, stack[stack < threshold] = 0 mache das gleiche
    # allerdings haben wir dann eine riesige Differenz zwischen den 0 und anderen Pixeln
    # subtrahieren wir erst, dann wird diese Differenz um threshold verringert
    # damit haben wir einen viel besseren Kontrast
    stack = stack - (threshold * threshold_modifier)
    stack[stack < 0] = 0

    end_time = time.time()

    print(f"Volumen mit threshold={threshold} und modifier={threshold_modifier} in {end_time - start_time}s bearbeitet")
    return stack


def find_largest_component(binary_image):
    # Label the objects in the image
    labeled_array, num_features = label(binary_image)

    # Find the size of each connected component
    sizes = np.bincount(labeled_array.ravel())
    sizes[0] = 0  # Set size of background (label 0) to 0

    # Find the largest component
    max_label = sizes.argmax()

    # Create a mask for the largest component
    largest_component = (labeled_array == max_label)

    return largest_component
