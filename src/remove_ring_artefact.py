from skimage import feature, transform, color, draw, filters
import numpy as np
import matplotlib.pyplot as plt

"""
This function can be used to mask the actual tissue from the stance scanner
@param volume: volume to work on
@param orientation_slice_num: a slice that the mask is based on
"""
def remove_ring_artefact(volume, orientation_slice_num=None, debug_mode=False):

    # If no orientation slice is set, we use the center one
    if not orientation_slice_num:
        orientation_slice_num = len(volume) // 2

    orientation_slice = volume[orientation_slice_num]

    if debug_mode:
        plt.imshow(orientation_slice)
        plt.show()

    radius_range = (orientation_slice.shape[0] * 0.2, orientation_slice.shape[0] * 0.5)

    if debug_mode:
        print(f"RRA: Radius range: {radius_range}")

    ## First find the edges
    # Compute the gradient magnitude using Sobel filter
    gradient_magnitude = filters.sobel(orientation_slice)

    # Calculate the median of the gradient magnitudes
    median_val = np.median(gradient_magnitude)

    # Set high and low thresholds based on the median
    # Typically, a high threshold can be set as 1.33 times the median of the gradient magnitudes
    # Low threshold can be half of the high threshold
    T_high = 100000 * median_val
    T_low = 0.5 * T_high

    if debug_mode:
        print(f"RRA: T_low: {T_low} T_high: {T_high}")
        print(f"RRA: Gradient magnitude median: {median_val}")
        print(f"RRA: Gradient magnitude max: {orientation_slice.max()}")

    max = orientation_slice.max()
    sigmas = [10, 5, 2]
    for sigma in sigmas:
        edges = feature.canny(orientation_slice, sigma=sigma, low_threshold=T_low, high_threshold=T_high)

        if debug_mode:
            plt.imshow(edges)
            plt.show()

        # Then run the hough algorithm on them => use a range of radius = 400-800
        # step size 10 is fine enough and is a lot faster then e.g. 1
        hough_radii = np.arange(radius_range[0], radius_range[1], 10)
        hough_res = transform.hough_circle(edges, hough_radii)

        if len(hough_res) > 0:
            break

    if debug_mode:
        plt.imshow(hough_res[0])
        plt.show()

    # Find the most prominent circle
    accums, cxs, cys, radii = transform.hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)

    if debug_mode:
        print(f"RRA: Found circle at: {cxs[0]}, {cys[0]} with {radii[0]}")

    # TODO: Eval
    # Is -10 good ? => for now just based on some testing aroung
    radius = radii[0] - 75
    cx = cxs[0]
    cy = cys[0]

    # Then draw a disk from these parameters
    rr, cc = draw.disk((cx, cy), radius)
    mask = np.zeros_like(orientation_slice)
    mask[cc, rr] = 1

    if debug_mode:
        plt.imshow(mask)
        plt.show()

    mean = np.mean(orientation_slice, where=mask == 1)
    std = np.std(orientation_slice, where=mask == 1)

    hist, bin_edges = np.histogram(orientation_slice[mask], bins=1000)

    # Calculate cumulative histogram (cumulative counts)
    cum_hist = np.cumsum(hist)

    # Find the total number of elements in the histogram
    total_count = cum_hist[-1]

    # Find the 25th percentile bin
    percentile_25_value = None
    for i, count in enumerate(cum_hist):
        if count >= total_count * 0.25:
            percentile_25_value = bin_edges[i]
            break

    if debug_mode:
        print("The 25th percentile of the histogram is approximately:", percentile_25_value)

    mean_air = np.mean(orientation_slice, where=mask==0)
    std_air = np.std(orientation_slice, where=mask==0)

    if debug_mode:
        print(f"Air mean: {mean_air} std: {std_air}")
        print(f"Content mean: {mean} std: {std}")
    #print(f"ROI 25% percentile: {perc_25}")

    background_value = mean - std

    for slice in volume:
        slice[mask == 0] = background_value

    return (mask, mean_air, std_air, percentile_25_value, background_value)

