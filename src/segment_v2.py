import napari
import numpy as np
from skimage import filters, exposure
from scipy import ndimage as ndi
from skimage.segmentation import watershed

from src.gpu_tv_denoising import denoise_chambolle_tv_gpu
from src.performance import measure
from src import utils, vol_utils
from src.remove_ring_artefact import remove_ring_artefact

import matplotlib.pyplot as plt

# Concentrate on 2.75 microns
# Dataset: Laffert-von_marius_05.02/tom_2023-10-10-10-05-25_00

# 1. Load the volume
# (extension): remove ring artefact
# 2. gaussian filter with small sigma
# 3. remove air
# 4. cutout relevant part (maybe as second step or not at all, if air removal is good enough ?)
# 5. normalize
# 6. run clahe (contrast limited adaptive histogram equalization), to remove overly bright or dark areas
# 7. run otsu to segment the tissue


# Optimized for Laffert 2.75 microns scan
@measure
def segment_v2(stack, debug_mode=False):

    print("1.1 Removing Ring artefact")
    # 1.1 (extension) => works inplace
    (mask, air_mean, air_std, perc_25, background_value) = remove_ring_artefact(stack, debug_mode=debug_mode)

    # 3.
    # ! is not necessary when using the ring artefact remover...
    # ring artefact is either the plastic tube or the air/wax border, so there is no air remaining, except for bubbles
    # our objects don't really contain bubbles, so we ignore this for performance reasons.... => should be reactivated sometime
    # print("3. Removing air")
    # stack = vol_utils.remove_air(stack, threshold_modifier=0.8)

    if debug_mode:
        plt.imshow(stack[len(stack) // 2])
        plt.show()

    print("2. Removing Air")

    if debug_mode:
        print(stack.mean())
        print(stack.std())

    # 3.1 air removal with previosly discovered air mean and std
    threshold = air_mean + (air_std * 0.8)
    stack[stack < threshold] = threshold # evtl threshold modifier ?
    print(stack.max())
    print(stack.mean())
    stack[stack < 0] = 0

    if debug_mode:
        plt.imshow(stack[len(stack) // 2])
        plt.show()

    if debug_mode:
        print("Air threshold: " + str(threshold))
        print("Stack Mean: " + str(stack.mean()))

    print("3. Denoising with Chambolle")

    # stack = denoise_tv_chambolle(stack, weight=0.1)
    stack = denoise_chambolle_tv_gpu(stack)

    if debug_mode:
        plt.imshow(stack[len(stack) // 2])
        plt.show()

    # 6. run clahe
    print("6. Running CLAHE")
    stack = exposure.equalize_adapthist(stack)

    if debug_mode:
        plt.imshow(stack[len(stack) // 2])
        plt.show()

    # 7. run otsu, but only on the masked center
    print("7. Running otsu")
    compressed = np.asarray([np.ma.masked_array(slice, mask) for slice in stack])
    thresh = filters.threshold_otsu(compressed)
    stack = stack > thresh

    if debug_mode:
        plt.imshow(stack[len(stack) // 2])
        plt.show()

    # 8. find largest connected component => thus filter out small dirt particles
    print("8. Finding largest connected component")
    stack = vol_utils.find_largest_component(stack)

    if debug_mode:
        plt.imshow(stack[len(stack) // 2])
        plt.show()

    return stack

def watershed_2d(tissue_mask, markers=None):
    distance = ndi.distance_transform_edt(tissue_mask)
    return watershed(-distance, markers=markers, mask=tissue_mask)



if __name__ == "__main__":
    stack = utils.load_raw_volume("Datensätze/Laffert-von_marius_05.02/tom_2023-10-11-18-13-09_00")
    # since my computer is not fast enough, we need to cut out the relevant part
    # slices 750-1050 are very clean almost without artifacts or cracks
    stack = stack[750:1050, 400:-400, 400:-400]
    segmentation = segment_v2(stack)
    viewer = napari.Viewer()
    viewer.add_image(stack, name="Original")
    viewer.add_image(segmentation, name="Segmentation")
    napari.run()
