import cupy as cp
import numpy as np


num_slices_once = 150 # max of 200 slices fit into gpu memory


# Taken from scikit-image
# Adapted to run on gpu
def _denoise_tv_chambolle_nd(image, weight=0.1, eps=2.0e-4, max_num_iter=200):
    """Perform total-variation denoising on n-dimensional images.

    Parameters
    ----------
    image : ndarray
        n-D input data to be denoised.
    weight : float, optional
        Denoising weight. The greater `weight`, the more denoising (at
        the expense of fidelity to `input`).
    eps : float, optional
        Relative difference of the value of the cost function that determines
        the stop criterion. The algorithm stops when:

            (E_(n-1) - E_n) < eps * E_0

    max_num_iter : int, optional
        Maximal number of iterations used for the optimization.

    Returns
    -------
    out : ndarray
        Denoised array of floats.

    Notes
    -----
    Rudin, Osher and Fatemi algorithm.
    """

    ndim = image.ndim
    p = cp.zeros((image.ndim,) + image.shape, dtype=image.dtype)
    g = cp.zeros_like(p)
    d = cp.zeros_like(image)
    i = 0
    while i < max_num_iter:
        if i > 0:
            # d will be the (negative) divergence of p
            d = -p.sum(0)
            slices_d = [
                slice(None),
            ] * ndim
            slices_p = [
                slice(None),
            ] * (ndim + 1)
            for ax in range(ndim):
                slices_d[ax] = slice(1, None)
                slices_p[ax + 1] = slice(0, -1)
                slices_p[0] = ax
                d[tuple(slices_d)] += p[tuple(slices_p)]
                slices_d[ax] = slice(None)
                slices_p[ax + 1] = slice(None)
            out = image + d
        else:
            out = image
        E = (d**2).sum()

        # g stores the gradients of out along each axis
        # e.g. g[0] is the first order finite difference along axis 0
        slices_g = [
            slice(None),
        ] * (ndim + 1)
        for ax in range(ndim):
            slices_g[ax + 1] = slice(0, -1)
            slices_g[0] = ax
            g[tuple(slices_g)] = cp.diff(out, axis=ax)
            slices_g[ax + 1] = slice(None)

        norm = cp.sqrt((g**2).sum(axis=0))[cp.newaxis, ...]
        E += weight * norm.sum()
        tau = 1.0 / (2.0 * ndim)
        norm *= tau / weight
        norm += 1.0
        p -= tau * g
        p /= norm
        E /= float(image.size)
        if i == 0:
            E_init = E
            E_previous = E
        else:
            if cp.abs(E_previous - E) < eps * E_init:
                break
            else:
                E_previous = E
        i += 1
    return out

def denoise_chambolle_tv_gpu(stack):
    cp._default_memory_pool.free_all_blocks()
    denoised_stack = np.zeros_like(stack)
    start_slice = 0
    num_slices = stack.shape[0]
    while start_slice < num_slices:
        end_slice = min(num_slices, start_slice + num_slices_once)
        to_denoise = cp.asarray(stack[start_slice:end_slice]).astype("float32")
        print(f"Copied Array {start_slice}-{end_slice} to gpu")
        denoised_stack[start_slice:end_slice] = _denoise_tv_chambolle_nd(to_denoise, weight=1e4).get()
        print("Denoising finished, freeing memory")
        start_slice += num_slices_once
        cp._default_memory_pool.free_all_blocks()

    return denoised_stack