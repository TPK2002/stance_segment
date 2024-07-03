from skimage.filters import sobel

edge_removal_constant = 0.05


def remove_edge_effect(stack, background_value, mask):
    edges = sobel(stack, mask=mask)
    print("Edge Effect removal constant: " + str(edge_removal_constant))
    edges = edges > edge_removal_constant  # TODO: find out more, about this constant....
    stack[edges] = background_value
