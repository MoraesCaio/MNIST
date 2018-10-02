import numpy as np


def padding(np_ar, kernel, extension=False):
    """Performs either zero or extension padding over a numpy array

    Args:
        np_ar (numpy.ndarray): input numpy.ndarray
        kernel (numpy.ndarray): numpy.ndarray kernel (required for new size calculation)
        extension (bool, optional): if True, returns a extension padded numpy array \
                                    if False, returns a zero padded numpy array

    Returns:
        numpy.ndarray: numpy.ndarray
    """
    # adding margins
    hks = int((kernel.shape[0] - 1) / 2)  # half_kernel_size
    padded_ar = np.zeros([np_ar.shape[0] + 2 * hks, np_ar.shape[1] + 2 * hks, *np_ar.shape[2:]], dtype=np_ar.dtype)
    padded_ar[hks:-hks, hks:-hks] = np.copy(np_ar)

    if not extension:
        # zero padding
        return padded_ar

    # extension padding on edges
    padded_ar[:hks, :] = padded_ar[hks:hks + 1, :]
    padded_ar[:, :hks] = padded_ar[:, hks:hks + 1]
    padded_ar[-hks:, :] = padded_ar[-(hks + 1):-hks, :]
    padded_ar[:, -hks:] = padded_ar[:, -(hks + 1):-hks]

    return padded_ar


def convolution(np_ar, kernel, stride=1, bias=0.0):
    """Applies 2d or 3d convolution.

    Args:
        np_ar (numpy.ndarray): 2or3d numpy.ndarray image
        kernel (numpy.ndarray): 2or3d numpy.ndarray kernel
        stride (int): stride for convolution loops
        bias (float): bias for convolutions

    Returns:
        numpy.ndarray: 3d numpy.ndarray activation map
    """
    ein_op1 = 'yx' if len(np_ar.shape) == 2 else 'yxc'
    ein_op2 = 'yx' if len(kernel.shape) == 2 else 'yxc'
    ein_eq = ein_op1 + ',' + ein_op2 + '->'

    dim = kernel.shape[0]
    conv_shape0 = ((np_ar.shape[0] - dim) // stride) + 1
    conv_shape1 = ((np_ar.shape[1] - dim) // stride) + 1
    conv = np.zeros((conv_shape0, conv_shape1), dtype=np_ar.dtype)

    j = 0
    for y in range(dim - 1, np_ar.shape[0], stride):
        i = 0
        for x in range(dim - 1, np_ar.shape[1], stride):
            window = np_ar[(y - dim + 1): (y + 1), (x - dim + 1): (x + 1)]
            conv[j, i] = np.einsum(ein_eq, window, kernel) + bias
            i += 1
        j += 1

    return conv


def conv_colors(np_ar, kernel_2d, stride=1, bias=0):
    """Applies 2d convolution through each layer of np_ar 3rd dimension

    Args:
        np_ar (numpy.ndarray): 3d numpy.ndarray image
        kernel_2d (numpy.ndarray): 2d numpy.ndarray kernel
        stride (int): stride for convolution loops
        bias (int): bias for convolutions

    Returns:
        numpy.ndarray: 3d numpy.ndarray resulting image
    """

    channels = []
    for i in range(np_ar.shape[2]):
        channels.append(convolution(np_ar[:, :, i], kernel_2d, stride, bias))

    shape = (channels[0].shape[0], channels[0].shape[1], np_ar.shape[2])
    result = np.zeros(shape, dtype=np_ar.dtype)

    for ch, channel in enumerate(channels):
        result[:, :, ch] = channel

    return result


def format_img_ar(np_ar):
    """Clips array in 0 - 255 range.

    Args:
        np_ar (numpy.ndarray): input numpy.ndarray

    Returns:
        numpy.uint8: Clipped array
    """
    return np.uint8(np.clip(np_ar, 0, 255))


def relu(np_ar):
    """Activation function ReLU (rectified linear unit):
        0, if x < 0
        x, if x >= 0

    Args:
        np_ar (numpy.ndarray): input numpy.ndarray

    Returns:
        TYPE: Description
    """
    return np.maximum(0, np_ar)


def tanh(np_ar):
    """Hyperbolic tangent element-wise.

    Args:
        np_ar (numpy.ndarray): input numpy.ndarray

    Returns:
        numpy.ndarray: hyperbolic tangent array of input array
    """
    return (2 / (1 + np.exp(-2 * np_ar))) - 1


def normalize(np_ar, min_val, max_val):
    """Normalize values to 0 - 255 range

    Args:
        np_ar (numpy.ndarray): input numpy.ndarray
        min_val (int | float): mininum value for normalization
        max_val (int | float): maximum value for normalization

    Returns:
        numpy.ndarray: Normalized np_ar
    """
    # [0:1] normalization
    np_min = np.min(np_ar)
    np_range = np.max(np_ar) - np_min

    # avoiding division by zero due to unitary arrays
    if np_range == 0:
        return np_ar

    norm01 = (np_ar - np_min) / np_range

    # [min_val:max_val] scaling
    new_range = max_val - min_val
    return (norm01 * new_range) + min_val


def ar3d_to_file(fid, np_ar, dtype='int64', sep3=','):
    """
    Saves a 3d numpy array onto a text file with following the structure:
        value(sep3)value(sep3)(sep2)
        (sep1)
        value(sep3)value(sep3)(sep2)

    Args:
        fid (str): File name
        np_ar (numpy.ndarray): 3d numpy.ndarray to save
        dtype (numpy.dtype): numpy.dtype to save
        sep3 (str): 3rd dimension separator

    Raises:
        ValueError: If the numpy array given is not an 3d numpy.array
    """
    if type(np_ar) != np.ndarray:
        raise ValueError('np_ar must be a numpy.array.')
    if len(np_ar.shape) != 3:
        raise ValueError('np_ar must be a 3d numpy.array.')

    sep2, sep1 = '\n', '\n'
    np_ar = np_ar.astype(dtype=dtype)
    with open(fid, 'w') as f:
        for y in range(np_ar.shape[0]):
            for x in range(np_ar.shape[1]):
                for c in range(np_ar.shape[2]):
                    # each value must be written separatedly,
                    #  so the sep3 can be written between each one of them
                    f.write(str(np_ar[y, x, c]))
                    f.write(sep3)
                f.write(sep2)
            f.write(sep1)


def file_to_ar3d(fid, dtype='int64', sep3=','):
    """
    Reads a 3d numpy array from text file with following the structure:
        value(sep3)value(sep3)(sep2)
        (sep1)
        value(sep3)value(sep3)(sep2)

    Args:
        fid (str): File name
        dtype (numpy.dtype): output numpy.dtype
        sep3 (str): 3rd dimension separator

    Returns:
        numpy.ndarray: 3d numpy.ndarray parsed
    """
    with open(fid, 'r') as f:
        lines = f.readlines()
        sep2, sep1 = '\n', '\n'

        # calculating output array shape
        y_shape, x_shape, c_shape = 0, 0, 0
        found_y, found_x = False, False
        for l in lines:
            if sep3 not in l and l.endswith(sep1):
                found_y = True
                y_shape += 1
            elif not found_y:
                if not found_x:
                    c_shape = l.count(sep3)
                x_shape += 1
                found_x = True

        # parsing file
        np_ar = np.zeros((y_shape, x_shape, c_shape), dtype=dtype)
        y, x = 0, 0
        for l in lines:
            if sep3 in l and l.endswith(sep2):
                np_ar[y, x, :] = l.split(sep3)[:-1]
                x += 1
            elif l.endswith(sep1):
                x = 0
                y += 1

        return np_ar
