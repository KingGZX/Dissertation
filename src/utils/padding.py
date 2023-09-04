import lttb
from scipy import interpolate
import numpy as np
import math

"""
aim of this script is to pad the gait cycles,
since there existing huge differences between the frames of gait cycles of different people,
some patients with sever symptoms may need 800 frames [FPS = 60] to finish a gait cycle,
while we normal people only need about 100 frames even much less to finish a gait cycle.
"""


def downsample(data, n_out):
    """Downsample ``data`` to ``n_out`` points using the LTTB algorithm.

    Parameters
    ----------
    data : numpy.array

    Returns
    -------
    numpy.array
        Array of shape (n_out, ...)
    """

    if n_out > data.shape[0]:
        raise ValueError("n_out must be <= number of rows in data")

    if n_out == data.shape[0]:
        return data

    if n_out < 3:
        raise ValueError("Can only downsample to a minimum of 3 points")

    frames, features = data.shape

    # stack the frames vertically to the most left side of original data matrix
    time = np.arange(1, frames + 1).reshape(-1, 1)
    data = np.hstack((time, data))
    features += 1

    # Split data into bins
    n_bins = n_out - 2
    data_bins = np.array_split(data[1 : len(data) - 1], n_bins)

    # Prepare output array
    # First and last points are the same as in the input.
    out = np.zeros((n_out, features))
    out[0] = data[0]
    out[len(out) - 1] = data[len(data) - 1]

    # Largest Triangle Three Buckets (LTTB):
    # In each bin, find the point that makes the largest triangle
    # with the point saved in the previous bin
    # and the centroid of the points in the next bin.
    for i in range(len(data_bins)):
        this_bin = data_bins[i]

        if i < n_bins - 1:
            next_bin = data_bins[i + 1]
        else:
            next_bin = data[len(data) - 1 :]

        a = out[i]
        bs = this_bin
        c = next_bin.mean(axis=0)

        bs_len = len(bs)
        areas = np.zeros(bs_len)

        vec_1 = a - c
        for j in range(bs_len):
            vec_2 = a - bs[j]
            area = 0.5 * np.sqrt(np.dot(vec_1, vec_1) * np.dot(vec_2, vec_2) - np.square(np.dot(vec_1, vec_2)))
            areas[j] = area

        out[i + 1] = bs[np.argmax(areas)]

    out = out[:, 1:]
    return out


def padding(data, avg=120):
    """
    :param
        data is in shape [channel, frames, joints]
    """
    # reshape it back to 2D matrix
    assert len(data.shape) == 3, "data does not  have 3 dimensions"
    data = np.transpose(data, axes=[1, 0, 2])
    frames, channels, joints = data.shape

    data = np.reshape(data, newshape=(frames, -1))

    if frames < avg:
        ratio = avg / frames
        x = np.arange(1, channels * joints + 1)
        y = np.arange(1, frames + 1, dtype=np.float)
        y *= ratio
        z = data
        f = interpolate.interp2d(x, y, z, kind="linear")

        ts = np.arange(1, avg + 1)
        data = f(x, ts)

    elif frames > avg:
        # downsample
        # "downsample" api of package lttb only supports 2 columns data
        # the first column needs to be time, increasing
        # the second columns represents the observations
        # thus, it's not very helpful to our dataset
        # lttb.downsample()

        data = downsample(data, n_out=avg)

    data = np.reshape(data, newshape=(channels, avg, joints))

    return data

