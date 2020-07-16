import numpy as np
import cv2

TAG_CHAR = np.array([202021.25], np.float32)


def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))


def writeFlow(filename, uv, v=None):
    """ Write optical flow to file.

    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert (uv.ndim == 3)
        assert (uv.shape[2] == 2)
        u = uv[:, :, 0]
        v = uv[:, :, 1]
    else:
        u = uv

    assert (u.shape == v.shape)
    height, width = u.shape
    f = open(filename, 'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width * nBands))
    tmp[:, np.arange(width) * 2] = u
    tmp[:, np.arange(width) * 2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()

def resizeFlow(flow, size):
    """
    Resize the flow to a given size
    Args:
        flow: [h, w, 2]
        size: [h, w]

    Returns:
        resized flow

    """
    h, w = flow.shape[0], flow.shape[1]

    h_r, w_r = size[0], size[1]

    x = cv2.resize(flow[:, :, 0], (w_r, h_r)) * (w_r / w)
    y = cv2.resize(flow[:, :, 1], (w_r, h_r)) * (h_r / h)

    flow = np.concatenate((x[:, :, np.newaxis], y[:, :, np.newaxis]), axis=2)

    return flow


def flow2hsv(flow, tool_type='plt'):
    """
    This function convert the frame extracted from compressed video to rgb image.
     :param flow: 3D array, [H, W, 2]
    :param tool_type: string, 'plt' or 'cv2', the tools used to show
    :return: RGB image
    """

    s = np.shape(flow)
    new_s = (s[0], s[1], 3)
    hsv = np.zeros(new_s)  # make a hsv motion vector map

    hsv[:, :, 1] = 255

    mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
    hsv[:, :, 0] = ang * 180 / np.pi / 2  # direction
    hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # magnitude

    hsv = hsv.astype(np.uint8)  # change to uint8

    if tool_type == 'plt':
        hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)  # for plt
    elif tool_type == 'cv2':
        hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  # for cv2

    return hsv


def crop_flows(tlbrs, flow):
    """Crop flows based on the given boxes

    Args:
        tlbrs: [N, 4], [x1, y1, x2, y2]
        flow: [h, w, 2]
    """
    flow_list = []
    tlbrs = tlbrs.copy().astype(np.int)
    for box in tlbrs:
        if box[2] > box[0] and box[3] > box[1]:
            flow_tmp = flow[box[1]:box[3], box[0]:box[2], :]
        else:
            flow_tmp = None
        flow_list.append(flow_tmp)
    return flow_list
