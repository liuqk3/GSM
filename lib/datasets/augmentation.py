import cv2
import numpy as np
import random

def trans():
    if random.randint(1, 100) < 30:
        return True

class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     ConvertColor(),
        >>>     RandomSaturation(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, im_list):
        """
        Args:
            im_list: list, each image in im_list has the shape [height, width]
        """
        for t in self.transforms:
            im_list = t(im_list)
        return im_list


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, im_list):
        """
        Args:
            im_list: list, each image in im_list has the shape [height, width, channel].
                Note that the images are in HSV format.
        """

        im_after = []
        for im in im_list:
            # im = im.copy()
            if self.current == 'BGR' and self.transform == 'HSV':
                im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            elif self.current == 'HSV' and self.transform == 'BGR':
                im = cv2.cvtColor(im, cv2.COLOR_HSV2BGR)
            else:
                raise NotImplementedError
            im_after.append(im)
        return im_after


class RandomSaturation(object):
    def __init__(self, lower=0.8, upper=1.2):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, im_list):
        """
        Args:
            im_list: list, each image in im_list has the shape [height, width, channel].
                Note that the images are in HSV format.
        """
        if trans(): # random.randint(0, 2):
            alpha = random.uniform(self.lower, self.upper)
            im_after = []
            for im in im_list:
                # im = im.copy()
                im[:, :, 1] = im[:, :, 1] * alpha
                im_after.append(im)
            return im_after
        else:
            return im_list


class RandomHue(object):
    def __init__(self, delta=9.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, im_list):
        """
        Args:
            im_list: list, each image in im_list has the shape [height, width]
            Note that the images are in HSV format.
        """
        if True:#random.randint(0, 2):
            delta = random.uniform(-self.delta, self.delta)
            im_after = []
            for im in im_list:
                # im = im.copy()
                im = im.astype(np.float)
                im[:, :, 0] = im[:, :, 0] + delta
                im[:, :, 0][im[:, :, 0] > 360.0] -= 360.0
                im[:, :, 0][im[:, :, 0] < 0.0] += 360.0
                im = im.astype(np.uint8)
                im_after.append(im)
            return im_after
        else:
            return im_list


class RandomBrightness(object):
    def __init__(self, delta=8):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, im_list):
        """
        Args:
            im_list: list, each image in im_list has the shape [height, width]
        """
        if trans():#random.randint(0, 2):
            im_after = []
            delta = random.uniform(-self.delta, self.delta)
            for im in im_list:
                # im = im.copy()
                im = im + delta
                im = im.astype(np.uint8)
                im_after.append(im)
        return im_list


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, im_list):
        """
        Args:
            im_list: list, each image in im_list has the shape [height, width]
        """
        if trans():#random.randint(0, 2):
            swap = self.perms[random.randint(0, len(self.perms)-1)]
            shuffle = SwapChannels(swap)  # shuffle channels
            im_after = []
            for im in im_list:
                # im = im.copy()
                im = shuffle(im)
                im_after.append(im)
            return im_after
        else:
            return im_list

class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # image = image.copy()
        image = image[:, :, self.swaps]
        return image

class RandomContrast(object):
    def __init__(self, lower=0.8, upper=1.2):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, im_list):
        """
        Args:
            im_list: list, each image in im_list has the shape [height, width]
        """
        if trans():#random.randint(0, 2):
            alpha = random.uniform(self.lower, self.upper)

            im_after = []
            for im in im_list:
                # im mcopy()
                im = im * alpha
                im = im.astype(np.uint8)
                im_after.append(im)
            return im_after
        else:
            return im_list

class ImageAugmentation(object):
    def __init__(self):
        self.augs = [
            RandomContrast(),
            ConvertColor(current='BGR', transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, im_list):
        """
        Args:
            im_list: list, each image in im_list has the shape [height, width]
        """
        im_list = self.rand_brightness(im_list)

        if random.randint(0, 2):
            distort = Compose(self.augs[:-1])
        else:
            distort = Compose(self.augs[1:])
        im_list = distort(im_list)

        im_list = self.rand_light_noise(im_list)

        return im_list


if __name__ == '__main__':

    import glob
    import os

    im_path = "/home/liuqk/Dataset/MOT/MOT17Det/train/MOT17-02/img1/"

    im_path = glob.glob(os.path.join(im_path, '*.jpg'))
    im_path.sort()

    wait_time = 1
    my_aug = ImageAugmentation()
    for im in im_path:
        im = cv2.imread(im)
        #im = im.astype(np.float)
        im  = [im]
        im = my_aug(im)[0]
        #im = im.astype(np.uint8)
        cv2.imshow('', im)
        key = cv2.waitKey(wait_time)


    pass






