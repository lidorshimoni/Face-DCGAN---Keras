import os, cv2, random
import numpy as np
import tensorflow as tf
import pandas as pd


class Dataset(object):
    sample_size = None
    batch_size = None
    crop = None
    filter = None
    dims = None
    shape = [None, None, None]
    image_size = None
    data_dir = None
    ignore_image_description = None
    y_dim = None  # number of facial features
    data_file = None
    data = None
    data_y = None

    def load_data(self):
        raise NotImplementedError

    def get_next_batch(self):
        raise NotImplementedError

    def save(self, dir):
        np.save(dir + 'data.npy', self.data)
        if not self.ignore_image_description:
            np.save(dir + 'data_y.npy', self.data_y)

    def load(self, dir):
        self.data = np.load(dir + 'data.npy')
        if not self.ignore_image_description:
            self.data_y = np.load(dir + 'data_y.npy')


class Anime(Dataset):
    def __init__(self, output_size=64, channel=3, sample_size=2e4, batch_size=64, crop=True, filter=True,
                 ignore_image_description=True,
                 data_dir='/home/lidor/Desktop/FDGAN/anime/data'):
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.filter = filter
        self.dims = output_size * output_size
        self.shape = [output_size, output_size, channel]
        self.image_size = output_size
        self.data_dir = data_dir
        self.ignore_image_description = ignore_image_description
        self.data = None
        self.y_dim = 1

    def load_data(self):

        images_dir = self.data_dir

        X = []
        count = 1
        print('\n===LOADING DATA===')
        err = 0
        while count < self.sample_size + 1:
            print('\rLoading: {}- {}/{}'.format(count + err, count, self.sample_size), end='\r')
            try:
                image = cv2.imread(os.path.join(images_dir, str(count + err) + ".png"))
                if image is None:
                    raise Exception
            except Exception as e:
                err += 1
                continue
            X.append(image)
            count += 1

        seed = 547
        X = np.array(X)
        np.random.seed(seed)
        np.random.shuffle(X)

        self.data = X / 255

    def get_next_batch(self, iter_num):
        ro_num = self.sample_size // self.batch_size - 1

        if iter_num % ro_num == 0:
            length = len(self.data)
            perm = np.arange(length)
            np.random.shuffle(perm)
            self.data = np.array(self.data)
            self.data = self.data[perm]

        return self.data[
               int(iter_num % ro_num) * self.batch_size: int(iter_num % ro_num + 1) * self.batch_size], np.zeros(
            shape=(64, 1))

    def save(self, dir):
        return super().save(dir)

    def load(self, dir):
        return super().load(dir)

    def text_to_vector(self, text):
        return np.zeros(shape=(64, 1))


class CelebA(Dataset):

    def __init__(self, op_size, channel, sample_size, batch_size, crop, filter, y_features=None,
                 data_dir='W:\Projects\General\FDGAN\kiryatgat-1502-fdgan-master\CelebA'):

        self.dataname = 'CelebA'
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.crop = crop
        self.filter = filter
        self.dims = op_size * op_size
        self.shape = [op_size, op_size, channel]
        self.image_size = op_size
        self.data_dir = data_dir
        self.y_dim = len(y_features)
        self.data_file = 'list_attr_celeba.csv'
        self.y_features = y_features

    def load_data(self):

        images_dir = os.path.join(self.data_dir, 'img_align_celeba/img_align_celeba')

        X = []
        y = []

        faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        data = pd.read_csv(os.path.join(self.data_dir, self.data_file))

        i = 0
        count = 0
        gender = {"Male": 0, "Female": 0}
        print('\n===LOADING DATA===')
        while count < self.sample_size:
            img = data['image_id'][i]
            print('\rLoading: {} - Loaded: {}'.format(img, count), end='')
            image = cv2.imread(os.path.join(images_dir, img))
            if self.crop:
                h, w, c = image.shape
                # crop 4/6ths of the image
                cr_h = h // 6
                cr_w = w // 6
                crop_image = image[cr_h:h - cr_h, cr_w:w - cr_w]
                image = crop_image
            image = cv2.resize(image, (self.image_size, self.image_size))
            face = faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                                flags=cv2.CASCADE_SCALE_IMAGE)
            if type(face) is np.ndarray:
                features = np.zeros(self.y_dim)
                if (data["Male"][i] == 1 and gender["Male"] <= self.y_features["Male"]) or (
                        data["Male"][i] == -1 and gender["Female"] <= self.y_features["Male"]):
                    for index, feat in enumerate(self.y_features.keys()):
                        features[index] = int(data[feat][i])

                    # take equal number of male and females
                    if data["Male"][i] == 1:
                        gender["Male"] += 1
                    else:
                        gender["Female"] += 1
                    X.append(image)
                    y.append(features)
                    count += 1
            i += 1

        print('\n\n===DATA STATS===')
        for index, feat in enumerate(self.y_features):
            print(feat + " : ", sum([1 for i in y if i[index] == 1]))

        X = np.array(X)
        y = np.array(y)

        seed = 547

        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)

        self.data = X / 255.

    def get_next_batch(self, iter_num):
        ro_num = self.sample_size // self.batch_size - 1

        if iter_num % ro_num == 0:
            length = len(self.data)
            perm = np.arange(length)
            np.random.shuffle(perm)
            self.data = np.array(self.data)
            self.data = self.data[perm]
            self.data_y = np.array(self.data_y)
            self.data_y = self.data_y[perm]

        return self.data[
               int(iter_num % ro_num) * self.batch_size: int(iter_num % ro_num + 1) * self.batch_size], self.data_y[int(
            iter_num % ro_num) * self.batch_size: int(iter_num % ro_num + 1) * self.batch_size]

    def text_to_vector(self, text):
        text = text.lower()
        key_words = [w.replace('_', ' ').lower() for w in self.y_features]
        vec = np.ones(self.y_dim) * -1
        for i, key in enumerate(key_words, 0):
            if key in text:
                vec[i] = 1
        # print(vec)
        batch_vector = np.tile(vec, (self.batch_size, 1))
        return batch_vector

    def save(self, dir):
        np.save(dir + '/data.npy', self.data)
        np.save(dir + '/data_y.npy', self.data_y)

    def load(self, dir):
        self.data = np.load(dir + '/data.npy')
        self.data_y = np.load(dir + '/data_y.npy')


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3, 4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter must have dimensions: HxW or HxWx3 or HxWx4')


def inverse_transform(images):
    return (images + 1.) / 2.


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return cv2.imwrite(path, image)


def avg(list):
    return sum(list) / len(list)


def interpolate_points(p1, p2, n_steps=10):
    # interpolate ratios between the points
    ratios = np.linspace(0, 1, num=n_steps)
    # linear interpolate vectors
    vectors = list()
    for ratio in ratios:
        v = (1.0 - ratio) * p1 + ratio * p2
        vectors.append(v)
    return np.asarray(vectors)
