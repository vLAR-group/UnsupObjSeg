import os.path

from modules.utils import flatten_all_but_last, ensure_3d
from shapeguard import ShapeGuard
import sonnet as snt
import tensorflow.compat.v1 as tf


BYTE_FEATURES = ['mask', 'image']
MAX_NUM_ENTITIES = 7

class IODINEDataset(snt.AbstractModule):
    num_true_objects = 1
    num_channels = 3

    factors = {}

    def __init__(
        self,
        path,
        batch_size,
        image_dim,
        crop_region=None,
        shuffle_buffer=1000,
        max_num_objects=None,
        min_num_objects=None,
        grayscale=False,
        name="dataset",
        **kwargs,
    ):
        super().__init__(name=name)
        self.path = os.path.abspath(os.path.expanduser(path))
        self.batch_size = batch_size
        self.crop_region = crop_region
        self.image_dim = image_dim
        self.shuffle_buffer = shuffle_buffer
        self.max_num_objects = max_num_objects
        self.min_num_objects = min_num_objects
        self.grayscale = grayscale
        self.dataset = None

    def _build(self, subset="train"):
        dataset = self.dataset

        if subset == "train":
            # normal mode returns a shuffled dataset iterator
            if self.shuffle_buffer is not None:
                dataset = dataset.shuffle(self.shuffle_buffer)
        elif subset == "summary":
            # for generating summaries and overview images
            # returns a single fixed batch
            dataset = dataset.take(self.batch_size)

        # repeat and batch
        dataset = dataset.repeat().batch(self.batch_size, drop_remainder=True)
        # dataset = dataset.apply(tf.data.experimental.ignore_errors())

        iterator = dataset.make_one_shot_iterator()
        data = iterator.get_next()

        # preprocess the data to ensure correct format, scale images etc.
        data = self.preprocess(data)
        return data

    def preprocess(self, data):
        sg = ShapeGuard(dims={
            "B": self.batch_size,
            "H": self.image_dim[0],
            "W": self.image_dim[1]
        })
        image = sg.guard(data["image"], "B, h, w, C")
        mask = sg.guard(data["mask"], "B, L, h, w, 1")

        # to float
        image = tf.cast(image, tf.float32) / 255.0
        # mask = tf.cast(mask, tf.float32) / 255.0

        # crop
        if self.crop_region is not None:
            height_slice = slice(self.crop_region[0][0], self.crop_region[0][1])
            width_slice = slice(self.crop_region[1][0], self.crop_region[1][1])
            image = image[:, height_slice, width_slice, :]

            mask = mask[:, :, height_slice, width_slice, :]

        flat_mask, unflatten = flatten_all_but_last(mask, n_dims=3)

        # rescale
        size = tf.constant(
            self.image_dim, dtype=tf.int32, shape=[2], verify_shape=True)
        image = tf.image.resize_images(
            image, size, method=tf.image.ResizeMethod.BILINEAR)
        mask = tf.image.resize_images(
            flat_mask, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        if self.grayscale:
            image = tf.reduce_mean(image, axis=-1, keepdims=True)

        output = {
            "image": sg.guard(image[:, None], "B, T, H, W, C"),
            "mask": sg.guard(unflatten(mask)[:, None], "B, T, L, H, W, 1"),
            "factors": self.preprocess_factors(data, sg),
        }

        if "visibility" in data:
            output["visibility"] = sg.guard(data["visibility"], "B, L")
        else:
            output["visibility"] = tf.ones(sg["B, L"], dtype=tf.float32)

        return output

    def preprocess_factors(self, data, sg):
        return {
            name: sg.guard(ensure_3d(data[name]), "B, L, *")
            for name in self.factors
        }

    def get_placeholders(self, batch_size=None):
        batch_size = batch_size or self.batch_size
        sg = ShapeGuard(
            dims={
                "B": batch_size,
                "H": self.image_dim[0],
                "W": self.image_dim[1],
                "L": self.num_true_objects,
                "C": 3,
                "T": 1,
                "chn": 2,
            })
        return {
            "image": tf.placeholder(dtype=tf.float32, shape=sg["B, T, H, W, C"]),
            "mask": tf.placeholder(dtype=tf.float32, shape=sg["B, T, L, H, W, 1"]),
            "visibility": tf.placeholder(dtype=tf.float32, shape=sg["B, L"]),
            "factors": {
                name:
                tf.placeholder(dtype=dtype, shape=sg["B, L, {}".format(size)])
                for name, (dtype, size) in self.factors
            },
        }

class MultiObjectDataset(IODINEDataset):

    def __init__(
        self,
        path,
        image_dim=(128, 128),
        **kwargs,
    ):
        super().__init__(path=path, image_dim=image_dim, **kwargs)
        self.dataset = dataset(self.path)


def _parse_record(example_photo):
    features = {
        'image': tf.FixedLenFeature((), tf.string),
        'mask': tf.FixedLenFeature((), tf.string)
    }
    parsed_features = tf.parse_single_example(example_photo,features=features)

    parsed_features['image'] = tf.decode_raw(parsed_features['image'], tf.uint8)
    parsed_features['mask'] = tf.decode_raw(parsed_features['mask'], tf.uint8)

    parsed_features['image'] = tf.reshape(parsed_features['image'], (128, 128, 3))
    parsed_features['mask'] = tf.reshape(parsed_features['mask'], (128, 128, MAX_NUM_ENTITIES, 1))

    parsed_features['mask'] = tf.transpose(parsed_features['mask'], [2,0,1,3])

 
    return parsed_features

def dataset(tfrecords_path, read_buffer_size=None,
            map_parallel_calls=None):
    dataset = tf.data.TFRecordDataset(tfrecords_path, compression_type=None)
    dataset = dataset.map(_parse_record)
    return dataset
