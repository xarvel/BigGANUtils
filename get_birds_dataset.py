import tensorflow as tf

def preprocess_image(img):
    return tf.cast(img, tf.float32) / 127.5 - 1.

def get_tfrecord_dataset(
        seed: int,
        buffer_size: int,
        batch_size: int,
        tfrecord_path: str,
        is_training: bool,
        image_size: int,
):
    def parse_example(proto):
        features = {
            "image": tf.io.FixedLenFeature([], tf.string),
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'channels': tf.io.FixedLenFeature([], tf.int64),
            'label_text': tf.io.FixedLenFeature([], tf.string),
            'label_onehot': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            'label_number': tf.io.FixedLenFeature([], tf.int64),
        }

        parsed = tf.io.parse_single_example(
            serialized=proto,
            features=features
        )

        image, label = parsed["image"], parsed["label_number"]
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (image_size, image_size))

        return image, label

    def augment(image, label):
        image = tf.image.random_flip_left_right(
            image, seed=seed
        )

        return image, label

    tfrecord_files = tf.io.gfile.glob(tfrecord_path)

    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(parse_example)

    if is_training:
        dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True, seed=seed)
        dataset = dataset.repeat()
        dataset = dataset.map(augment)

    dataset = dataset.map(lambda image, label: (preprocess_image(image), label))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


DATASET_SIZE = 13262
BUFFER_SIZE = DATASET_SIZE
TFRECORD_PATH = 'gs://brids-xarvel/*.tfrec'

def get_birds_dataset(batch_size: int, image_size: int, is_training: bool, seed: int):
    return get_tfrecord_dataset(
        batch_size=batch_size,
        image_size=image_size,
        buffer_size=BUFFER_SIZE,
        tfrecord_path=TFRECORD_PATH,
        is_training=is_training,
        seed=seed
    )
