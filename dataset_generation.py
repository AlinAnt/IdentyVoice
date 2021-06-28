import os
from pathlib import Path

import numpy as np
import tensorflow as tf
# значение для перемешивания набора
SHUFFLE_SEED = 43

# количество образцов, используемых для проверки
VALID_SPLIT = 0.1

# используемая частота дискретизации.
SAMPLING_RATE = 16000

BATCH_SIZE = 128


def generate_sets(labels, DATASET_AUDIO_PATH):
    
    people_ids = labels

    audio_paths = []
    labels_new = []
    for label, name in enumerate(people_ids):
        #print(f"Processing speaker {name[0:5]}")
        dir_path = Path(DATASET_AUDIO_PATH) / name
        speaker_sample_paths = [os.path.join(dir_path, filepath)
                                for filepath in os.listdir(dir_path)
                                if filepath.endswith('.wav')]
        audio_paths += speaker_sample_paths
        labels_new += [label] * len(speaker_sample_paths)
    print(f'Found {len(audio_paths)} files belonging to {len(people_ids)} classes.')

    # Shuffle
    rng = np.random.RandomState(SHUFFLE_SEED)
    rng.shuffle(audio_paths)
    rng = np.random.RandomState(SHUFFLE_SEED)
    rng.shuffle(labels_new)


    # Split into training and validation
    val_num = int(VALID_SPLIT * len(audio_paths))
    print(f"Using {len(audio_paths) - val_num} files for training.")
    train_audio_paths = audio_paths[:-val_num]
    train_labels = labels_new[:-val_num]

    print(f'Using {val_num} files for validation.')
    valid_audio_paths = audio_paths[-val_num:]
    valid_labels = labels_new[-val_num:]

    # create 2 datasets
    train = to_dataset(train_audio_paths, train_labels)
    train = train.shuffle(buffer_size=BATCH_SIZE * 8, seed=SHUFFLE_SEED).batch(BATCH_SIZE)

    val = to_dataset(valid_audio_paths, valid_labels)
    val = val.shuffle(buffer_size=32 * 8, seed=SHUFFLE_SEED).batch(32)
    
    
    # Быстрое преобразование Фурье
    train = train.map(lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train = train.prefetch(tf.data.experimental.AUTOTUNE)

    val = val.map(
        lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    val = val.prefetch(tf.data.experimental.AUTOTUNE)
    #print(train)
    
    return train, val


def to_dataset(audio_path, labels):
    path_ds = tf.data.Dataset.from_tensor_slices(audio_path)
    audio_ds = path_ds.map(lambda x: to_audio(x))
    #print('audio_ds:', len(audio_ds))
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    #print('label_ds:', len(label_ds))
    return tf.data.Dataset.zip((audio_ds, label_ds))

def to_audio(path):
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1, 3 * SAMPLING_RATE)
    #print(len(audio))
    return audio


def audio_to_fft(audio):
    audio = tf.squeeze(audio, axis=-1)
    fft = tf.signal.fft(
        tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64)
    )
    fft = tf.expand_dims(fft, axis=-1)

    # Возвращает абсолютное значение первой половины FFT
    # которое представляет только положительные частоты
    return tf.math.abs(fft[:, : (audio.shape[1] // 2), :])


