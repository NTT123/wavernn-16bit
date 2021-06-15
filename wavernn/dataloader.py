import random
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm.auto import tqdm

from .config import FLAGS


def wav2mel(y):
    mel = librosa.feature.melspectrogram(
        y,
        sr=FLAGS.sample_rate,
        n_fft=FLAGS.n_fft,
        hop_length=FLAGS.hop_length,
        win_length=FLAGS.win_length,
        window='hann',
        center=True,
        n_mels=FLAGS.n_mels,
        fmin=FLAGS.fmin,
        fmax=FLAGS.fmax,
        pad_mode='reflect',
        power=1
    )
    mel = np.log(mel + 1e-5)
    return mel


def load_data_on_memory(wav_dir: Path):
    # load all data files on memory
    dataset = []
    for fp in tqdm(sorted(wav_dir.glob('*.wav'))):
        y, sr = sf.read(fp, dtype='int16')
        assert sr == FLAGS.sample_rate, "wrong sample rate"
        mel = wav2mel(y.astype(np.float32) / 2**15)
        dataset.append((mel, y))
    return dataset


def create_data_iter(dataset, seq_len, batch_size):
    batch = []

    while True:
        random.shuffle(dataset)
        for mel, y in dataset:
            R = random.randint(seq_len, mel.shape[1]-1)
            L = R - seq_len
            m1 = mel[:, L: R]
            y1 = y[(L*FLAGS.hop_length):(R*FLAGS.hop_length)]
            batch.append((m1, y1))
            if len(batch) == batch_size:
                m2, y2 = zip(*batch)
                m2 = np.stack(m2, axis=0)
                m2 = np.swapaxes(m2, 1, 2)
                y2 = np.stack(y2, axis=0)

                yield m2, y2
                batch = []
