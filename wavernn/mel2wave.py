
import haiku as hk
import jax

from wavernn.model import Vocoder


@hk.transform_with_state
def inference(mel):
    net = Vocoder()
    return net.inference(mel)


def mel2wave(params, aux, rng, mel):
    mel = mel[None]
    rng = jax.random.PRNGKey(42)
    wav = inference.apply(params, aux, rng, mel)[0]
    return jax.device_get(wav[0])


if __name__ == '__main__':
    import pickle
    from argparse import ArgumentParser
    from pathlib import Path

    import numpy as np
    import soundfile as sf

    parser = ArgumentParser()
    parser.add_argument('-m', '--mel-file', type=Path, required=True)
    parser.add_argument('-c', '--ckpt-file', type=Path, required=True)
    parser.add_argument('-o', '--output-wav-file', type=Path, required=True)
    parser.add_argument('-s', '--random-seed', type=int, default=42)
    args = parser.parse_args()

    with open(args.ckpt_file, 'rb') as f:
        dic = pickle.load(f)
    rng = jax.random.PRNGKey(args.random_seed)
    mel = np.load(args.mel_file)
    wav = mel2wave(dic['params'], dic['aux'], rng, mel).astype(np.int16)
    from .config import FLAGS
    sf.write(str(args.output_wav_file), data=wav, samplerate=FLAGS.sample_rate)
    print('Output file at', args.output_wav_file)
