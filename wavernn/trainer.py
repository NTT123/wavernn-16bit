import pickle
import time
from pathlib import Path
from typing import Deque

import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import soundfile as sf

from .config import FLAGS
from .dataloader import create_data_iter, load_data_on_memory
from .mel2wave import mel2wave
from .model import WaveRNN


def mol_llh(params, target, bits=16):
    mean_hat, scale_hat, weight_hat = jnp.split(params, 3, axis=-1)
    mean = jnp.tanh(mean_hat)
    scale = jax.nn.softplus(scale_hat)
    log_weight = jax.nn.log_softmax(weight_hat, axis=-1)

    x = target[..., None] - mean
    neg_x = -jnp.abs(x)
    upper = (neg_x + 1.0 / 2**bits) * scale
    lower = (neg_x - 1.0 / 2**bits) * scale
    log_cdf_upper = jax.nn.log_sigmoid(upper)
    log_cdf_lower = jax.nn.log_sigmoid(lower)
    log_pr = (log_cdf_upper + jnp.log1p(- jnp.exp(log_cdf_lower - log_cdf_upper)))
    llh = jax.nn.logsumexp(log_pr + log_weight, axis=-1)
    return llh


@hk.without_apply_rng
@hk.transform_with_state
def loss_fn(inputs):
    mel, signal = inputs
    x = signal.astype(jnp.float32) / 2**15
    pad = FLAGS.pad
    x = x[:, (pad-1):-pad]
    xinput = x[:, :-1]
    xtarget = x[:, 1:]
    mol_params = WaveRNN(num_mixtures=FLAGS.num_mixtures)(xinput, mel)
    llh = mol_llh(mol_params, xtarget)
    loss = -jnp.mean(llh)
    return loss, mol_params[0], xtarget[0]


def loss_(params, aux, inputs):
    (loss, mol_params, target), aux = loss_fn.apply(params, aux, inputs)
    return loss, (mol_params, target, aux)


value_and_grad_fn = jax.value_and_grad(loss_, has_aux=True)
optimizer = optax.chain(
    optax.clip_by_global_norm(1),
    optax.adam(
        optax.exponential_decay(FLAGS.learning_rate,
                                100_000, 0.5, False, 1e-6)
    )
)


@jax.jit
def update_fn(params, aux, optim_state, inputs):
    (loss, (logpr, target, new_aux)), grads = value_and_grad_fn(params, aux, inputs)
    updates, new_optim_state = optimizer.update(grads, optim_state, params)
    new_params = optax.apply_updates(params, updates)
    return (loss, logpr, target), (new_params, new_aux, new_optim_state)


def save_ckpt(ckpt_dir, step, params, aux, optim_state):
    dic = {'step': step, 'params': params, 'aux': aux, 'optim': optim_state}
    with open(ckpt_dir / f'ckpt_{step:08d}.pickle', 'wb') as f:
        pickle.dump(dic, f)


def load_ckpt(path):
    with open(path, 'rb') as f:
        dic = pickle.load(f)
    return dic


def load_latest_ckpt(ckpt_dir):
    files = sorted(ckpt_dir.glob('ckpt_*.pickle'))
    if len(files) > 0:
        return load_ckpt(files[-1])
    else:
        return None


def train(args):
    rng = jax.random.PRNGKey(42)
    dataset = load_data_on_memory(args.wav_dir)
    # keep the first 100 clips for evaluation
    if len(dataset) > 100:
        test_mel = dataset[0][0].T
        test_y = dataset[0][1]
        dataset = dataset[100:]
    else:
        test_mel = dataset[0][0].T[:800]
        test_y = dataset[0][1][:22050*10]
        dataset = dataset
    data_iter = create_data_iter(dataset, FLAGS.n_frames, FLAGS.batch_size)
    params, aux = loss_fn.init(rng, next(data_iter))
    optim_state = optimizer.init(params)

    dic = load_latest_ckpt(args.ckpt_dir)

    if dic is not None:
        params = dic['params']
        last_step = dic['step']
        aux = dic['aux']
        optim_state = dic['optim']
    else:
        last_step = -1

    losses = Deque(maxlen=1000)
    start = time.perf_counter()
    for step in range(last_step + 1, FLAGS.training_steps):
        batch = next(data_iter)
        (loss, mol_params, target), (params, aux, optim_state) = update_fn(
            params, aux, optim_state, batch)
        losses.append(loss)

        if step % 100 == 0:
            loss = sum(losses).item() / len(losses)
            end = time.perf_counter()
            delta = end-start
            start = end
            print(f'step {step} train loss {loss:.5f}  {delta:.3f}s')

        if step % 1000 == 0:
            save_ckpt(args.ckpt_dir, step, params, aux, optim_state)
            logpr = mol_llh(
                mol_params[None, :, :], jnp.linspace(-1.0, 1.0, 256)[:, None], bits=8)
            pr = jax.device_get(jnp.exp(logpr))
            plt.figure(figsize=(20, 5))
            plt.imshow(pr, aspect='auto', cmap='hot')
            plt.plot((target + 1.0) / 2. * 255., c='yellow', lw=2)
            plt.savefig(args.ckpt_dir / f'predicted_distribution_{step}.png')
            plt.close()

            last_step = step
            w = mel2wave(params, aux, rng, test_mel)
            w = jax.device_get(w.astype(jnp.float32))
            sf.write(str(
                args.ckpt_dir / f'test_clip_{step}.wav'), data=w, samplerate=FLAGS.sample_rate)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--wav-dir', required=True,
                        type=Path, help="Path to wav directory")
    parser.add_argument('--ckpt-dir', required=True,
                        type=Path, help="Path to checkpoint directory")
    args = parser.parse_args()
    train(args)
