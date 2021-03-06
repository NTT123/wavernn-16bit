import pickle
import time
from pathlib import Path
from typing import Deque

import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import soundfile as sf

from .config import FLAGS
from .dataloader import create_data_iter, load_data_on_memory
from .mel2wave import mel2wave
from .model import Vocoder


@hk.without_apply_rng
@hk.transform_with_state
def loss_fn(inputs):
    mel, signal = inputs
    signal = signal.astype(jnp.int32) + 2 ** 15
    high = jnp.bitwise_and(
        jnp.right_shift(signal, FLAGS.num_fine_bits),
        int("0b" + "1" * FLAGS.num_coarse_bits, 2),
    )
    low = jnp.bitwise_and(signal, int("0b" + "1" * FLAGS.num_fine_bits, 2))
    high1 = jnp.roll(high, -1, -1)
    x = jnp.stack((high, low, high1), axis=-1)
    pad = FLAGS.pad
    x = x[:, (pad - 1) : -pad]
    xinput = x[:, :-1]
    xtarget = x[:, 1:, :-1]
    cllh, fllh = Vocoder()(xinput, mel)
    coarse_dim = 2 ** FLAGS.num_coarse_bits
    clogprs = jax.nn.one_hot(xtarget[..., 0], num_classes=coarse_dim, axis=-1) * cllh
    fine_dim = 2 ** FLAGS.num_fine_bits
    flogprs = jax.nn.one_hot(xtarget[..., 1], num_classes=fine_dim, axis=-1) * fllh
    closs = -jnp.sum(clogprs, axis=-1)
    floss = -jnp.sum(flogprs, axis=-1)
    loss = jnp.mean(closs + floss)
    return loss, cllh[0], xtarget[0]


def loss_(params, aux, inputs):
    (loss, logprs_hat, target), aux = loss_fn.apply(params, aux, inputs)
    return loss, (logprs_hat, target, aux)


value_and_grad_fn = jax.value_and_grad(loss_, has_aux=True)
optimizer = optax.chain(
    optax.clip_by_global_norm(1),
    optax.adam(optax.exponential_decay(FLAGS.learning_rate, 100_000, 0.5, False, 1e-6)),
)


@jax.jit
def update_fn(params, aux, optim_state, inputs):
    (loss, (logpr, target, new_aux)), grads = value_and_grad_fn(params, aux, inputs)
    updates, new_optim_state = optimizer.update(grads, optim_state, params)
    new_params = optax.apply_updates(params, updates)
    return (loss, logpr, target), (new_params, new_aux, new_optim_state)


def save_ckpt(ckpt_dir, step, params, aux, optim_state):
    dic = {"step": step, "params": params, "aux": aux, "optim": optim_state}
    with open(ckpt_dir / f"ckpt_{step:08d}.pickle", "wb") as f:
        pickle.dump(dic, f)


def load_ckpt(path):
    with open(path, "rb") as f:
        dic = pickle.load(f)
    return dic


def load_latest_ckpt(ckpt_dir):
    files = sorted(ckpt_dir.glob("ckpt_*.pickle"))
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
        test_y = dataset[0][1][: FLAGS.sample_rate * 10]
        dataset = dataset
    data_iter = create_data_iter(dataset, FLAGS.n_frames, FLAGS.batch_size)
    params, aux = loss_fn.init(rng, next(data_iter))
    optim_state = optimizer.init(params)

    dic = load_latest_ckpt(args.ckpt_dir)

    if dic is not None:
        params = dic["params"]
        last_step = dic["step"]
        aux = dic["aux"]
        optim_state = dic["optim"]
    else:
        last_step = -1

    losses = Deque(maxlen=1000)
    start = time.perf_counter()
    for step in range(last_step + 1, FLAGS.training_steps):
        batch = next(data_iter)
        (loss, logpr, target), (params, aux, optim_state) = update_fn(
            params, aux, optim_state, batch
        )
        losses.append(loss)

        if step % 100 == 0:
            loss = sum(losses).item() / len(losses)
            end = time.perf_counter()
            delta = end - start
            start = end
            print(f"step {step} train loss {loss:.5f}  {delta:.3f}s")

        if step % 1000 == 0:
            save_ckpt(args.ckpt_dir, step, params, aux, optim_state)

            pr = jax.device_get(jnp.exp(logpr))
            plt.figure(figsize=(20, 5))
            plt.imshow(pr.T, aspect="auto", cmap="hot")
            plt.plot(target[..., 0], c="yellow", lw=1)
            plt.savefig(args.ckpt_dir / f"predicted_distribution_{step}.png")
            plt.close()

            last_step = step
            w = mel2wave(params, aux, rng, test_mel)
            w = jax.device_get(w.astype(jnp.int16))
            sf.write(
                str(args.ckpt_dir / f"test_clip_{step}.wav"),
                data=w,
                samplerate=FLAGS.sample_rate,
            )


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--wav-dir", required=True, type=Path, help="Path to wav directory"
    )
    parser.add_argument(
        "--ckpt-dir", required=True, type=Path, help="Path to checkpoint directory"
    )
    args = parser.parse_args()
    train(args)
