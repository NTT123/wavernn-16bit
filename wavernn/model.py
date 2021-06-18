import haiku as hk
import jax
import jax.numpy as jnp

from .config import FLAGS


class UpsampleNetwork(hk.Module):
    def __init__(self, num_output_channels, is_training=True):
        super().__init__()
        self.input_conv = hk.Conv1D(512, 3, padding='VALID', with_bias=False)
        self.input_bn = hk.BatchNorm(True, True, 0.99)
        self.dilated_conv_1 = hk.Conv1D(
            512, 2, 1, rate=2, padding='VALID', with_bias=False)
        self.dilated_bn_1 = hk.BatchNorm(True, True, 0.99)
        self.dilated_conv_2 = hk.Conv1D(
            512, 2, 1, rate=4, padding='VALID', with_bias=False)
        self.dilated_bn_2 = hk.BatchNorm(True, True, 0.99)

        self.upsample_conv_1 = hk.Conv1DTranspose(
            512, kernel_shape=1, stride=2, padding='SAME', with_bias=False)
        self.upsample_bn_1 = hk.BatchNorm(True, True, 0.99)
        self.upsample_conv_2 = hk.Conv1DTranspose(
            512, kernel_shape=1, stride=2, padding='SAME', with_bias=False)
        self.upsample_bn_2 = hk.BatchNorm(True, True, 0.99)
        self.upsample_conv_3 = hk.Conv1DTranspose(
            num_output_channels, kernel_shape=1, stride=4, padding='SAME', with_bias=False)
        self.upsample_bn_3 = hk.BatchNorm(True, True, 0.99)
        self.is_training = is_training

    def __call__(self, mel):
        x = jax.nn.relu(self.input_bn(self.input_conv(mel),
                                      is_training=self.is_training))
        res_1 = jax.nn.relu(self.dilated_bn_1(
            self.dilated_conv_1(x), is_training=self.is_training))
        x = x[:, 1:-1] + res_1
        res_2 = jax.nn.relu(self.dilated_bn_2(
            self.dilated_conv_2(x), is_training=self.is_training))
        x = x[:, 2:-2] + res_2

        x = jax.nn.relu(self.upsample_bn_1(
            self.upsample_conv_1(x), is_training=self.is_training))
        x = jax.nn.relu(self.upsample_bn_2(
            self.upsample_conv_2(x), is_training=self.is_training))
        x = jax.nn.relu(self.upsample_bn_3(
            self.upsample_conv_3(x), is_training=self.is_training))

        # tile x16
        N, L, D = x.shape
        x = jnp.tile(x[:, :, None, :], (1, 1, 16, 1))
        x = jnp.reshape(x, (N, -1, D))

        return x


class WaveRNN(hk.Module):
    def __init__(self, num_mixtures: int = 10, is_training=True):
        super().__init__()
        self.rnn = hk.LSTM(FLAGS.rnn_dim)
        self.mol_projection = hk.Sequential([
            hk.Linear(FLAGS.rnn_dim),
            jax.nn.relu,
            hk.Linear(num_mixtures * 3)
        ])
        self.upsample = UpsampleNetwork(
            num_output_channels=FLAGS.rnn_dim//2, is_training=is_training)
        self.is_training = is_training
        self.num_mixtures = num_mixtures

    def inference(self, mel):
        mel = self.upsample(mel)

        def loop(inputs, prev_state):
            cond, rng1, rng2 = inputs
            xp, hx = prev_state
            x = jnp.concatenate((cond, xp[..., None]), axis=-1)
            x, new_hx = self.rnn(x, hx)
            mol_params = self.mol_projection(x)
            mean_hat, scale_hat, weight_hat = jnp.split(mol_params, 3, axis=-1)
            mean = jnp.tanh(mean_hat)
            scale = jax.nn.softplus(scale_hat)
            idx = jax.random.categorical(rng1, weight_hat, axis=-1)
            mask = jax.nn.one_hot(idx, num_classes=self.num_mixtures)
            mean = jnp.sum(mean * mask, axis=-1)
            scale = jnp.sum(scale * mask, axis=-1)
            r = jax.random.logistic(rng2, mean.shape)
            x = (r / scale) + mean
            x = jnp.clip(x, a_min=-1., a_max=1.)
            return x, (x, new_hx)

        h0 = self.rnn.initial_state(1)
        x0 = jnp.array([0.0])
        h0 = (x0, h0)
        rng1s = jax.random.split(jax.random.PRNGKey(42), mel.shape[1]).T
        rng2s = jax.random.split(jax.random.PRNGKey(43), mel.shape[1]).T
        x, _ = hk.dynamic_unroll(
            loop, (mel, rng1s, rng2s), h0, time_major=False)
        return x

    def __call__(self, x, mel):
        mel = self.upsample(mel)
        B, L, D = mel.shape
        hx = self.rnn.initial_state(B)
        x = jnp.concatenate((mel, x[..., None]), axis=-1)
        x, _ = hk.dynamic_unroll(self.rnn, x, hx, time_major=False)
        mol_params = self.mol_projection(x)
        return mol_params
