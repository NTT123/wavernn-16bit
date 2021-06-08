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


class WaveRNNOriginal(hk.Module):
    """The orginal wavernn model."""

    def __init__(self, hidden_dim: int = 1024, cond_dim=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.R = hk.Linear(3 * hidden_dim, with_bias=True,
                           w_init=hk.initializers.VarianceScaling())
        self.I_W = hk.get_parameter(
            'I_W', (cond_dim + 128*3, hidden_dim*3), init=hk.initializers.VarianceScaling())
        self.I_b = hk.get_parameter('I_b', (1, 3*hidden_dim), init=jnp.zeros)
        assert hidden_dim % 2 == 0, "Need an even hidden dim"
        d = hidden_dim // 2
        mask = jnp.ones_like(self.I_W)
        embed_dim = hidden_dim // 8
        mask = mask.at[-embed_dim:, 0*d:1*d].set(0.0)
        mask = mask.at[-embed_dim:, 2*d:3*d].set(0.0)
        mask = mask.at[-embed_dim:, 4*d:5*d].set(0.0)
        self.I_W_mask = mask
        self.O1 = hk.Linear(hidden_dim//2)
        self.O2 = hk.Linear(256)
        self.O3 = hk.Linear(hidden_dim//2)
        self.O4 = hk.Linear(256)
        self.c_embed = hk.Embed(256, embed_dim)
        self.f_embed = hk.Embed(256, embed_dim)

    def initial_state(self, batch_size: int):
        return jnp.zeros((batch_size, self.hidden_dim))

    def step(self, inputs, hx):
        # inputs: N x (cond_dim+3) (c[t-1], f[t-1], c[t])
        # hx: N x hidden_Dim

        ut_1, rt_1, et_1 = jnp.split(self.R(hx), 3, axis=-1)

        # Input -> hidden with masked weights
        x = jnp.dot(inputs, self.I_W * self.I_W_mask)
        b = jnp.broadcast_to(self.I_b, x.shape)
        x = x + b
        ut_2, rt_2, et_2 = jnp.split(x, 3, axis=-1)

        ut = jax.nn.sigmoid(ut_1 + ut_2)
        rt = jax.nn.sigmoid(rt_1 + rt_2)
        et = jnp.tanh(rt * et_1 + et_2)
        ht = ut * hx + (1. - ut) * et
        yc, yf = jnp.split(ht, 2, axis=-1)
        return (yc, yf), ht

    def inference(self, mels):
        N, L, D = mels.shape

        c0 = jnp.array([127]).astype(jnp.int32)
        f0 = jnp.array([0]).astype(jnp.int32)

        def loop(inputs, prev_state):
            mel, rng1, rng2 = inputs
            ct, ft, hx = prev_state
            ct = self.c_embed(ct)
            ft = self.f_embed(ft)
            x = jnp.concatenate((mel, ct, ft, ct), axis=-1)
            (yc, _), _ = self.step(x, hx)
            clogits = self.O2(jax.nn.relu(self.O1(yc)))
            new_ct = jax.random.categorical(rng1[0], clogits, axis=-1)
            new_ct_embed = self.c_embed(new_ct)
            x = jnp.concatenate((mel, ct, ft, new_ct_embed), axis=-1)
            (_, yf), new_hx = self.step(x, hx)
            flogits = self.O4(jax.nn.relu(self.O3(yf)))
            new_ft = jax.random.categorical(rng2[0], flogits, axis=-1)
            return (new_ct, new_ft), (new_ct, new_ft, new_hx)

        rng1s = jax.random.split(hk.next_rng_key(), L)[None]
        rng2s = jax.random.split(hk.next_rng_key(), L)[None]

        h0 = self.initial_state(N)
        (ct, ft), _ = hk.dynamic_unroll(
            loop, (mels, rng1s, rng2s), (c0, f0, h0), time_major=False)
        return ct*256 + ft - 2**15

    def __call__(self, x, mel):
        coarse, fine, coarse_t = jax.tree_map(
            jnp.squeeze, jnp.split(x, 3, axis=-1))
        cx = self.c_embed(coarse)
        fx = self.f_embed(fine)
        cx_ = self.c_embed(coarse_t)
        inputs = jnp.concatenate((mel, cx, fx, cx_), axis=-1)

        # inputs: N L (cond_dim + 3)
        N, L, D = inputs.shape
        hx = self.initial_state(N)
        (yc, yf), _ = hk.dynamic_unroll(self.step, inputs, hx, time_major=False)
        logits = jnp.stack(
            (
                self.O2(jax.nn.relu(self.O1(yc))),
                self.O4(jax.nn.relu(self.O3(yf)))
            ),
            axis=-1
        )
        return jax.nn.log_softmax(logits, axis=-2)


class WaveRNN(hk.Module):
    def __init__(self, is_training=True):
        super().__init__()
        self.rnn = WaveRNNOriginal(FLAGS.rnn_dim, FLAGS.rnn_dim//2)
        self.upsample = UpsampleNetwork(
            num_output_channels=FLAGS.rnn_dim//2, is_training=is_training)
        self.is_training = is_training

    def inference(self, mel):
        mel = self.upsample(mel)
        return self.rnn.inference(mel)

    def __call__(self, x, mel):
        mel = self.upsample(mel)
        log_pr = self.rnn(x, mel)
        return log_pr
