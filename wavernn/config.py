
class FLAGS:
    batch_size = 128
    training_steps = 1_000_000
    # initial learning rate (5e-4), exponential decay to 1e-6
    learning_rate = 512e-6

    rnn_dim = 1024

    # data dsp
    sample_rate = 22050
    n_fft = 1024
    n_mels = 80
    hop_length = 256
    win_length = 1024
    fmin = 0
    fmax = 8000
    n_frames = 12  # per sequence
    pad = 1024
