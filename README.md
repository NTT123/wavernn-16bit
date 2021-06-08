# wavernn-16bit
The (unofficial) vanilla version of WaveRNN. Read the WaveRNN paper at [here](https://arxiv.org/abs/1802.08435).

**Note**:
- We use the upsample network from [Lyra paper](https://arxiv.org/abs/2102.09660).
- We use embed values for coarse and fine value instead of real values as described in the paper. The reason is that I noticed some click artifacts in synthesised sound when using real values.
- We use 10 coarse bits and 6 fine bits. My theory is that low coarse resolution causes click artifacts. Also, we use 3/4 of RNN cells for coarse bits prediction and 1/4 for fine bits prediction.

To install:
```sh
git clone https://github.com/NTT123/wavernn-16bit.git
cd wavernn-16bit
pip3 install -e .
```

To train WaveRNN on dataset:
```sh
python3 -m wavernn.trainer --wav-dir=path/to/wav/directory --ckpt-dir=path/to/checkpoint/directory
```

To synthesize speech from melspectrogram:
```sh
python3 -m wavernn.text2mel -m path/to/mel_file.npy -c path/to/training/checkpoint.pickle -o path/to/output.wav
```