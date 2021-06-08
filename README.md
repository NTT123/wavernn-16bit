# wavernn-16bit
The (unofficial) vanilla version of WaveRNN. Read the WaveRNN paper at [here](https://arxiv.org/abs/1802.08435).

**Note**:
- We use embed values for coarse and fine value instead of real values as described in the paper. The reason is that I noticed some click artifacts in synthesised sound when using real values.

To install:
```sh
git clone https://github.com/NTT123/wavernn-16bit.git
cd wavernn-16bit
pip3 install -e .
```

To train WaveRNN on dataset:
```sh
python3 -m wavernn.trainer --wav-dir=[path/to/wav/directory] --ckpt-dir=[path/to/model/checkpoints]
```

See other config
