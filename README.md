# wavernn-16bit
The (unofficial) vanilla version of WaveRNN. Read the WaveRNN paper at [here](https://arxiv.org/abs/1802.08435).

**Note**:
- We use the upsample network from [Lyra paper](https://arxiv.org/abs/2102.09660).

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
python3 -m wavernn.mel2wave -m path/to/mel_file.npy -c path/to/training/checkpoint.pickle -o path/to/output.wav
```