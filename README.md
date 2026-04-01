# RUN
```
source chip/bin/activate
cd Desktop/Chip/
export PATH=$PATH:/home/kipelkin/Desktop/Chip/bin
python ReadPredict.py
```

## Environment setup

The block above is an example of environment setup and launch for the original environment.

## Project description

This project predicts the signal for the CA1 region of the rat hippocampus from the signal recorded in the CA3 region.

Two inference backends are supported:

- `lstm` uses the PyTorch model from `lstm_model.pth`
- `reservoir` uses the reservoir + readout weights from a `.pickle` file

The main entry point is `ReadPredict.py`. It can:

- read input data from the ADC script
- run inference with the selected model
- generate `Digits.txt`
- insert the generated DAC array into `ADC1256/ADC1256.ino`
- compile and upload the Arduino sketch

For the current Windows/Conda setup:

```powershell
conda activate chip
cd "D:\Codex Projects\Chip Reservoir"
python ReadPredict.py --model lstm
python ReadPredict.py --model reservoir --model-file "C:\Users\nikolay\Downloads\Telegram Desktop\231201_1_d.pickle"
```

Generated plots are saved to `plots/lstm` and `plots/reservoir`.

## Command line flags

### `ReadPredict.py`

- `--model {lstm,reservoir}`  
  Selects the inference backend.

- `--model-file PATH`  
  Path to the model file. For `lstm` it is a `.pth` file, for `reservoir` it is a `.pickle` file.

- `--input-signal PATH`  
  Path to the input signal file. Default: `VisibleAnswer.txt`.

- `--digits-output PATH`  
  Path to the generated `Digits.txt`-style file.

- `--plots-root PATH`  
  Root directory for generated plots. The script creates model-specific subfolders inside it, for example `plots/lstm` and `plots/reservoir`.

- `--skip-read`  
  Skips ADC reading and uses the existing input signal file.

- `--skip-arduino`  
  Skips `arduino-cli compile` and `arduino-cli upload`.

- `--board-type FQBN`  
  Arduino board type for `arduino-cli`. Default: `arduino:avr:uno`.

- `--port PORT`  
  Serial port used for uploading the sketch.

- `--project-path PATH`  
  Path to the Arduino project directory. Default: `ADC1256`.

### Model scripts

`NetworkAndFilters.py` and `NetworkAndReservoir.py` can also be launched directly with these flags:

- `--model-file PATH`
- `--input-signal PATH`
- `--digits-output PATH`
- `--plots-dir PATH`
