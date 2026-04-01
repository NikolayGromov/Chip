import argparse
import os
import pickle
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
MPL_CONFIG_DIR = BASE_DIR / ".matplotlib"
MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib.pyplot as plt

HS = 7
TRAIN_WINDOW = 1000
HOP_LEN = TRAIN_WINDOW // 2

RESERVOIR_SIZE = 50
RESERVOIR_CONNECTIVITY = 0.5
RESERVOIR_SR = 1.0
RESERVOIR_LR = 0.01
RESERVOIR_INPUT_SCALING = 0.02
RESERVOIR_SEED = 42
STATE_START_SEED = 84
HEATING_DURATION = 8000


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Digits.txt with the reservoir model.")
    parser.add_argument("--model-file", default="ridge.pickle", help="Path to the pickled reservoir readout.")
    parser.add_argument("--input-signal", default="VisibleAnswer.txt", help="Path to the input signal file.")
    parser.add_argument("--digits-output", default="Digits.txt", help="Path to the generated Digits.txt file.")
    parser.add_argument("--plots-dir", default="plots/reservoir", help="Directory for generated plots.")
    return parser.parse_args()


ARGS = parse_args()


def resolve_path(path_str):
    path = Path(path_str)
    return path if path.is_absolute() else BASE_DIR / path


PLOTS_DIR = resolve_path(ARGS.plots_dir)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def normalization(signal):
    return np.array((signal - signal.mean()) / signal.std())


def remove_arts(signal):
    corrected = signal.copy()
    norm_signal = normalization(signal)
    for index in range(100, len(signal)):
        if norm_signal[index] > HS:
            corrected[index] = corrected[index - 100]
    return corrected


def gaus_filter(signal, sigma=35):
    min_original = np.min(signal)
    filtered = gaussian_filter(signal, sigma)
    min_filtered = np.min(filtered)
    return filtered * min_original / min_filtered


def get_weights_window(concat_len):
    right = stats.norm.pdf(np.arange(0, concat_len, 1), 0, concat_len / 5)
    left = stats.norm.pdf(np.arange(-concat_len, 0, 1), 0, concat_len / 5)
    weights = np.concatenate((np.expand_dims(right, axis=1), np.expand_dims(left, axis=1)), axis=1)
    return weights / np.expand_dims(weights.sum(axis=1), axis=1)


def smooth_signal(signal, win, hop):
    smoothed = []
    weights = get_weights_window(hop)
    smoothed.append(signal[:hop])
    for window_start in range(0, len(signal) - win, win):
        left_signal = signal[window_start + (win - hop):window_start + win] * weights[:, 0]
        right_signal = signal[window_start + win:window_start + win + hop] * weights[:, 1]
        smoothed.append(left_signal + right_signal)
    return np.array(smoothed).flatten()


def create_sequences(input_data, tw, hop):
    sequences = []
    signal_len = len(input_data)
    for index in range(0, signal_len - tw, hop):
        sequences.append(np.asarray(input_data[index:index + tw], dtype=np.float64))
    return sequences


def load_readout(model_path):
    with open(model_path, "rb") as model_file:
        data = pickle.load(model_file)
    if not isinstance(data, (list, tuple)) or len(data) < 2:
        raise ValueError("Reservoir model file must contain readout weights and bias.")
    w_readout = np.asarray(data[0], dtype=np.float64)
    bias = np.asarray(data[1], dtype=np.float64).reshape(1, -1)
    return w_readout, bias


class ReservoirPredictor:
    def __init__(self, model_path):
        try:
            from reservoirpy.nodes import Reservoir
        except ImportError as exc:
            raise ImportError(
                "reservoirpy is required for reservoir inference. Install it in the active environment."
            ) from exc

        self.reservoir = Reservoir(
            RESERVOIR_SIZE,
            sr=RESERVOIR_SR,
            lr=RESERVOIR_LR,
            rc_connectivity=RESERVOIR_CONNECTIVITY,
            input_scaling=RESERVOIR_INPUT_SCALING,
            seed=RESERVOIR_SEED,
        )
        self.init_state = np.random.default_rng(seed=STATE_START_SEED).random(RESERVOIR_SIZE) - 0.5
        self.reservoir.initialize(np.zeros((1,), dtype=np.float64))
        self.w_readout, self.bias = load_readout(model_path)


    def _warm_up(self):
        self.reservoir.reset()
        self.reservoir.state = {"out": self.init_state.copy()}
        self.reservoir.run(np.zeros((HEATING_DURATION, 1), dtype=np.float64))


    def predict(self, sequence):
        self._warm_up()
        states = np.asarray(self.reservoir.run(sequence.reshape(-1, 1)), dtype=np.float64)
        prediction = states @ self.w_readout + self.bias
        return prediction.reshape(-1)


def get_pred_signal(predictor, win, hop, sequences):
    preds = []
    for sequence in tqdm(sequences):
        preds.append(predictor.predict(sequence))
    preds = np.array(preds).flatten()
    return smooth_signal(preds, win, hop)


def convert(signal, old_low=0, old_high=3.3, new_low=0, new_high=4095):
    interval_0_1 = (signal - old_low) / (old_high - old_low)
    scaled = new_low + (new_high - new_low) * interval_0_1
    return scaled.astype(int)


def save_plot(filename):
    plt.savefig(PLOTS_DIR / filename)
    plt.close()


def main():
    ca3 = np.loadtxt(resolve_path(ARGS.input_signal))

    norm_signal = normalization(ca3)
    if (norm_signal[norm_signal > 0]).max() < np.abs(norm_signal[norm_signal < 0]).max():
        ca3 *= -1
        norm_signal *= -1

    indices = np.where(norm_signal > HS)[0]
    plt.figure(figsize=(16, 9))
    plt.plot(norm_signal, label="input")
    plt.plot([0, len(norm_signal)], [HS, HS], label="Hs")
    plt.legend()
    save_plot("Hs.png")

    last_max_index = indices[-1]
    print(indices)
    print(len(indices))
    print("argmax", last_max_index)

    plt.figure(figsize=(16, 9))
    plt.plot(ca3[last_max_index - 100:last_max_index + 100], label="input")
    plt.legend()
    save_plot("RawInputSignalPart.png")

    plt.figure(figsize=(16, 9))
    plt.plot(remove_arts(ca3[last_max_index - 1000:last_max_index + 2000]), label="input")
    plt.legend()
    save_plot("RemovedArtefact.png")

    plt.figure(figsize=(16, 9))
    plt.plot(normalization(remove_arts(ca3[last_max_index - 1000:last_max_index + 2000])), label="input")
    plt.legend()
    save_plot("RemovedArtefactAndNorm.png")

    plt.figure(figsize=(16, 9))
    plt.plot(
        gaus_filter(normalization(remove_arts(ca3[last_max_index - 1000:last_max_index + 2000])), sigma=5),
        label="input",
    )
    plt.legend()
    save_plot("RemovedArtefactAndNormAndFilter.png")

    signal_test_ca3 = gaus_filter(
        normalization(remove_arts(ca3[last_max_index - 1000:last_max_index + 2000]))
    ) / 10
    print(signal_test_ca3.shape)

    plt.figure(figsize=(16, 9))
    plt.plot(signal_test_ca3, label="input")
    plt.legend()
    save_plot("NetworkInput.png")

    test_sequences = create_sequences(signal_test_ca3, TRAIN_WINDOW, HOP_LEN)
    predictor = ReservoirPredictor(resolve_path(ARGS.model_file))
    sm_preds = get_pred_signal(predictor, TRAIN_WINDOW, HOP_LEN, test_sequences)
    print(len(test_sequences))
    print(sm_preds.shape)
    print("It computed")

    plt.figure(figsize=(16, 9))
    plt.plot(sm_preds, label="predicted")
    plt.legend()
    save_plot("Predicted.png")

    ca1 = np.concatenate([np.zeros(2016), sm_preds[500:1500][::16], np.zeros(2017)]) / 10
    print("ca1", np.unique(ca1))
    if ca1.min() < 0:
        ca1 = np.clip((-ca1), a_min=0, a_max=None)
    print("ca1", np.unique(ca1))

    ca1_converted = (convert(ca1) * 1).astype(int)
    print("len", (ca1_converted > 0).sum())
    print("uniq", np.unique(ca1_converted))
    print("Max", np.max(ca1_converted))

    digits_output_path = resolve_path(ARGS.digits_output)
    np.savetxt(digits_output_path, ca1_converted, fmt="%i", newline=", ")

    with open(digits_output_path, "r") as file_handle:
        text = file_handle.read()
    with open(digits_output_path, "w") as file_handle:
        file_handle.write("{" + text[:-2] + "};\n")


if __name__ == "__main__":
    main()
