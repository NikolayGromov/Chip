import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

NETWORK_SCRIPTS = {
    "lstm": REPO_ROOT / "NetworkAndFilters.py",
    "reservoir": REPO_ROOT / "NetworkAndReservoir.py",
}
DEFAULT_MODEL_FILES = {
    "lstm": REPO_ROOT / "lstm_model.pth",
    "reservoir": Path(r"C:/Users/nikolay/Downloads/Telegram Desktop/231201_1_d.pickle"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Read data, run prediction, and update the Arduino sketch.")
    parser.add_argument("--model", choices=sorted(NETWORK_SCRIPTS), default="lstm", help="Prediction backend.")
    parser.add_argument(
        "--model-file",
        help="Path to the model file. If omitted, a backend-specific default is used.",
    )
    parser.add_argument("--input-signal", default="VisibleAnswer.txt", help="Path to the input signal file.")
    parser.add_argument("--digits-output", default="Digits.txt", help="Path to the generated Digits.txt file.")
    parser.add_argument("--plots-root", default="plots", help="Root directory for generated plots.")
    parser.add_argument("--skip-read", action="store_true", help="Skip ADC data acquisition.")
    parser.add_argument("--skip-arduino", action="store_true", help="Skip arduino-cli compile and upload.")
    parser.add_argument("--board-type", default="arduino:avr:uno", help="Arduino board fqbn.")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Serial port for Arduino upload.")
    parser.add_argument(
        "--project-path",
        default=str(REPO_ROOT / "ADC1256"),
        help="Path to the Arduino project directory.",
    )
    return parser.parse_args()


def resolve_path(path_str):
    path = Path(path_str)
    return path if path.is_absolute() else REPO_ROOT / path


def default_model_file(model_name):
    model_path = DEFAULT_MODEL_FILES[model_name]
    if model_name == "reservoir" and not model_path.exists():
        return REPO_ROOT / "ridge.pickle"
    return model_path


def run_python(script_path, *script_args):
    subprocess.run([sys.executable, str(script_path), *script_args], check=True, cwd=REPO_ROOT)


def update_arduino_digits(digits_path):
    with open(digits_path, "r") as digits_file:
        digits_line = digits_file.readline().strip()

    adc_path = REPO_ROOT / "ADC1256" / "ADC1256.ino"
    with open(adc_path, "r+") as adc_file:
        adc_text = adc_file.read()

        pattern = re.compile(
            r"(const\s+PROGMEM\s+uint16_t\s+DACLookup_FullSine_12Bit\[4096\]\s*=\s*)(\{.*?\};)",
            re.DOTALL,
        )
        updated_text, replacements = pattern.subn(r"\1" + digits_line, adc_text, count=1)
        if replacements != 1:
            raise ValueError("Could not locate the DACLookup_FullSine_12Bit array in ADC1256.ino.")

        adc_file.seek(0)
        adc_file.write(updated_text)
        adc_file.truncate()


def main():
    args = parse_args()
    input_signal_path = resolve_path(args.input_signal)
    digits_output_path = resolve_path(args.digits_output)
    model_file_path = resolve_path(args.model_file) if args.model_file else default_model_file(args.model)
    plots_dir_path = resolve_path(args.plots_root) / args.model

    if not args.skip_read:
        run_python(REPO_ROOT / "ADC1256" / "ads1256_read.py")

    run_python(
        NETWORK_SCRIPTS[args.model],
        "--model-file",
        str(model_file_path),
        "--input-signal",
        str(input_signal_path),
        "--digits-output",
        str(digits_output_path),
        "--plots-dir",
        str(plots_dir_path),
    )

    update_arduino_digits(digits_output_path)

    if args.skip_arduino:
        return

    project_path = resolve_path(args.project_path)
    subprocess.run(
        ["arduino-cli", "compile", "--fqbn", args.board_type, str(project_path)],
        check=True,
        cwd=REPO_ROOT,
    )
    subprocess.run(
        ["arduino-cli", "upload", "-p", args.port, "--fqbn", args.board_type, str(project_path)],
        check=True,
        cwd=REPO_ROOT,
    )


if __name__ == "__main__":
    main()
