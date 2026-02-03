import subprocess

# Замените BOARD_TYPE на тип вашей платы, например, "arduino:avr:uno"
BOARD_TYPE = "arduino:avr:uno"
PORT = "/dev/ttyACM0"  # Замените на порт, к которому подключена ваша плата
PROJECT_PATH = "/home/kipelkin/Desktop/Chip/ADC1256"
# Компиляция скетча
subprocess.run(["arduino-cli", "compile", "--fqbn", BOARD_TYPE, PROJECT_PATH])

# Загрузка скетча на плату
subprocess.run(["arduino-cli", "upload", "-p", PORT, "--fqbn", BOARD_TYPE, PROJECT_PATH])
