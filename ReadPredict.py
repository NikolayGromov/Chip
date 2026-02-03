import subprocess

subprocess.run(["python", "/home/kipelkin/Desktop/Chip/ADC1256/ads1256_read.py"])
subprocess.run(["python", "NetworkAndFilters.py"])

# Открываем файл Digits.txt для чтения
with open("Digits.txt", "r") as digits_file:
    # Читаем строку из файла
    digits_line = digits_file.readline().strip()

# Открываем файл ADC1256.ino для чтения и записи
with open("ADC1256/ADC1256.ino", "r+") as adc_file:
    # Читаем все строки из файла
    lines = adc_file.readlines()
    
    # Вставляем строку из Digits.txt на 15-ю позицию в списке строк
    # (14 индекс, так как счет начинается с 0)
    lines[14] = digits_line    
    # Перемещаем курсор в начало файла
    adc_file.seek(0)
    
    # Записываем измененный список строк обратно в файл
    adc_file.writelines(lines)
    
# Замените BOARD_TYPE на тип вашей платы, например, "arduino:avr:uno"
BOARD_TYPE = "arduino:avr:uno"
PORT = "/dev/ttyACM0"  # Замените на порт, к которому подключена ваша плата
PROJECT_PATH = "/home/kipelkin/Desktop/Chip/ADC1256"
# Компиляция скетча
subprocess.run(["arduino-cli", "compile", "--fqbn", BOARD_TYPE, PROJECT_PATH])

# Загрузка скетча на плату
subprocess.run(["arduino-cli", "upload", "-p", PORT, "--fqbn", BOARD_TYPE, PROJECT_PATH])
