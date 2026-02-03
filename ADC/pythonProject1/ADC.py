import serial
import time
from tqdm import tqdm  # Импортирование tqdm для строки загрузки
import matplotlib.pyplot as plt  # Импорт для построения графиков

# Настройки подключения к Arduino
port = "/dev/ttyACM0"  # Укажите правильный COM-порт
baudrate = 115200  # Укажите ту же скорость, что и в Arduino (9600 изначально)

try:
    arduino = serial.Serial(port, baudrate, timeout=1)
    time.sleep(2)  # Задержка для установления соединения с ардуино после открытия порта

    # Параметры
    samples = 4000 # Количество значений для считывания
    reference_voltage = 5.0  # Опорное напряжение Arduino
    bit_resolution = 1023  # Разрешение АЦП (10 бит)

    # Открытие файлов для записи
    with open("output_raw_data.txt", "w") as raw_file, open("output_voltage_data.txt", "w") as voltage_file:
        print("Сбор данных...")

        # Создаём списки для хранения данных
        raw_values = []  # Сырой сигнал от Arduino
        voltages = []  # Преобразованные значения (в напряжении)

        # Добавление строки загрузки
        for _ in tqdm(range(samples), desc="Сбор данных", unit="шаг"):
            try:
                # Считываем строку данных из Serial
                data_raw = arduino.readline().decode('utf-8').strip()

                # Проверяем, является ли строка числом
                if data_raw.replace("-", "").isdigit():
                    raw_value = int(data_raw)  # Преобразование строки в целое число
                    raw_values.append(raw_value)  # Сохранение в список

                    # Преобразование в напряжение
                    voltage =  (raw_value / bit_resolution) * reference_voltage
                    voltages.append(voltage)  # Сохранение преобразованного значения

                    # Запись данных в файлы
                    raw_file.write(f"{raw_value}\n")
                    voltage_file.write(f"{voltage:.4f}\n")
            except KeyboardInterrupt:
                print("Прервано пользователем.")
                break
            except Exception as e:
                print(f"Ошибка: {e}")

    print("Сбор завершён. Сырые данные сохранены в 'output_raw_data.txt', данные напряжения — в 'output_voltage_data.txt'.")
    arduino.close()

    # Построение графиков
    print("Построение графиков...")

    # График сырых данных
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(raw_values, label="Сырые данные", color="blue")
    plt.xlabel("Номер измерения")
    plt.ylabel("Сырое значение (ед.)")
    plt.title("График сырых данных")
    plt.grid(True)
    plt.legend()

    # График преобразованных данных
    plt.subplot(2, 1, 2)
    plt.plot(voltages, label="Напряжение", color="green")
    plt.xlabel("Номер измерения")
    plt.ylabel("Напряжение (В)")
    plt.title("График напряжения")
    plt.grid(True)
    plt.legend()

    # Показать графики
    plt.tight_layout()
    plt.show()

except serial.SerialException:
    print("Ошибка: Устройство не подключено или не обнаружено.")
