import spidev
import time
import numpy as np
import matplotlib.pyplot as plt

# Константы для команд ADS1256
CMD_RDATAC = 0x03
CMD_SDATAC = 0x0F
CMD_RESET = 0xFE

# Настройка SPI
spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 2000000
spi.mode = 0b01  # Режим SPI Mode 1

# Константы для перевода в вольты
VREF = 5  # Референсное напряжение АЦП
GAIN = 1
RESOLUTION = 2**24  # Разрешение АЦП (24 бита)

def start_rdatac_mode():
    """Запуск режима непрерывного чтения данных"""
    spi.xfer2([CMD_RDATAC])
    time.sleep(0.01)
    spi.xfer2([0x00, 0x00, 0x00])  # Отправляем 24 бита (нулевые байты)

def stop_rdatac_mode():
    """Остановка режима непрерывного чтения данных"""
    spi.xfer2([CMD_SDATAC])
    time.sleep(0.01)

def read_data():
    raw_data = spi.readbytes(3)
    value = (raw_data[0] << 16) | (raw_data[1] << 8) | raw_data[2]
    
    if value & 0x800000:  # Проверка старшего бита (знака)
        value -= 1 << 24
    
    return value



def convert_to_volts(adc_value):
    """Преобразование ADC значения в вольты"""
    voltage = GAIN * adc_value * VREF / (RESOLUTION) 
    return voltage

def moving_average(data, window_size):
    """Фильтр скользящего среднего"""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

try:
    # Сброс ADC
    spi.xfer2([CMD_RESET])
    time.sleep(0.05)

    # Запуск режима RDATAC
    start_rdatac_mode()
    print("RDATAC режим активирован. Считываем данные...")

    data_volts = []  # Массив для хранения значений в вольтах
    num_samples = 3000  # Количество значений для записи

    for _ in range(num_samples):
        time.sleep(0.001)  # Задержка между считываниями (1 мс)
        adc_value = read_data()
        voltage = convert_to_volts(adc_value)
        data_volts.append(voltage)
        print(f"ADC Value: {adc_value}, Voltage: {voltage:.6f} V")  # Для контроля

    # Применение фильтра скользящего среднего
    window_size = 20  # Размер окна фильтрации
    filtered_data_volts = moving_average(data_volts, window_size)

    # Построение графика значений в вольтах
    plt.figure(figsize=(10, 6))
    plt.plot(range(window_size // 2, len(filtered_data_volts) + window_size // 2), filtered_data_volts, label="Filtered Voltage (V)", linewidth=1)
    plt.title("ADC Data Readings in Volts (Filtered)")
    plt.xlabel("Sample Number")
    plt.ylabel("Voltage (V)")
    plt.legend()
    plt.grid()
    plt.show()


except KeyboardInterrupt:
    print("Программа остановлена пользователем.")

finally:
    stop_rdatac_mode()
    spi.close()
