import matplotlib.pyplot as plt

# Открываем файл и считываем данные
with open('InputData/output_data.txt', 'r') as file:
    data = file.readlines()

# Преобразуем строки в числа с плавающей точкой
values = [float(line.strip()) for line in data]

# Создаем график с линиями
plt.plot(values)  # Удален параметр marker

# Добавляем заголовки и подписи
plt.title('График значений из файла')
plt.xlabel('Индекс')
plt.ylabel('Значение')

# Отображаем график
plt.show()
