const int numReadings = 10;          // Количество измерений для усреднения
int readings[numReadings];           // Массив для хранения измерений
int readIndex = 0;                   // Текущий индекс в массиве
int total = 0;                       // Сумма текущих измерений
int average = 0;                     // Усреднённое значение

void setup() {
  Serial.begin(115200);              // Запускаем последовательный порт
  for (int i = 0; i < numReadings; i++) {
    readings[i] = 0;                 // Инициализируем массив нулями
  }
}

void loop() {
  // Считываем сырое значение с пина A1
  int rawValue = analogRead(A1);

  // Вычитаем старое значение из суммы
  total = total - readings[readIndex];

  // Сохраняем новое значение в массив
  readings[readIndex] = rawValue;

  // Добавляем новое значение в сумму
  total = total + readings[readIndex];

  // Переход к следующему индексу
  readIndex = (readIndex + 1) % numReadings;

  // Расчёт среднего значения
  average = total / numReadings;

  // Поскольку все входящие значения от 0 до 1023, среднего шкалирования не требуется.

  // Прямое усреднённое значение
  int smoothedValue = average;

  // Отправляем усреднённое значение через Serial
  Serial.println(smoothedValue);

  delay(1);  
}
