# Реализация машинного перевода

## Цель проекта
 Создание модели, которая сможет осуществлять перевод текстов с одного языка на другой (в текущей реализации - с немецкого на английский)

## Виртуальное окружение
Python 3.11.11

Все необходимые версии библиотек лежат в requirements.txt, для установки выполните следующую команду:
```bush
pip install requirements.txt. 
```

## Используемые технологии
- <b>torch</b> - в качестве основного фреймворка для обучения нейронных сетей;
- <b>transformers</b> - для токенизации текстов;
- <b>evaluate</b> - для подсчета метрик.

## Структура проекта
- <a href=https://github.com/sonador88/text_translation/blob/main/main.ipynb>main.ipynb</a> - основной файл со всей логикой программы;
- <a href=https://github.com/sonador88/text_translation/blob/main/transformer.py>transformer.py</a> - здесь хранится реализация трансформера;
- <a href=https://github.com/sonador88/text_translation/blob/main/warmup_cosine.py>warmup_cosine.py</a> - файл с классом, осуществляющим технологию warmup;
- <a href=https://github.com/sonador88/text_translation/blob/main/ema.py>ema.py</a> - файл с классом, осуществляющим технологию EMA;
- <a href=https://github.com/sonador88/text_translation/tree/main/data>data</a> -  папка с входными данными и выходными.

## Алгоритм решения задачи
- Подгружаем данные для обучения;
- Разбиваем данные по батчам;
- Токенизируем тексты;
- В процессе обучения трансформера обновляем шаг оптимизации по warmup, обновляем параметры модели техникой EMA;
- После каждой эпохи делаем прогноз для тестовой выборки и считаем accuracy и bleu;
- Прогноз для тестовой выборки и запись результатов в файл <a href=https://github.com/sonador88/text_translation/blob/main/data/multi30k_solution.tsv>data/multi30k_solution.tsv</a>.

## Полученный результат и дальнейшее использование
В результате запуска <i>main.ipynb</i> будут сгенерированы переводы текстов, написанных на немецком языке ( <a href=https://github.com/sonador88/text_translation/blob/main/data/multi30k_grader.csv>data/multi30k_grader.csv</a>), на английский язык. Результаты сохраняются в файл  <a href=https://github.com/sonador88/text_translation/blob/main/data/multi30k_solution.tsv>data/multi30k_solution.tsv</a>

Полученные метрики: bleu > 0.31, accuracy > 0.65. 

Модель продемонстрировала хорошую способность к переводу с немецкого языка на английский, данную реализацию можно использовать для написания сервисов-переводчиков, изменить на другой язык не будет большой проблемой, если собрать соответствующие обучающие и тренировочные датасеты под нужные языки и подобрать токенизаторы.
