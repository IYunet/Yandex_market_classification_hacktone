# Классификация фотографий товаров с Яндекс маркета

Нужно обучить модель определять тип инфографики/фона.
Классификация фото по качеству на следующие классы: 
- хорошая инфографика
- плохая инфографика
- инфографика не определена
- фото без инфографики с хорошим однотонным фоном 
- фото без инфографики с хорошим интерьерным фоном  
- фото без инфографики с плохим фоном
- фото без инфографики, тип фона не определен 

Работа вся происходит в облачном сервисе от Яндекса DataSphere

Решение - создать три модели:
- 1 модель определяет есть ли на фото инфографика, а после работает 2 или 3 модель в зависимости от результата 1 модели
- 2 модель определяет какой тип инфографики на фото
- 3 модель определяет какой фон на фото

В папке `original_information` лежит вся инфомрация, 
которую предоставил Яндекс Маркет в рамках хакатона. 
(или можно получить по [ссылке](https://disk.yandex.ru/d/7av1MzlLM9lOg), если еще не закрылм)

В папке `datasets` лежат следующие датасеты:

- `datasets/data.csv` - оригинальный датасет с разметкой (немного отформатирован)
- `datasets/data_correct.csv` - пути к картинкам в DataSphere и их разметка
- `datasets/data_for_binary.csv` - для обучение бинарной классификации: `фото с инфографикой` или фото `без инфографики`
- `datasets/data_for_infographics.csv`- для определение типа инфографики
- `datasets/data_for_binary.csv` - для определение типа фона
- `datasets/original_df_for_binary.csv` - оригинальный дасет, но с разметкой для двух классов

В файле `datasets/pandas_to_csv.ipynb` - описано как созданы датасеты

В файле `datasets/clone_image.ipynb` - код процесса загрузки в облако картинок (суммарный вес картинок примерно 120гб)

В файле `datasets/create_new_csv.ipynb`- код процесса формирование путей для картинок в облаке

# Запуск модели для классификации на фото и инфографику
Доля верных ответ на test составляет 86%

- `Можно локально запустить файл main.py`
- `Или через Docker`

## Запуск через Docker

Если arm архитектура процессора, то: `docker build --platform=linux/amd64 -t YOUR_NAME .`

Иначе: `docker build -t YOUR_NAME .`

## Теперь запускаем

Если arm архитектура процессора, то: `docker run -it --rm --name my-flask-app -p 2345:YOUR_PORT YOUR_NAME`

Иначе: `docker run -it --rm --platform=linux/amd64 --name my-flask-app -p 2345:YOUR_PORT YOUR_NAME`

## Пример сборки на ARM:
`docker build --platform=linux/amd64 -t my-model .`

`docker run -it --rm --platform=linux/amd64 --name my-flask-app -p 2345:2345 my-model`

В `datasets/original_df_for_binary.csv` можно найти ссылки на фотки и протестировать модель

Пример ссылки на фото: `avatars.mds.yandex.net/get-marketpic/8447408/picf59511c785f4cec0ee964ebdd0d44584/orig`

Нужно ее подставить в `?link=`

Пример запроса: `https://localhost:2345/predict/?link=avatars.mds.yandex.net/get-marketpic/8447408/picf59511c785f4cec0ee964ebdd0d44584/orig`