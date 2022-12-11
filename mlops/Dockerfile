FROM python:3.9

WORKDIR /mlops
# Скачиваем/обновляем необходимые библиотеки для проекта 
COPY requirements.txt /mlops
RUN pip3 install --upgrade pip -r requirements.txt
# |ВАЖНЫЙ МОМЕНТ| копируем содержимое папки, где находится Dockerfile, 
# в рабочую директорию контейнера
COPY . /mlops
# Устанавливаем порт, который будет использоваться для сервера
EXPOSE 12345