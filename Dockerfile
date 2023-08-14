FROM pytorch/pytorch:latest

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install -r /app/requirements.txt

COPY main.py /app/
COPY models/model_gr_or_ph.pth /app/models/

EXPOSE 2345

# Задаем команду, которая будет выполняться при запуске контейнера
CMD ["python", "main.py"]
