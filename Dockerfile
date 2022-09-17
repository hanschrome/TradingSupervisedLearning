FROM python:slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

CMD [ "python3", "main.py"]
