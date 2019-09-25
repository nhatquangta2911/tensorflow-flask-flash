FROM python:3.6

COPY . /app

WORKDIR /app

RUN pip install -r ./requirements.txt

RUN pip install tensorflow==2.0.0-rc2

ENTRYPOINT [ "python" ]

CMD [ "app.py" ]

