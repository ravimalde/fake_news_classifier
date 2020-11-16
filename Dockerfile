FROM python:3.6.9

LABEL Ravi Malde <ravidmalde@gmail.com>

ENV PYTHONDONTWRITEBYTECODE 1

ENV PYTHONUNBUFFERED 1

RUN apt-get update \
    && apt-get -y install gcc make \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

WORKDIR /Users/ravimalde/data_science/projects/fake-news-classifier

COPY ./requirements.txt /src/app/requirements.txt

RUN pip install -r /src/app/requirements.txt

COPY . .

ENTRYPOINT [ "python" ]

CMD ["./app.py"]



