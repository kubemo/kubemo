FROM python:3.8-slim

ADD . /kubemo

WORKDIR /kubemo

RUN pip install .

