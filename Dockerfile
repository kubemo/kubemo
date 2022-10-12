FROM python:3.8-slim

COPY . /kubemo

RUN pip install .

