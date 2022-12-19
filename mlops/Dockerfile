FROM python:3.10.9-slim

COPY mlops/requirements.txt .
RUN pip3 install -r requirements.txt

COPY . /mlops

WORKDIR /mlops

EXPOSE 12345