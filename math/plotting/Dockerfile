FROM python:3.5.9-stretch

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

RUN pip install -U ipykernel

CMD ["bash"]
