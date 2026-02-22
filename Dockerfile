FROM python:3.10

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "face.py"]