# Base image
FROM python:3.6.8

RUN mkdir /app
WORKDIR /app

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

COPY . /app/

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir pycocotools==2.0.0

ENV FLASK_APP=app.py

EXPOSE 5000

WORKDIR /app/mrcnn
CMD ["flask", "run", "--host=0.0.0.0"]