# Base image
FROM ubuntu:18.04

# create and set working directory
RUN mkdir /app
WORKDIR /app

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ADD . /app/

RUN apt-get update && apt-get install \
    -y --no-install-recommends python3 python3-virtualenv git

RUN apt-get install -y --no-install-recommends \
    libjpeg8-dev libtiff5-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libgtk2.0-dev \
    liblapacke-dev checkinstall

ENV VIRTUAL_ENV=/opt/venv

RUN python3 -m virtualenv --python=/usr/bin/python3 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install -r requirements.txt
RUN pip install git+https://github.com/waleedka/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI
ENV FLASK_APP=mrcnn/app.py

EXPOSE 5000

#CMD ["flask", "run"]
CMD ["/bin/bash"]
