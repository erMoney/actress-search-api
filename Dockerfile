FROM heroku/heroku:16

RUN apt-get update && apt-get install -y \
    python3-pip \
    libsm6 \
    build-essential \
    cmake \
    pkg-config \
    libx11-dev \
    libatlas-base-dev \
    libgtk-3-dev \
    libboost-python-dev

ADD ./webapp/requirements.txt /tmp/requirements.txt

RUN pip3 install --upgrade pip && \
    pip3 install -r /tmp/requirements.txt

RUN git clone https://github.com/cmusatyalab/openface openface && \
    cd openface && python setup.py install


ADD ./webapp /opt/webapp/
WORKDIR /opt/webapp

RUN useradd -m myuser
USER myuser

CMD gunicorn --bind 0.0.0.0:$PORT wsgi
