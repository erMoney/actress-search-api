FROM heroku/heroku:16

RUN apt-get update && apt-get install -y \
    python3-pip \
    libsm6

ADD ./webapp/requirements.txt /tmp/requirements.txt

RUN pip3 install -r /tmp/requirements.txt

ADD ./webapp /opt/webapp/
WORKDIR /opt/webapp

RUN useradd -m myuser
USER myuser

CMD gunicorn --bind 0.0.0.0:$PORT wsgi
