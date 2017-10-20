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

ONBUILD RUN FILE_ID="0B_Z70-KWunxkOWhkTldIUjBGZW8" && \
    curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /dev/null && \
    CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)" && \
    curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${FILE_ID}" -o model.h5

CMD gunicorn --bind 0.0.0.0:$PORT wsgi
