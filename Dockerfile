FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
VOLUME /usr/src/app/out
COPY . .

CMD [ "python", "./main.py" ]