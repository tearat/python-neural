FROM python:3.7

WORKDIR /home/rrayd/my/python/neuro

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD [ "python", "./neuro.py" ]
