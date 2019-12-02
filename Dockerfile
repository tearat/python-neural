FROM python:3.7

WORKDIR /home/rrayd/my/python/neuro

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "./neuro.py" ]
