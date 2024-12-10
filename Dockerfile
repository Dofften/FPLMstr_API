
FROM python:3.12.0


WORKDIR /app


COPY . /app


RUN pip install --no-cache-dir --upgrade -r requirements.txt


CMD ["python", "app.py"]