FROM python:3.11

WORKDIR /app/work

COPY . /app/work
RUN pip install --upgrade pip
RUN pip install .

ENV HOME /app/work

CMD ["gunicorn", "-c", "gunicorn_config.py", "--reload", "src.app_layout:server"]
