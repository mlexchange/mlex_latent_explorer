FROM python:3.11

WORKDIR /app/work

COPY . /app/work
RUN pip install --upgrade pip
RUN pip install .

ENV HOME /app/work

CMD ["python", "frontend.py"]