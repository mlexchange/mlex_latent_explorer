FROM python:3.9
LABEL maintainer="THE MLEXCHANGE TEAM"

COPY pyproject.toml pyproject.toml

RUN pip3 install --upgrade pip &&\
    pip3 install .

WORKDIR /app/work
ENV HOME /app/work
COPY src src

CMD ["bash"]
CMD python3 src/frontend.py
