FROM python:3.9
LABEL maintainer="THE MLEXCHANGE TEAM"

RUN ls

RUN pip3 install --upgrade pip &&\
    pip3 install -r .

WORKDIR /app/work
ENV HOME /app/work
COPY src src

CMD ["bash"]
CMD python3 src/frontend.py
