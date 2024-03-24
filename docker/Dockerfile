FROM python:3.9
LABEL maintainer="THE MLEXCHANGE TEAM"

RUN ls
COPY docker/requirements.txt requirements.txt

RUN pip3 install --upgrade pip &&\
    pip3 install -r requirements.txt\
    pip install git+https://github.com/mlexchange/mlex_file_manager\
    pip install git+https://github.com/mlexchange/mlex_dash_component_editor

WORKDIR /app/work
ENV HOME /app/work
COPY src src

CMD ["bash"]
CMD python3 src/frontend.py
