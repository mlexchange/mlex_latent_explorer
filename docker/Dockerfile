# FROM python:3.9
# LABEL maintainer="THE MLEXCHANGE TEAM"

# RUN ls
# COPY docker/requirements.txt requirements.txt

# RUN apt-get update && apt-get install -y \
#     build-essential \
#     wget \
#     python3-pip\
#     ffmpeg\
#     libsm6\
#     libxext6 

# RUN pip3 install --upgrade pip &&\
#     pip3 install --timeout=2000 -r requirements.txt\
#     pip install git+https://github.com/taxe10/mlex_file_manager

# RUN git clone https://github.com/mlexchange/mlex_dash_component_editor

# # EXPOSE 8000

# WORKDIR /app/work
# ENV HOME /app/work
# COPY src src
# # ENV PYTHONUNBUFFERED=1
# RUN mv /mlex_dash_component_editor/src/dash_component_editor.py /app/work/src/dash_component_editor.py

# CMD ["bash"]
# #CMD python3 src/frontend.py
# CMD sleep 3600

FROM python:3.9
LABEL maintainer="THE MLEXCHANGE TEAM"

RUN ls
COPY docker/requirements.txt requirements.txt

RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    python3-pip\
    ffmpeg\
    libsm6\
    libxext6 

RUN pip3 install --upgrade pip &&\
    pip3 install --timeout=2000 -r requirements.txt\
    pip install git+https://github.com/taxe10/mlex_file_manager

RUN git clone https://github.com/mlexchange/mlex_dash_component_editor
WORKDIR /app/work
ENV HOME /app/work
COPY src src
RUN mv /mlex_dash_component_editor/src/dash_component_editor.py /app/work/src/dash_component_editor.py

CMD ["bash"]
#CMD sleep 3600 
CMD python3 src/frontend.py


