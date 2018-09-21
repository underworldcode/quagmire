FROM lmoresi/unimelb-debian-python:1.0.6

## =============================================================
## base - image ... whatever functionality you want to provide !
## This is my unix / python stuff (but it doesn't have underworld)
##
## This dockerfile builds an image from this content, and serves the
## sample web pages and notebooks at port 8080
##
## docker run -p 8181:8080 --name="docker-web-notebooks-test" -t lmoresi/lmoresi/docker-web-notebooks-module
## and then browse the docker VM ip address on port 8181 (for example)
##
## OR just use kitematic and click on the preview image
##
## =============================================================

## Quagmire requires petsc4py, petsc and mpi4py, and stripy.
## mkdocs is needed for the web front end to the notebooks

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  petsc-dev \
  libhdf5-openmpi-dev \
  xauth

RUN pip install mkdocs mkdocs-bootswatch pymdown-extensions\
                stripy \
                litho1pt0 \
                mpi4py \
                petsc4py


## These break the current installation of lavavu so we simply remove them

RUN DEBIAN_FRONTEND=noninteractive apt-get remove -y --no-install-recommends \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev

RUN pip install lavavu

# script for xvfb-run.  all docker commands will effectively run under this via the entrypoint
RUN printf "#\041/bin/sh \n rm -f /tmp/.X99-lock && xvfb-run -s '-screen 0 1600x1200x16' \$@" >> /usr/local/bin/xvfbrun.sh && \
                    chmod +x /usr/local/bin/xvfbrun.sh


RUN useradd --create-home --home-dir /home/jovyan --shell /bin/bash --user-group jovyan
WORKDIR  /home/jovyan

## These are the build templates etc
##
ADD --chown=jovyan:jovyan  mkdocs.yml mkdocs.yml
ADD --chown=jovyan:jovyan  jupyter-server-theme jupyter-server-theme
ADD --chown=jovyan:jovyan  quagmire quagmire
ADD --chown=jovyan:jovyan  fortran fortran
ADD --chown=jovyan:jovyan  scripts scripts
ADD --chown=jovyan:jovyan  docs docs
ADD --chown=jovyan:jovyan  setup.py setup.py

## Now we should build the quagmire code

RUN pip install .

USER jovyan

## Update / build the site
RUN ./scripts/run-sitebuilder.py

# Trust all notebooks
# RUN find -name \*.ipynb  -print0 | xargs -0 jupyter trust

# Persistent / Shared space outside the container
VOLUME /home/jovyan/Data/Resources
VOLUME /home/jovyan/build


# expose notebook port and server port
EXPOSE 8080 9999

# note we use xvfb which to mimic the X display for lavavu
ENTRYPOINT ["/usr/local/bin/tini", "--", "xvfbrun.sh"]

CMD scripts/run-jupyter.py
