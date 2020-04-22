#################################################
#  Short docker file to distribute some notebooks
#################################################

ARG FROMIMG_ARG=underworldcode/petsc4py:3.11.2-slim
FROM ${FROMIMG_ARG}

#################################################
# Non standard as the files come from the package

USER root

### DEPENDENCIES
RUN python3 -m pip install --upgrade --no-cache-dir \
	jupyter \
	stripy \
	lavavu \
	pyvirtualdisplay \
	pytest \
	webdav


# Add examples

# RUN ipython -c 'import quagmire; quagmire.documentation.install_documentation(path="Examples/Notebooks")'
# ADD  --chown=jovyan:jovyan $NB_DIR/0-StartHere.ipynb Examples/Notebooks/0-StartHere.ipynb


# RUN ipython profile create --parallel --profile=mpi && \
    # echo "c.IPClusterEngines.engine_launcher_class = 'MPIEngineSetLauncher'" >> $NB_WORK/.ipython/profile_mpi/ipcluster_config.py

# copy this file over so that no password is required
# COPY --chown=jovyan:users ./docs/development/docker/base/jupyter_notebook_config.json $NB_WORK/.jupyter/jupyter_notebook_config.json


## These are supplied by the build script
## build-dockerfile.sh

ARG NB_USER_NAME=jovyan
ARG IMAGENAME_ARG
ARG PROJ_NAME_ARG=quagmire
ARG NB_PORT_ARG=8888
ARG NB_PASSWD_ARG=""
ARG NB_DIR_ARG="Notebooks"
ARG START_NB_ARG="0-StartHere.ipynb"

# The args need to go into the environment so they
# can be picked up by commands/templates (defined previously)
# when the container runs

ENV IMAGENAME=$IMAGENAME_ARG
ENV PROJ_NAME=$PROJ_NAME_ARG
ENV NB_PORT=$NB_PORT_ARG
ENV NB_PASSWD=$NB_PASSWD_ARG
ENV NB_DIR=$NB_DIR_ARG
ENV START_NB=$START_NB_ARG
ENV NB_USER=$NB_USER_NAME
ENV NB_WORK=/home/$NB_USER


# create a volume
VOLUME $NB_WORK/workspace
WORKDIR $NB_WORK

# expose notebook port server port
EXPOSE $NB_PORT


# change ownership of everything
RUN groupadd -fr $NB_USER
RUN usermod -a -G $NB_USER $NB_USER
# RUN useradd -ms /bin/bash $NB_USER
RUN chown -R jovyan:jovyan $NB_WORK


## NOW INSTALL NOTEBOOKS

### Quagmire - Notebooks

ENV MODULE_DIR="quagmire-src"
ADD --chown=jovyan:jovyan . / $MODULE_DIR/
RUN cd $MODULE_DIR && python3 -m pip install --no-deps --no-cache-dir --upgrade . && \
    cd .. && rm -rf $MODULE_DIR

RUN ipython3 -c 'import quagmire; quagmire.documentation.install_documentation(path="/home/jovyan/")'


# Trust all notebooks
USER $NB_USER
RUN find -name \*.ipynb  -print0 | xargs -0 jupyter trust

WORKDIR /home/jovyan/

ENV PATH=$PETSC_BIN:$PATH

# launch notebook
CMD ["jupyter", "notebook", "--ip='0.0.0.0'", "--NotebookApp.token='' ", "--no-browser"]
