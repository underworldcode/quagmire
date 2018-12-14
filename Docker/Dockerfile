#################################################
#  Short docker file to distribute some notebooks
#################################################

ARG FROMIMG_ARG
FROM ${FROMIMG_ARG}

#################################################
# Non standard as the files come from the package

USER root
RUN useradd -ms /bin/bash jovyan

### DEPENDENCIES 
RUN python3 -m pip install --no-deps --upgrade --no-cache-dir stripy

### Quagmire - Notebooks

ENV MODULE_DIR="quagmire-src"
ADD --chown=jovyan:jovyan $MODULE_DIR $MODULE_DIR
RUN cd $MODULE_DIR && python3 -m pip install --no-deps --upgrade .


# Add examples
ADD  --chown=jovyan:jovyan $MODULE_DIR/quagmire/Examples  Examples

# RUN ipython -c 'import quagmire; quagmire.documentation.install_documentation(path="Examples/Notebooks")'
# ADD  --chown=jovyan:jovyan $NB_DIR/0-StartHere.ipynb Examples/Notebooks/0-StartHere.ipynb

# change ownership of everything
ENV NB_USER jovyan
RUN chown -R jovyan:jovyan $MODULE_DIR /home/jovyan
USER jovyan

## These are supplied by the build script
## build-dockerfile.sh

ARG IMAGENAME_ARG
ARG PROJ_NAME_ARG
ARG NB_PORT_ARG
ARG NB_PASSWD_ARG
ARG NB_DIR_ARG
ARG START_NB_ARG

# The args need to go into the environment so they
# can be picked up by commands/templates (defined previously)
# when the container runs

ENV IMAGENAME=$IMAGENAME_ARG
ENV PROJ_NAME=$PROJ_NAME_ARG
ENV NB_PORT=$NB_PORT_ARG
ENV NB_PASSWD=$NB_PASSWD_ARG
ENV NB_DIR=$NB_DIR_ARG
ENV START_NB=$START_NB_ARG


## NOW INSTALL NOTEBOOKS

# (This is not standard - nothing to do here )

## The notebooks (and other files we are serving up)
## ADD --chown=jovyan:jovyan  $NB_DIR /home/jovyan/Notebooks

# expose notebook port server port
EXPOSE $NB_PORT

# Trust all notebooks
RUN find -name \*.ipynb  -print0 | xargs -0 jupyter trust

# launch notebook
CMD ["jupyter", "notebook", "--ip='0.0.0.0'", "--no-browser"]