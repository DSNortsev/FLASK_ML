#!/bin/bash

APP="flask_ml"

# Stop all running containers
docker stop $(docker ps | grep ${APP} | grep -o '^\S*' | xargs)

# Remove docker image if exists
docker rmi -f $(docker images | grep ${APP} | awk '{print $3}')

# Build docker image
docker build -t ${APP} .

# Run Flask app in docker container
docker run -i -d -p 1313:1313 -e HOST=0.0.0.0 -e DEBUG=False ${APP}