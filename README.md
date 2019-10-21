# Mask R-CNN Deployment Task

This repo is an attempt to deploy the Mask R-CNN implementation from https://github.com/gabrielgarza/Mask_RCNN using Docker and Flask. The Dockerfile creates an image that will run a flask app that allows you to upload a file and returns the image after running object detection and segmentation.

## Running the docker container locally

Clone the repo and cd into the top level of the directory.

Build the docker image.

'''bash
docker build -t mrcnn_docker -f Dockerfile .
'''

Run the container.

'''bash
docker run -p 5000:5000 -it mrcnn_docker
'''

The container will run the flask app on http://0.0.0.0:5000 