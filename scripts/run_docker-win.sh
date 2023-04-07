#!/bin/bash

PWD=`pwd`
DIRNAME=`echo $PWD | awk -F "/" '{ print $NF }'`

MEM=8G

IMAGE="nvcr.io/nvidia/pytorch:21.03-py3"

DISPLAY=:0

# docker run --rm --runtime=nvidia -v ${PWD}:/root/${DIRNAME} --shm-size $MEM -it ${IMAGE} /bin/bash
# docker run --rm --gpus all --runtime=nvidia -v ./:/root/SceneEgo --shm-size 8G -it nvcr.io/nvidia/pytorch:21.03-py3 /bin/bash
# docker run --rm  --runtime=nvidia -v ./:/root/SceneEgo --shm-size 8G -it nvcr.io/nvidia/pytorch:21.03-py3 /bin/bash
docker run --rm  --runtime=nvidia -v ./:/root/SceneEgo --shm-size 8G -v /tmp/.X11-unix:/tmp/.X11-unix:rw -e DISPLAY=${DISPLAY} -it nvcr.io/nvidia/pytorch:22.10-py3 /bin/bash
