#!/bin/bash

PATH_NOW=$(dirname $(readlink -f "$0"))


CODE_DIR="${PATH_NOW}/code"
RES_DIR="${PATH_NOW}/code"

docker run -it --rm -h tabular_casa -v $CODE_DIR:/entry/code:ro -v $RES_DIR:/entry/results \
-u $(id -u):$(id -g) --gpus '"device='"$1"'"' --shm-size=10.24gb tabular:casa /bin/bash #-c "cd code && python tabular.py"

#echo "$1"
# aaa