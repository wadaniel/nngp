set -e
#docker run -it -u $(id -u):$(id -g) \
docker run \
    --mount type=bind,source="$HOME/projects/nngp/",target=/home/sepp/gml \
	--rm -it "$1" /bin/bash

