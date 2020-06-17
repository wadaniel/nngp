set -e
docker run -it -u $(id -u):$(id -g) \
	--mount type=bind,source="$HOME/projects/nngp/",target=/gml \
	--rm f41e06396ff3 /bin/bash

