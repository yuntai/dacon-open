#!/bin/bash

set -euxo pipefail

docker stop colab || true && docker rm colab || true

PASSWORD=qezh34quptyojlmnq2hxutmnop93zwxeseriqbaser
FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:20.12-py3

docker image prune -af

docker build -tcolab -<<EOF
FROM ${FROM_IMAGE_NAME}

WORKDIR /workspace

RUN pip install seaborn xgboost statsmodels transformers wandb pytorch_lightning torchmetrics
RUN pip install --upgrade "jupyter_http_over_ws>=0.0.7" && jupyter serverextension enable --py jupyter_http_over_ws
RUN pip install ipywidgets && jupyter nbextension enable --py widgetsnbextension

RUN ipython profile create && echo "c.TerminalInteractiveShell.editing_mode = 'vi'" >> ~/.ipython/profile_default/ipython_config.py

RUN mkdir /colab && echo -e "set -euxo pipefail\\n \\n OPENBLAS_CORETYPE=nehalem jupyter notebook \\n --NotebookApp.allow_origin='https://colab.research.google.com' --port=8888 \\n --NotebookApp.port_retries=0 --no-browser --ip=0.0.0.0 \\n --NotebookApp.token=$PASSWORD\\n" > /colab/run_colab.sh

EXPOSE 8888
ENTRYPOINT ["/bin/bash", "/colab/run_colab.sh"]
EOF


#docker run --gpus all --ipc=host --shm-size=1g --rm --name colab --ulimit memlock=-1 --ulimit stack=67108864 -d --ip 0.0.0.0 -p 9999:8888 -v /mnt/tmp:/mnt/tmp -v $(pwd):/workspace -v /mnt/datasets:/mnt/datasets colab
docker run --gpus all --ipc=host --rm --name colab --ulimit memlock=-1 --ulimit stack=67108864 -d --ip 0.0.0.0 -p 9999:8888 -v /mnt/tmp:/mnt/tmp -v $(pwd):/workspace -v /mnt/datasets:/mnt/datasets colab
