FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu20.04

RUN set -ex \
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3-dev \
        python3-pip \
 && rm -rf /var/cache/apt \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY ./ /code

RUN pip3 install --no-cache-dir -r requirements.txt \
 && pip3 install -e .

CMD ["scoutbot", "--help"]
