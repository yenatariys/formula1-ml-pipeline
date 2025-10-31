FROM apache/spark:latest

RUN apt-get update \
	&& apt-get install -y --no-install-recommends python3-pip \
	&& python3 -m pip install --no-cache-dir numpy pandas pyarrow \
	&& rm -rf /var/lib/apt/lists/*
