FROM apache/spark:4.1.3

USER root

# -------------------------------
# Install Python 3.12 (Spark 4.1.3)
# -------------------------------
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    wget \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

RUN mkdir -p /opt/conda/bin && \
    ln -sf /usr/bin/python3.12 /opt/conda/bin/python && \
    ln -sf /usr/bin/python3.12 /opt/conda/bin/python3

# Spark Connect Python client deps (pyspark 4.1 "connect" extra). Used by the Helm
# chart's `helm test` pod and by pandas/Arrow UDFs on executors.
# --upgrade: jammy ships pip 22 in dist-packages, which breaks on 3.12 (no distutils).
RUN python3.12 -m ensurepip --upgrade \
    && python3.12 -m pip install --no-cache-dir --break-system-packages \
       pandas==2.3.3 pyarrow==21.0.0 numpy==2.3.4 \
       grpcio==1.76.0 grpcio-status==1.76.0 googleapis-common-protos==1.71.0 \
       zstandard==0.25.0

ENV PYSPARK_PYTHON=/opt/conda/bin/python
ENV PYSPARK_DRIVER_PYTHON=/opt/conda/bin/python

# -----------------------------------------
# Add Hadoop AWS + Iceberg (Spark 4.x)
# -----------------------------------------
RUN curl -fsSL https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.4.2/hadoop-aws-3.4.2.jar \
      -o /opt/spark/jars/hadoop-aws-3.4.2.jar && \
    curl -fsSL https://repo1.maven.org/maven2/software/amazon/awssdk/bundle/2.41.1/bundle-2.41.1.jar \
      -o /opt/spark/jars/aws-java-sdk-bundle-2.41.1.jar && \
    curl -fsSL https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-spark-runtime-4.1_2.13/1.12.0/iceberg-spark-runtime-4.1_2.13-1.12.0.jar \
      -o /opt/spark/jars/iceberg-spark-runtime-4.1_2.13-1.12.0.jar && \
    curl -fsSL https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-aws-bundle/1.12.0/iceberg-aws-bundle-1.12.0.jar \
      -o /opt/spark/jars/iceberg-aws-bundle-1.12.0.jar

# NOTE: Iceberg goes through the HMS Iceberg REST catalog (type=rest). Iceberg's
# HiveCatalog (Hive 2.3.10 client) cannot talk to HMS 4.2.1 — see AGENTS.md pitfall #1.

# -----------------------------------------
# Permissions
# -----------------------------------------
RUN chmod -R 755 /opt/spark \
    && chown -R 185:185 /opt/spark

USER 185
