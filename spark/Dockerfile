FROM apache/spark:4.1.0

USER root

# -------------------------------
# Install Python 3.12 (Spark 4.1)
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

ENV PYSPARK_PYTHON=/opt/conda/bin/python
ENV PYSPARK_DRIVER_PYTHON=/opt/conda/bin/python

# -----------------------------------------
# REMOVE Spark-bundled Hive client libraries
# -----------------------------------------
#RUN rm -f /opt/spark/jars/hive-* \
#    && rm -f /opt/spark/jars/spark-hive_* \
#    && rm -f /opt/spark/jars/hive-metastore-* \
#    && rm -f /opt/spark/jars/hive-exec-* \
#    && rm -f /opt/spark/jars/hive-common-*

# -----------------------------------------
# Add Hadoop AWS + Iceberg (Spark 4.x)
# -----------------------------------------
RUN curl -L https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.4.2/hadoop-aws-3.4.2.jar \
      -o /opt/spark/jars/hadoop-aws-3.4.2.jar && \
    curl -L https://repo1.maven.org/maven2/software/amazon/awssdk/bundle/2.41.1/bundle-2.41.1.jar \
      -o /opt/spark/jars/aws-java-sdk-bundle-2.41.1.jar && \
    curl -L https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-spark-runtime-4.1_2.13/1.11.0/iceberg-spark-runtime-4.1_2.13-1.11.0.jar \
      -o /opt/spark/jars/iceberg-spark-runtime-4.1_2.13-1.11.0.jar && \
    curl -L https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-aws-bundle/1.11.0/iceberg-aws-bundle-1.11.0.jar \
      -o /opt/spark/jars/iceberg-aws-bundle-1.11.0.jar

# -----------------------------------------
# Add Hive Metastore 4.1 client jars (EXPLICIT)
# -----------------------------------------
#ENV HIVE_VERSION=4.1.0
#
#RUN curl -L https://repo1.maven.org/maven2/org/apache/hive/hive-metastore/${HIVE_VERSION}/hive-metastore-${HIVE_VERSION}.jar \
#      -o /opt/spark/jars/hive-metastore-${HIVE_VERSION}.jar && \
#    curl -L https://repo1.maven.org/maven2/org/apache/hive/hive-exec/${HIVE_VERSION}/hive-exec-${HIVE_VERSION}.jar \
#      -o /opt/spark/jars/hive-exec-${HIVE_VERSION}.jar && \
#    curl -L https://repo1.maven.org/maven2/org/apache/hive/hive-common/${HIVE_VERSION}/hive-common-${HIVE_VERSION}.jar \
#      -o /opt/spark/jars/hive-common-${HIVE_VERSION}.jar

# -----------------------------------------
# Permissions
# -----------------------------------------
RUN chmod -R 755 /opt/spark \
    && chown -R 185:185 /opt/spark

USER 185
