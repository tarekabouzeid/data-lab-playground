FROM apache/spark:4.0.1


USER root

# Install Java 17 and Python 3.12 (required for Spark 4.0)
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1


# Download additional JARs for S3 and Iceberg support
RUN curl -L https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar -o /opt/spark/jars/hadoop-aws-3.3.4.jar && \
    curl -L https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.367/aws-java-sdk-bundle-1.12.367.jar -o /opt/spark/jars/aws-java-sdk-bundle-1.12.367.jar && \
    curl -L https://search.maven.org/remotecontent?filepath=org/apache/iceberg/iceberg-spark-runtime-4.0_2.13/1.10.0/iceberg-spark-runtime-4.0_2.13-1.10.0.jar -o /opt/spark/jars/iceberg-spark-runtime-4.0_2.13-1.10.0.jar && \
    curl -L https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-aws-bundle/1.10.0/iceberg-aws-bundle-1.10.0.jar -o /opt/spark/jars/iceberg-aws-bundle-1.10.0.jar && \
    rm -rf /opt/spark/jars/hadoop-aws-3.4.1.jar


RUN chmod -R 755 /opt/spark/ && \
    chown -R 185:185 /opt/spark/

USER 185
