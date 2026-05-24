# Pin specific version with Python 3.12 from quay.io registry
FROM quay.io/jupyter/base-notebook:python-3.12

USER root

# Install Java 17 and tools in single layer with cleanup
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jdk \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# Download and install Spark (pinned version for cache stability)
ENV SPARK_VERSION=4.1.0
ENV HADOOP_VERSION=3.4.2
ENV SPARK_HOME=/opt/spark
ENV HIVE_VERSION=4.1.0
ENV PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin

# Download Spark and prepare directories in single layer
RUN wget -q https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.tgz \
    && tar -xzf spark-${SPARK_VERSION}-bin-hadoop3.tgz \
    && mv spark-${SPARK_VERSION}-bin-hadoop3 $SPARK_HOME \
#    && rm -f /opt/spark/jars/hive-* \
#    && rm -f /opt/spark/jars/spark-hive_* \
#    && rm -f /opt/spark/jars/hive-metastore-* \
#    && rm -f /opt/spark/jars/hive-exec-* \
#    && rm -f /opt/spark/jars/hive-common-* \
#    && curl -L https://repo1.maven.org/maven2/org/apache/hive/hive-metastore/${HIVE_VERSION}/hive-metastore-${HIVE_VERSION}.jar -o /opt/spark/jars/hive-metastore-${HIVE_VERSION}.jar \
#    && curl -L https://repo1.maven.org/maven2/org/apache/hive/hive-exec/${HIVE_VERSION}/hive-exec-${HIVE_VERSION}.jar -o /opt/spark/jars/hive-exec-${HIVE_VERSION}.jar \
#    && curl -L https://repo1.maven.org/maven2/org/apache/hive/hive-common/${HIVE_VERSION}/hive-common-${HIVE_VERSION}.jar -o /opt/spark/jars/hive-common-${HIVE_VERSION}.jar \
    && rm spark-${SPARK_VERSION}-bin-hadoop3.tgz \
    && mkdir -p /home/jovyan/.local/share/jupyter/runtime \
    && mkdir -p /home/jovyan/.jupyter \
    && mkdir -p /home/jovyan/.ipython \
    && mkdir -p /home/jovyan/.cache \
    && chown -R $NB_UID:$NB_GID $SPARK_HOME \
    && chown -R $NB_UID:$NB_GID /home/jovyan/.local \
    && chown -R $NB_UID:$NB_GID /home/jovyan/.jupyter \
    && chown -R $NB_UID:$NB_GID /home/jovyan/.ipython \
    && chown -R $NB_UID:$NB_GID /home/jovyan/.cache

# Download all JARs in single layer for better caching
RUN curl -fsSL https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/${HADOOP_VERSION}/hadoop-aws-${HADOOP_VERSION}.jar \
         -o $SPARK_HOME/jars/hadoop-aws-${HADOOP_VERSION}.jar \
    && curl -fsSL https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-spark-runtime-4.1_2.13/1.11.0/iceberg-spark-runtime-4.1_2.13-1.11.0.jar \
         -o $SPARK_HOME/jars/iceberg-spark-runtime-4.1_2.13-1.11.0.jar \
    && curl -fsSL https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-aws-bundle/1.11.0/iceberg-aws-bundle-1.11.0.jar \
         -o $SPARK_HOME/jars/iceberg-aws-bundle-1.11.0.jar \
    && curl -fsSL https://repo1.maven.org/maven2/software/amazon/awssdk/bundle/2.41.1/bundle-2.41.1.jar \
         -o $SPARK_HOME/jars/aws-java-sdk-bundle-2.41.1.jar

# Switch to jovyan user for pip installations and Jupyter config
USER $NB_UID

# Install all Python packages in single layer with --no-cache-dir to reduce size
# Pin pyyaml>=6.0 first to avoid building old version from source on Python 3.12
# Note: Elyra removed due to incompatibility with JupyterLab 4.x (base image uses 4.4.5)
RUN pip install --no-cache-dir \
    'pyyaml>=6.0' \
    pyspark==${SPARK_VERSION} \
    trino \
    boto3 \
    pandas \
    numpy \
    matplotlib \
    seaborn \
    plotly \
    openai \
    ollama \
    arize-phoenix \
    qdrant-client \
    sentence-transformers \
    langchain==1.2.0 \
    langchain-classic==1.0.1 \
    langchain-community==0.4.1 \
    langchain-core==1.2.6 \
    langchain-experimental==0.4.1 \
    langchain-ollama==1.0.1 \
    langchain-openai==1.1.6 \
    langchain-qdrant==1.1.0 \
    langchain-text-splitters==1.1.0 \
    langgraph==1.0.5 \
    langgraph-checkpoint==3.0.1 \
    langgraph-prebuilt==1.0.5 \
    langgraph-sdk==0.3.1 \
    langsmith==0.6.0 \
    openinference-instrumentation-langchain==0.1.66 \
    && /opt/conda/bin/python -c "from jupyter_server.auth import passwd; print(passwd('123456'))" > /tmp/jupyter_password \
    && echo "c.ServerApp.password = '$(cat /tmp/jupyter_password)'" > /home/jovyan/.jupyter/jupyter_server_config.py \
    && echo "c.ServerApp.password_required = True" >> /home/jovyan/.jupyter/jupyter_server_config.py \
    && echo "c.ServerApp.token = ''" >> /home/jovyan/.jupyter/jupyter_server_config.py \
    && rm /tmp/jupyter_password

# Copy Spark configuration (should be last to avoid cache invalidation on config changes)
COPY --chown=$NB_UID:$NB_GID spark-defaults.conf $SPARK_HOME/conf/

# Set PySpark environment variables
ENV PYSPARK_PYTHON=/opt/conda/bin/python
ENV PYSPARK_DRIVER_PYTHON=/opt/conda/bin/python
ENV PYSPARK_DRIVER_PYTHON_OPTS='lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root'
