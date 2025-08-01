FROM jupyter/base-notebook:latest

USER root

# Install Java 17 and Python 3.12 (required for Spark 4.0)
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    openjdk-17-jdk \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Set JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# Download and install Spark 4.0.0
ENV SPARK_VERSION=4.0.0
ENV HADOOP_VERSION=3.3.4
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin

RUN wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.tgz && \
    tar -xzf spark-${SPARK_VERSION}-bin-hadoop3.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop3 $SPARK_HOME && \
    rm spark-${SPARK_VERSION}-bin-hadoop3.tgz

# Download additional JARs for S3 support
RUN curl -L https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar -o $SPARK_HOME/jars/hadoop-aws-3.3.4.jar && \
    curl -L https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.367/aws-java-sdk-bundle-1.12.367.jar -o $SPARK_HOME/jars/aws-java-sdk-bundle-1.12.367.jar

# Install pip for Python 3.12 - minimal setup
RUN python3.12 -m ensurepip --upgrade

# Create GenAI conda environment with ALL libraries (no duplication)
USER root
RUN conda create -n genai python=3.12 -y && \
    /opt/conda/envs/genai/bin/pip install --no-cache-dir \
    ipykernel \
    pyspark==4.0.0 \
    trino \
    boto3 \
    pandas \
    numpy \
    matplotlib \
    seaborn \
    plotly \
    jupyterlab \
    openai \
    ollama \
    arize-phoenix \
    requests \
    python-dotenv \
    qdrant-client \
    sentence-transformers \
    openinference-instrumentation-langchain \
    langchain_openai \
    langchain \
    langchain-qdrant \
    langchain-ollama \
    langchain-community \
    langchain-experimental \
    && /opt/conda/envs/genai/bin/python -m ipykernel install --user --name genai --display-name "GenAI DEV"

# Set ownership for conda environments
RUN chown -R $NB_UID:$NB_GID /opt/conda/envs/genai

# Create necessary directories and set ownership
RUN mkdir -p /home/jovyan/.local/share/jupyter/runtime && \
    mkdir -p /home/jovyan/.jupyter && \
    mkdir -p /home/jovyan/.ipython && \
    mkdir -p /home/jovyan/.cache && \
    chown -R $NB_UID:$NB_GID $SPARK_HOME && \
    chown -R $NB_UID:$NB_GID /home/jovyan/.local && \
    chown -R $NB_UID:$NB_GID /home/jovyan/.jupyter && \
    chown -R $NB_UID:$NB_GID /home/jovyan/.ipython && \
    chown -R $NB_UID:$NB_GID /home/jovyan/.cache && \
    chmod -R 755 /home/jovyan/.local && \
    chmod -R 755 /home/jovyan/.jupyter && \
    chmod -R 755 /home/jovyan/.ipython && \
    chmod -R 755 /home/jovyan/.cache

USER $NB_UID

# Set Jupyter password to "123456"
RUN mkdir -p /home/jovyan/.jupyter && \
    /opt/conda/envs/genai/bin/python -c "from jupyter_server.auth import passwd; print(passwd('123456'))" > /tmp/jupyter_password && \
    echo "c.ServerApp.password = '$(cat /tmp/jupyter_password)'" > /home/jovyan/.jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.password_required = True" >> /home/jovyan/.jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.token = ''" >> /home/jovyan/.jupyter/jupyter_server_config.py && \
    rm /tmp/jupyter_password

# Copy Spark configuration
COPY spark-defaults.conf $SPARK_HOME/conf/

# Set PySpark environment variables to use GenAI conda environment
ENV PYSPARK_PYTHON=python3.12
ENV PYSPARK_DRIVER_PYTHON=python3.12
ENV PYSPARK_DRIVER_PYTHON_OPTS='lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root'

# Set default kernel to GenAI environment
ENV JUPYTER_DEFAULT_KERNEL=genai