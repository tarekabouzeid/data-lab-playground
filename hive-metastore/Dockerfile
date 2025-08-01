FROM openjdk:8-jre-slim

# Install required packages
RUN apt-get update && apt-get install -y \
    curl \
    netcat \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV HIVE_VERSION=3.1.3
ENV HADOOP_VERSION=3.3.4
ENV POSTGRES_VERSION=42.5.1

# Create hive user
RUN groupadd -r hive && useradd -r -g hive hive

# Download and install Hadoop
RUN curl -L https://archive.apache.org/dist/hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz | tar -xz -C /opt/ && \
    ln -s /opt/hadoop-${HADOOP_VERSION} /opt/hadoop && \
    chown -R hive:hive /opt/hadoop*

# Download and install Hive
RUN curl -L https://archive.apache.org/dist/hive/hive-${HIVE_VERSION}/apache-hive-${HIVE_VERSION}-bin.tar.gz | tar -xz -C /opt/ && \
    ln -s /opt/apache-hive-${HIVE_VERSION}-bin /opt/hive && \
    chown -R hive:hive /opt/apache-hive*

# Download PostgreSQL JDBC driver and S3 JARs
RUN curl -L https://jdbc.postgresql.org/download/postgresql-${POSTGRES_VERSION}.jar -o /opt/hive/lib/postgresql-${POSTGRES_VERSION}.jar && \
    curl -L https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar -o /opt/hive/lib/hadoop-aws-3.3.4.jar && \
    curl -L https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.367/aws-java-sdk-bundle-1.12.367.jar -o /opt/hive/lib/aws-java-sdk-bundle-1.12.367.jar

# Set environment variables
ENV JAVA_HOME=/usr/local/openjdk-8
ENV HADOOP_HOME=/opt/hadoop
ENV HIVE_HOME=/opt/hive
ENV HADOOP_CONF_DIR=/opt/hadoop/etc/hadoop
ENV PATH=$PATH:$HADOOP_HOME/bin:$HIVE_HOME/bin

# Copy configuration files
COPY hive-site.xml /opt/hive/conf/
COPY entrypoint.sh /opt/entrypoint.sh

RUN chmod +x /opt/entrypoint.sh && \
    chown hive:hive /opt/entrypoint.sh

USER hive
WORKDIR /opt/hive

EXPOSE 9083

ENTRYPOINT ["/opt/entrypoint.sh"]
