FROM apache/hive:4.0.0

# Switch to root to install dependencies
USER root

# Set environment variables for dependency versions
ENV HADOOP_VERSION=3.4.2 \
    POSTGRES_VERSION=42.7.5

# Install curl and download all dependencies in a single layer (Debian Bullseye base)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://jdbc.postgresql.org/download/postgresql-${POSTGRES_VERSION}.jar \
       -o /opt/hive/lib/postgresql-${POSTGRES_VERSION}.jar \
    && curl -fsSL https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/${HADOOP_VERSION}/hadoop-aws-${HADOOP_VERSION}.jar \
       -o /opt/hive/lib/hadoop-aws-${HADOOP_VERSION}.jar \
    && curl -fsSL https://repo1.maven.org/maven2/software/amazon/awssdk/bundle/2.41.1/bundle-2.41.1.jar \
       -o /opt/hive/lib/aws-java-sdk-bundle-2.41.1.jar \
    && chown hive:hive /opt/hive/lib/*.jar

# Copy custom configuration files
COPY --chown=hive:hive hive-site.xml /opt/hive/conf/

# Switch back to hive user
USER hive

EXPOSE 9083

# Use the official image's entrypoint (no custom override needed)
