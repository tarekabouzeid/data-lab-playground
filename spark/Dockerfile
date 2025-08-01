FROM bitnami/spark:4.0.0

USER root

# Install curl
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Download additional JARs for S3 support
RUN curl -L https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar -o /opt/bitnami/spark/jars/hadoop-aws-3.3.4.jar && \
    curl -L https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.367/aws-java-sdk-bundle-1.12.367.jar -o /opt/bitnami/spark/jars/aws-java-sdk-bundle-1.12.367.jar && \
    rm -rf /opt/bitnami/spark/jars/hadoop-aws-3.4.1.jar

# Fix permissions for configuration directory
RUN chmod -R 755 /opt/bitnami/spark/conf && \
    chown -R 1001:1001 /opt/bitnami/spark/conf

USER 1001
