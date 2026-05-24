FROM apache/hive:4.0.0

# Switch to root to install dependencies
USER root

# Set environment variables for dependency versions
# Note: hadoop-aws and aws-java-sdk-bundle are already bundled inside apache/hive:4.0.0
# at /opt/hadoop/share/hadoop/tools/lib/ (Hadoop 3.3.6 + AWS SDK v1 1.12.367).
# Do NOT download hadoop-aws-3.4.x — BulkDelete (required by 3.4.x) does not exist
# in the bundled Hadoop 3.3.6 jars and causes ClassNotFoundException at runtime.
ENV POSTGRES_VERSION=42.7.5

# Install PostgreSQL JDBC and symlink the already-bundled Hadoop S3A JARs into
# Hive's lib dir so HMS classloader picks them up. Using symlinks avoids version
# drift between hadoop-aws and the bundled hadoop-common.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://jdbc.postgresql.org/download/postgresql-${POSTGRES_VERSION}.jar \
       -o /opt/hive/lib/postgresql-${POSTGRES_VERSION}.jar \
    && chown hive:hive /opt/hive/lib/postgresql-${POSTGRES_VERSION}.jar \
    && ln -s /opt/hadoop/share/hadoop/tools/lib/hadoop-aws-3.3.6.jar \
             /opt/hive/lib/hadoop-aws-3.3.6.jar \
    && ln -s /opt/hadoop/share/hadoop/tools/lib/aws-java-sdk-bundle-1.12.367.jar \
             /opt/hive/lib/aws-java-sdk-bundle-1.12.367.jar

# Copy custom configuration files
COPY --chown=hive:hive hive-site.xml /opt/hive/conf/

# Switch back to hive user
USER hive

EXPOSE 9083

# Use the official image's entrypoint (no custom override needed)
