# Hive Metastore 4.2.1 (standalone image: Java 21, Hadoop 3.4.1)
#
# Spark reaches Iceberg tables through the HMS built-in Iceberg REST catalog
# (http://hive-metastore:9084/iceberg) instead of the Thrift `get_table` call that
# HMS >= 4.0.1 removed. Trino's hive/lakehouse catalogs keep using Thrift on :9083.
FROM apache/hive:standalone-metastore-4.2.1

USER root

# The standalone image ships hadoop-aws 3.4.1 but neither the AWS SDK v2 bundle it
# needs nor a Postgres JDBC driver. bundle-2.24.6 is the SDK version hadoop-aws 3.4.1
# is built against. hadoop-aws lives in tools/lib, which is not on the HMS classpath.
ENV POSTGRES_JDBC_VERSION=42.7.5 \
    AWS_SDK_VERSION=2.24.6

RUN curl -fsSL https://repo1.maven.org/maven2/org/postgresql/postgresql/${POSTGRES_JDBC_VERSION}/postgresql-${POSTGRES_JDBC_VERSION}.jar \
        -o /opt/hive/lib/postgresql-${POSTGRES_JDBC_VERSION}.jar \
    && curl -fsSL https://repo1.maven.org/maven2/software/amazon/awssdk/bundle/${AWS_SDK_VERSION}/bundle-${AWS_SDK_VERSION}.jar \
        -o /opt/hive/lib/bundle-${AWS_SDK_VERSION}.jar \
    && ln -s /opt/hadoop/share/hadoop/tools/lib/hadoop-aws-3.4.1.jar /opt/hive/lib/hadoop-aws-3.4.1.jar \
    && chown hive:hive /opt/hive/lib/postgresql-${POSTGRES_JDBC_VERSION}.jar /opt/hive/lib/bundle-${AWS_SDK_VERSION}.jar

# The image entrypoint regenerates metastore-site.xml and core-site.xml from these
# templates (envsubst) on every start, so config must live in the templates.
COPY --chown=hive:hive metastore-site.xml.template core-site.xml.template /opt/hive/conf/

USER hive

# 9083 = Thrift metastore, 9084 = Iceberg REST catalog
EXPOSE 9083 9084
