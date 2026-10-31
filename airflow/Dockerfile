# Airflow for lab pipelines (Helm chart: airflow.labPipelines.enabled, e.g. the workstation profile).
# The stock image plus the clients the northwind_daily DAG uses: the Spark Connect client (same Spark line as the
# lab) in Airflow's environment, and dbt in its own virtualenv (its pins do not mix with Airflow's).
FROM apache/airflow:3.3.2-python3.12

ARG AIRFLOW_VERSION=3.3.2
ARG SPARK_VERSION=4.1.3
ARG DBT_CORE_VERSION=1.12.5
ARG DBT_TRINO_VERSION=1.10.6

USER root
RUN python3 -m venv /opt/dbt \
    && /opt/dbt/bin/pip install --no-cache-dir "dbt-core==${DBT_CORE_VERSION}" "dbt-trino==${DBT_TRINO_VERSION}" \
    && chmod -R a+rX /opt/dbt
USER airflow
# Keep Airflow at its version while adding the Spark Connect client.
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" "pyspark-client==${SPARK_VERSION}"

ENV DBT_BIN=/opt/dbt/bin/dbt \
    DBT_SEND_ANONYMOUS_USAGE_STATS=false
