#!/bin/bash
# End-to-end test for the lakehouse core: MinIO + HMS + Trino + Spark + Iceberg.
# Does NOT need a GPU: Ollama / Phoenix / Qdrant / Jupyter are not started.
#
# Usage:  tests/e2e/run-e2e.sh [--build] [--down]
#   --build   (re)build the hive-metastore, trino and spark images first
#   --down    stop the started services when finished
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

BUILD=false; DOWN=false
for arg in "$@"; do
  case $arg in
    --build) BUILD=true ;;
    --down)  DOWN=true ;;
    *) echo "Unknown option: $arg"; exit 2 ;;
  esac
done

SERVICES=(minio metastore-db hive-metastore trino spark-master spark-worker)

if [ "$BUILD" = true ]; then
  for svc in hive-metastore trino spark; do
    docker build -t "datalab-playground/$svc" "./$svc"
  done
fi

echo "🚀 Starting: ${SERVICES[*]}"
docker compose up -d "${SERVICES[@]}"

wait_for() {  # name, command, timeout seconds
  local name=$1 cmd=$2 timeout=${3:-180} start=$SECONDS
  echo -n "⏳ Waiting for $name "
  until eval "$cmd" >/dev/null 2>&1; do
    if (( SECONDS - start > timeout )); then echo " ❌ timeout"; return 1; fi
    echo -n "."; sleep 3
  done
  echo " ✅"
}

wait_for "MinIO" "docker exec minio mc alias set local http://localhost:9000 minioadmin minioadmin123"
docker exec minio mc mb local/warehouse --ignore-existing
wait_for "Hive Metastore :9083" "docker exec hive-metastore bash -c 'echo > /dev/tcp/localhost/9083'" 300
wait_for "Trino" "docker exec trino curl -sf http://localhost:8080/v1/info | grep -q '\"starting\":false'" 300
wait_for "Spark worker registration" "docker exec spark-master curl -sf http://localhost:8080/json/ | grep -q '\"aliveworkers\" *: *[1-9]'" 120

NETWORK=$(docker inspect -f '{{range $k, $v := .NetworkSettings.Networks}}{{$k}}{{end}}' spark-master)

echo "🧪 Running e2e test (driver container on network $NETWORK)"
set +e
docker run --rm --name e2e-driver --hostname e2e-driver --network "$NETWORK" \
  -e AWS_REGION=us-east-1 \
  -v "$ROOT/spark/conf/spark-defaults.conf:/opt/spark/conf/spark-defaults.conf:ro" \
  -v "$ROOT/tests/e2e:/e2e:ro" \
  datalab-playground/spark:latest \
  /opt/spark/bin/spark-submit \
    --master spark://spark-master:7077 \
    --conf spark.driver.host=e2e-driver \
    --conf spark.executor.memory=1g \
    /e2e/e2e_lakehouse.py
RC=$?
set -e

if [ "$DOWN" = true ]; then docker compose stop "${SERVICES[@]}"; fi
exit $RC
