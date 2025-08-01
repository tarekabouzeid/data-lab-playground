#!/bin/bash

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL to be ready..."
while ! nc -z metastore-db 5432; do
  sleep 1
done

echo "PostgreSQL is ready. Initializing schema..."

# Initialize schema if needed
if ! $HIVE_HOME/bin/schematool -dbType postgres -info; then
    echo "Initializing Hive Metastore schema..."
    $HIVE_HOME/bin/schematool -dbType postgres -initSchema
fi

echo "Starting Hive Metastore..."
exec $HIVE_HOME/bin/hive --service metastore
