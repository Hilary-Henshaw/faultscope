#!/usr/bin/env bash
# Creates all FaultScope Kafka topics with appropriate configurations.
# Run once after Kafka is available.

set -euo pipefail

BROKER="${KAFKA_BROKER:-localhost:9092}"
REPLICATION="${REPLICATION_FACTOR:-1}"
PARTITIONS="${DEFAULT_PARTITIONS:-3}"

wait_for_kafka() {
    local retries=30
    echo "Waiting for Kafka at ${BROKER}..."
    until kafka-topics.sh \
            --bootstrap-server "${BROKER}" \
            --list &>/dev/null; do
        retries=$((retries - 1))
        if [[ "${retries}" -le 0 ]]; then
            echo "ERROR: Kafka not available after 30 attempts"
            exit 1
        fi
        sleep 2
    done
    echo "Kafka is ready."
}

create_topic() {
    local topic="$1"
    local partitions="${2:-${PARTITIONS}}"
    local retention_ms="${3:-604800000}"  # 7 days default

    if kafka-topics.sh \
            --bootstrap-server "${BROKER}" \
            --describe \
            --topic "${topic}" &>/dev/null; then
        echo "  [EXISTS]  ${topic}"
        return
    fi

    kafka-topics.sh \
        --bootstrap-server "${BROKER}" \
        --create \
        --topic "${topic}" \
        --partitions "${partitions}" \
        --replication-factor "${REPLICATION}" \
        --config "retention.ms=${retention_ms}" \
        --config "cleanup.policy=delete" \
        --config "compression.type=lz4"

    echo "  [CREATED] ${topic} (${partitions} partitions)"
}

wait_for_kafka

echo ""
echo "Creating FaultScope topics..."

# Raw sensor readings — high throughput, shorter retention
create_topic "faultscope.sensors.readings"    6 86400000      # 1 day
# Computed features — moderate throughput
create_topic "faultscope.features.computed"   3 259200000     # 3 days
# RUL / health predictions — lower volume
create_topic "faultscope.predictions.rul"     3 604800000     # 7 days
# Triggered incidents
create_topic "faultscope.incidents.triggered" 3 2592000000    # 30 days
# Dead-letter queue for unprocessable messages
create_topic "faultscope.dlq"                 1 2592000000    # 30 days

echo ""
echo "All topics created successfully."
kafka-topics.sh --bootstrap-server "${BROKER}" --list | grep faultscope
