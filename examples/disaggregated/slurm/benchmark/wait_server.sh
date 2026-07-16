#!/bin/bash
set -euo pipefail

# Parse arguments
hostname=$1
port=$2
worker_failure_dir=${3:-}

# Constants for health check
# DeepSeek generation workers can spend more than 30 minutes in CPU-side
# checkpoint ingestion before registering their endpoints.
readonly TIMEOUT="${TRTLLM_DISAGG_SERVER_HEALTH_TIMEOUT:-1800}"
readonly HEALTH_CHECK_INTERVAL=10
readonly STATUS_UPDATE_INTERVAL=30


# Wait for server to be healthy
echo "Waiting for server ${hostname}:${port} to be healthy..."
start_time=$(date +%s)
while ! curl -s -o /dev/null -w "%{http_code}" "http://${hostname}:${port}/health" > /dev/null 2>&1; do
    if [ -n "${worker_failure_dir}" ] && compgen -G "${worker_failure_dir}/*.failed" > /dev/null; then
        echo "Error: Worker startup preconnect failed"
        for marker in "${worker_failure_dir}"/*.failed; do
            echo "Failure marker: ${marker}"
            sed -n '1,20p' "${marker}"
        done
        exit 1
    fi

    current_time=$(date +%s)
    elapsed=$((current_time - start_time))

    if [ $elapsed -ge $TIMEOUT ]; then
        echo "Error: Server not healthy after ${TIMEOUT} seconds"
        exit 1
    fi

    if [ $((elapsed % STATUS_UPDATE_INTERVAL)) -eq 0 ] && [ $elapsed -gt 0 ]; then
        echo "Waiting for server to be healthy... (${elapsed}s elapsed)"
    fi

    sleep $HEALTH_CHECK_INTERVAL
done

echo "Server is healthy and ready to accept requests!"
