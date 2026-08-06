#! /bin/bash
set -u
set -e
set -x

role=${1}
instance_id=${2}
model_path=${3}
port=${4}
numa_bind=${5}
log_dir=${6}
enable_nsys=${7}
config_file=${8}
cuda_devices=${9}
ctx_instances=${10:-1}
gen_instances=${11:-1}

# Set CUDA_VISIBLE_DEVICES from script argument (srun --export cannot
# reliably pass comma-separated values inside shared containers).
export CUDA_VISIBLE_DEVICES=${cuda_devices}
export LD_LIBRARY_PATH=/opt/nccl-gin/lib:/opt/mooncake-paged-gin/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# Paged GIN consumes these before its cache sender/receiver threads start.
# Other transfer backends ignore them.
export TRTLLM_MOONCAKE_PAGED_GIN_STARTUP_PRECONNECT="${TRTLLM_MOONCAKE_PAGED_GIN_STARTUP_PRECONNECT:-1}"
export TRTLLM_MOONCAKE_PAGED_GIN_PRECONNECT_ROLE="${role}"
export TRTLLM_MOONCAKE_PAGED_GIN_PRECONNECT_INSTANCE="$((instance_id + 1))"
export TRTLLM_MOONCAKE_PAGED_GIN_CTX_INSTANCES="${ctx_instances}"
export TRTLLM_MOONCAKE_PAGED_GIN_GEN_INSTANCES="${gen_instances}"
export TRTLLM_MOONCAKE_PAGED_GIN_PRECONNECT_DIR="${log_dir}/mooncake-paged-gin-preconnect"
export TRTLLM_MOONCAKE_PAGED_GIN_RENDEZVOUS_TIMEOUT_SECONDS="${TRTLLM_MOONCAKE_PAGED_GIN_RENDEZVOUS_TIMEOUT_SECONDS:-2100}"
export TRTLLM_MOONCAKE_PAGED_GIN_INIT_TIMEOUT_SECONDS="${TRTLLM_MOONCAKE_PAGED_GIN_INIT_TIMEOUT_SECONDS:-900}"

# Clear UCX_TLS for specific clusters
unset UCX_TLS

echo "SLURM_PROCID: ${SLURM_PROCID}, hostname: $(hostname), instance_id: ${instance_id}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "NCCL_DEBUG: ${NCCL_DEBUG:-<unset>}"
echo "NCCL_DEBUG_SUBSYS: ${NCCL_DEBUG_SUBSYS:-<unset>}"
echo "NCCL_DEBUG_FILE: ${NCCL_DEBUG_FILE:-<unset>}"
echo "NCCL_SOCKET_IFNAME: ${NCCL_SOCKET_IFNAME:-<unset>}"
echo "NCCL_LAUNCH_ORDER_IMPLICIT: ${NCCL_LAUNCH_ORDER_IMPLICIT:-<unset>}"
echo "NCCL_LAUNCH_RACE_FATAL: ${NCCL_LAUNCH_RACE_FATAL:-<unset>}"
echo "TRTLLM_MOONCAKE_PAGED_GIN_STARTUP_PRECONNECT: ${TRTLLM_MOONCAKE_PAGED_GIN_STARTUP_PRECONNECT}"
echo "TRTLLM_MOONCAKE_PAGED_GIN_PRECONNECT_ROLE: ${TRTLLM_MOONCAKE_PAGED_GIN_PRECONNECT_ROLE}"
echo "TRTLLM_MOONCAKE_PAGED_GIN_PRECONNECT_INSTANCE: ${TRTLLM_MOONCAKE_PAGED_GIN_PRECONNECT_INSTANCE}"
echo "TRTLLM_MOONCAKE_PAGED_GIN_PRECONNECT_DIR: ${TRTLLM_MOONCAKE_PAGED_GIN_PRECONNECT_DIR}"
echo "TRTLLM_MOONCAKE_PAGED_GIN_RENDEZVOUS_TIMEOUT_SECONDS: ${TRTLLM_MOONCAKE_PAGED_GIN_RENDEZVOUS_TIMEOUT_SECONDS}"
echo "TRTLLM_MOONCAKE_PAGED_GIN_INIT_TIMEOUT_SECONDS: ${TRTLLM_MOONCAKE_PAGED_GIN_INIT_TIMEOUT_SECONDS}"

if [ "${numa_bind}" = "true" ]; then
    numa_bind_cmd="numactl -m 0,1"
    echo "numactl -m 0,1 - Only allocate memory from nodes on GB200/GB300 NVL72"
else
    numa_bind_cmd=""
    echo "Not binding memory. If on GB200/GB300 NVL72, use \"numactl -m 0,1\" to only allocate memory from nodes."
fi

echo "config_file: ${config_file}"

nsys_prefix=""
if [ "${enable_nsys}" != "true" ]; then
    echo "nsys is not enabled, start normal flow"
else
    nsys_file=${log_dir}/nsys_worker_proc_${role}_${instance_id}_${SLURM_PROCID}
    echo "nsys is enabled on ${role} GPUs, TLLM_PROFILE_START_STOP=${TLLM_PROFILE_START_STOP}"
    nsys_prefix="nsys profile -o ${nsys_file} -f true -t cuda,nvtx,python-gil -c cudaProfilerApi --cuda-graph-trace node --capture-range-end=stop --gpu-metrics-devices=none"
fi

${nsys_prefix} trtllm-llmapi-launch ${numa_bind_cmd} \
    trtllm-serve ${model_path} \
        --host $(hostname) --port ${port} \
        --config ${config_file}
