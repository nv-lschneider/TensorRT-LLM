#!/bin/bash

# Add error handling
set -e
set -u
trap 'echo "Error occurred at line $LINENO"; exit 1' ERR

# Add parameter validation
if [ "$#" -lt 10 ]; then
    echo "Error: Missing required arguments, got $# arguments, args: $@"
    echo "Usage: $0 model_name dataset_file multi_round num_gen_servers concurrency_list streaming log_path hostname port ucx_warmup_requests [exact_warmup_rounds]"
    exit 1
fi

model_name=$1
dataset_file=$2
multi_round=$3
num_gen_servers=$4
concurrency_list=$5
streaming=$6
log_path=$7
hostname=$8
port=$9
ucx_warmup_requests=${10}
exact_warmup_rounds=${11:-0}

if ! [[ "${multi_round}" =~ ^[1-9][0-9]*$ ]]; then
    echo "multi_round must be a positive integer, got ${multi_round}" >&2
    exit 1
fi
if ! [[ "${exact_warmup_rounds}" =~ ^[0-9]+$ ]]; then
    echo "exact_warmup_rounds must be a non-negative integer, got ${exact_warmup_rounds}" >&2
    exit 1
fi

# check process id is not 0
if [[ ${SLURM_PROCID} != "0" ]]; then
    echo "Process id is ${SLURM_PROCID} for loadgen, exiting"
    exit 0
fi

do_get_logs(){
    local input_file=$1
    local output_file=$2
    local mode=$3
    local start_line=$4
    # check mode is ctx or gen
    if [ "${mode}" = "ctx" ]; then
        sed -n "${start_line},\$p" ${input_file} | grep -a "'num_generation_tokens': 0" > ${output_file} || true
    elif [ "${mode}" = "gen" ]; then
        sed -n "${start_line},\$p" ${input_file} | grep -a "'num_ctx_requests': 0, 'num_ctx_tokens': 0" > ${output_file} || true
    else
        echo "Invalid mode: ${mode}"
        return 1
    fi
    return 0
}

do_process_all_logs(){
    local input_folder=$1
    local output_folder=$2
    local mode=$3
    if [ "${mode}" != "line" ] && [ "${mode}" != "log" ] && [ "${mode}" != "clean" ]; then
        echo "Invalid mode: ${mode}"
        exit 1
    fi
    local ctx_log
    local ctx_num
    local gen_log
    local gen_num
    local line_count
    local start_line
    for ctx_log in ${input_folder}/3_output_CTX_*.log; do
        if [ -f "${ctx_log}" ]; then
            ctx_num=$(basename "${ctx_log}" | sed 's/3_output_CTX_\([0-9]*\)\.log/\1/')
            if [ "${mode}" = "line" ]; then
                line_count=$(wc -l < ${ctx_log})
                echo ${line_count} > ${output_folder}/ctx_only_line_${ctx_num}.txt
            elif [ "${mode}" = "log" ]; then
                if [ ! -f "${output_folder}/ctx_only_line_${ctx_num}.txt" ]; then
                    start_line=0
                else
                    start_line=$(cat ${output_folder}/ctx_only_line_${ctx_num}.txt)
                    rm -f ${output_folder}/ctx_only_line_${ctx_num}.txt
                fi
                do_get_logs ${ctx_log} ${output_folder}/ctx_only_${ctx_num}.txt "ctx" ${start_line}
            elif [ "${mode}" = "clean" ]; then
                rm -f ${ctx_log}
            fi
        fi
    done
    # process all the gen log files in the input folder
    for gen_log in ${input_folder}/3_output_GEN_*.log; do
        if [ -f "${gen_log}" ]; then
            gen_num=$(basename "${gen_log}" | sed 's/3_output_GEN_\([0-9]*\)\.log/\1/')
            if [ "${mode}" = "line" ]; then
                line_count=$(wc -l < ${gen_log})
                echo ${line_count} > ${output_folder}/gen_only_line_${gen_num}.txt
            elif [ "${mode}" = "log" ]; then
                if [ ! -f "${output_folder}/gen_only_line_${gen_num}.txt" ]; then
                    start_line=0
                else
                    start_line=$(cat ${output_folder}/gen_only_line_${gen_num}.txt)
                    rm -f ${output_folder}/gen_only_line_${gen_num}.txt
                fi
                do_get_logs ${gen_log} ${output_folder}/gen_only_${gen_num}.txt "gen" ${start_line}
            elif [ "${mode}" = "clean" ]; then
                rm -f ${gen_log}
            fi
        fi
    done
    if [ "${mode}" = "clean" ]; then
        if [ -d "${tmp_start_logs}" ]; then
            mkdir -p ${log_path}/start_logs
            cp ${tmp_start_logs}/3_output_CTX_*.log ${log_path}/start_logs/ 2>/dev/null || true
            cp ${tmp_start_logs}/3_output_GEN_*.log ${log_path}/start_logs/ 2>/dev/null || true
            rm -rf ${tmp_start_logs}
        fi
    fi
}

run_benchmark_phase(){
    local phase_dataset=$1
    local prompt_count=$2
    local concurrency=$3
    local result_dir=$4

    python -m tensorrt_llm.serve.scripts.benchmark_serving \
        --model ${model_name} \
        --backend openai \
        --host ${hostname} \
        --port ${port} \
        --dataset-name "trtllm_custom" \
        --dataset-path ${phase_dataset} \
        --num-prompts ${prompt_count} \
        --max-concurrency ${concurrency} \
        --trust-remote-code \
        --ignore-eos \
        --no-test-input \
        --save-result \
        --result-dir "${result_dir}" \
        --result-filename "result.json" \
        --percentile-metrics "ttft,tpot,itl,e2el" \
        $(if [ "${streaming}" = "false" ]; then echo "--non-streaming"; fi)
}

validate_benchmark_result(){
    local result_file=$1
    local expected=$2
    local phase=$3

    python3 - "${result_file}" "${expected}" "${phase}" <<'PY'
import json
import sys

result_path, expected, phase = sys.argv[1], int(sys.argv[2]), sys.argv[3]
with open(result_path) as result_file:
    result = json.load(result_file)

completed = result.get("completed", 0)
failed = result.get("num_prompts", expected) - completed
if completed != expected or failed:
    print(
        f"{phase} failure: completed={completed}/{expected}, failed={failed}; "
        "terminating the allocation.",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
}

tmp_start_logs=/tmp/${SLURM_JOB_ID}/start_logs
mkdir -p ${tmp_start_logs}
cp ${log_path}/3_output_CTX_*.log ${tmp_start_logs}/ 2>/dev/null || true
cp ${log_path}/3_output_GEN_*.log ${tmp_start_logs}/ 2>/dev/null || true

# Legacy UCX-only warmup. Final comparisons use the exact-shape warmup below.
if [ "${ucx_warmup_requests}" -gt 0 ]; then
    echo "warming up ucx connections with small requests... ${ucx_warmup_requests}"
    python -m tensorrt_llm.serve.scripts.benchmark_serving \
        --model ${model_name} \
        --dataset-name random \
        --random-ids \
        --random-input-len 100 \
        --random-output-len 10 \
        --num-prompts ${ucx_warmup_requests} \
        --host ${hostname} \
        --port ${port} \
        --ignore-eos \
        --non-streaming
    echo "UCX warmup done"
fi

echo "Hostname: ${hostname}, Port: ${port}"
echo "Starting benchmark..."
for concurrency in ${concurrency_list}; do
    concurrency=$((concurrency * num_gen_servers))
    num_prompts=$((concurrency * multi_round))
    measurement_dataset=${dataset_file}

    if [ "${exact_warmup_rounds}" -gt 0 ]; then
        warmup_prompts=$((concurrency * exact_warmup_rounds))
        required_dataset_rows=$((warmup_prompts + num_prompts))
        available_dataset_rows=$(wc -l < "${dataset_file}")
        if [ "${available_dataset_rows}" -lt "${required_dataset_rows}" ]; then
            echo "Dataset has ${available_dataset_rows} rows but warmup plus measurement requires ${required_dataset_rows}" >&2
            exit 1
        fi

        dataset_dir="${log_path}/request_datasets"
        warmup_dataset="${dataset_dir}/warmup_concurrency_${concurrency}.jsonl"
        measurement_dataset="${dataset_dir}/measurement_concurrency_${concurrency}.jsonl"
        mkdir -p "${dataset_dir}"
        sed -n "1,${warmup_prompts}p" "${dataset_file}" > "${warmup_dataset}"
        sed -n "$((warmup_prompts + 1)),${required_dataset_rows}p" "${dataset_file}" > "${measurement_dataset}"

        warmup_dir="${log_path}/warmup_concurrency_${concurrency}"
        mkdir -p "${warmup_dir}"
        do_process_all_logs "${log_path}/" "${warmup_dir}" "line"
        echo "Warming up exact request shape with concurrency ${concurrency} ... ${warmup_prompts} prompts"
        run_benchmark_phase "${warmup_dataset}" "${warmup_prompts}" "${concurrency}" "${warmup_dir}"
        validate_benchmark_result "${warmup_dir}/result.json" "${warmup_prompts}" "Exact-shape warmup"
        do_process_all_logs "${log_path}/" "${warmup_dir}" "log"
        echo "Exact-shape warmup with concurrency ${concurrency} done"
    fi

    echo "Benchmarking with concurrency ${concurrency} ... ${num_prompts} prompts"
    mkdir -p ${log_path}/concurrency_${concurrency}
    do_process_all_logs ${log_path}/ ${log_path}/concurrency_${concurrency} "line"
    run_benchmark_phase "${measurement_dataset}" "${num_prompts}" "${concurrency}" "${log_path}/concurrency_${concurrency}"

    result_file="${log_path}/concurrency_${concurrency}/result.json"
    validate_benchmark_result "${result_file}" "${num_prompts}" "Benchmark"
    echo "Benchmark with concurrency ${concurrency} done"
    do_process_all_logs ${log_path}/ ${log_path}/concurrency_${concurrency} "log"
done
# do_process_all_logs ${log_path}/ ${log_path}/concurrency_${concurrency} "clean"
