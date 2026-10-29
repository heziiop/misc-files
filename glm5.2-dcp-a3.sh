#!/bin/bash

export ASCEND_MF_TRANSFER_PROTOCOL="sdma"

# cpu高性能
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# 绑核
export SGLANG_SET_CPU_AFFINITY=1

# 内存碎片
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32

export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1200

P_IP=('61.47.19.68' '61.47.19.76')
D_IP=('61.47.19.70' '61.47.19.71')

# pd传输，IP设置为P节点
export ASCEND_MF_STORE_URL="tcp://${P_IP[0]}:24670"
# 模型路径
MODEL_PATH=/home/weights/GLM-5.2-w8a8

LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`
echo "${LOCAL_HOST1}"
echo "${LOCAL_HOST2}"


# prefill
for i in "${!P_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${P_IP[$i]}" || "$LOCAL_HOST2" == "${P_IP[$i]}" ]];
    then
        echo "${P_IP[$i]}"
        export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
        export TASK_QUEUE_ENABLE=2
        export HCCL_SOCKET_IFNAME=enp196s0f0
        export GLOO_SOCKET_IFNAME=enp196s0f0
        export DEEPEP_HCCL_BUFFSIZE=2500
        ### 这个不能去掉，开启pp时需要首层有topk_idnex
        export SGLANG_PP_LAYER_PARTITION=18,20,24,16
        
        # P节点
        python -m sglang.launch_server \
            --model-path "${MODEL_PATH}" \
            --disaggregation-mode prefill \
            --host ${P_IP[$i]} \
            --port 8003 \
            --dist-init-addr ${P_IP[0]}:5000 \
            --disaggregation-bootstrap-port 8998 \
            --trust-remote-code \
            --nnodes 2 \
            --node-rank $i \
            --tp-size 8 \
            --ep-size 8 \
            --dp-size 8 \
            --enable-dp-attention \
            --mem-fraction-static 0.88 \
            --attention-backend ascend \
            --device npu \
            --quantization modelslim \
            --disaggregation-transfer-backend ascend \
            --max-running-requests 64 \
            --served-model-name glm-5.2-w8a8 \
            --reasoning-parser glm45 \
            --tool-call-parser glm47 \
            --enable-metrics \
            --chunked-prefill-size 8192 \
            --max-prefill-tokens 65536 \
            --context-length 220000 \
            --moe-a2a-backend deepep \
            --deepep-mode normal \
            --disable-shared-experts-fusion \
            --disable-cuda-graph \
            --dtype bfloat16 \
            --enable-dp-lm-head \
            --moe-dense-tp 1 \
            --pp-size 4 \
            --enable-hierarchical-cache \
            --hicache-io-backend kernel_ascend \
            --enable-cache-report \
            --hicache-size 80
        break
    fi
done


# decode
for i in "${!D_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${D_IP[$i]}" || "$LOCAL_HOST2" == "${D_IP[$i]}" ]];
    then
        echo "${D_IP[$i]}"
        export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
        export SGLANG_ENABLE_SPEC_V2=1
        export HCCL_SOCKET_IFNAME=enp196s0f0
        export GLOO_SOCKET_IFNAME=enp196s0f0
        export DEEPEP_HCCL_BUFFSIZE=2500
        export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16
        export TASK_QUEUE_ENABLE=0
        # 增加后D节点才能拉起
        # export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
        # D节点
        python -m sglang.launch_server \
            --model-path "${MODEL_PATH}" \
            --disaggregation-mode decode \
            --host ${D_IP[$i]} \
            --port 8004 \
            --dist-init-addr ${D_IP[0]}:5000 \
            --trust-remote-code \
            --nnodes 2 \
            --node-rank $i \
            --tp-size 32 \
            --dp-size 8 \
            --enable-dp-attention \
            --dcp-size 4 \
            --ep-size 32 \
            --mem-fraction-static 0.88 \
            --max-running-requests 32 \
            --attention-backend ascend \
            --device npu \
            --quantization modelslim \
            --served-model-name glm-5.2-w8a8 \
            --reasoning-parser glm45 \
            --tool-call-parser glm47 \
            --enable-metrics \
            --moe-a2a-backend deepep \
            --deepep-mode low_latency \
            --cuda-graph-max-bs-decode 4 \
            --disaggregation-transfer-backend ascend \
            --watchdog-timeout 9000 \
            --context-length 220000 \
            --disable-shared-experts-fusion \
            --dtype bfloat16 \
            --speculative-draft-model-quantization unquant \
            --load-balance-method round_robin \
            --speculative-algorithm NEXTN \
            --speculative-num-steps 3 \
            --speculative-eagle-topk 1 \
            --speculative-num-draft-tokens 4
        break
    fi
done

