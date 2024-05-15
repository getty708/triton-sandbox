# ================ Triton Server ================
# SHM_SIZE = 4294967296 # 4GB
SHM_SIZE = 8589934592 # 8GB

TRITON_COMMON_PARAMS = \
	--model-repository=model_repository \
	--model-control-mode=explicit \
	--backend-config=onnxruntime,default-max-batch-size=8 \
	--pinned-memory-pool-byte-size ${SHM_SIZE} \
	--cuda-memory-pool-byte-size 0:${SHM_SIZE}

NSYS_COMMON_PARAMS = \
	-t nvtx,osrt,cuda --force-overwrite=true

NUM_MODELS := 4

.PHONY: start-triton
start-triton:
	tritonserver \
		${TRITON_COMMON_PARAMS} \
		--load-model dummy_model

.PHONY: start-triton-all
start-triton-all:
	tritonserver \
		${TRITON_COMMON_PARAMS} \
		--load-model pipeline_1 \
		--load-model pipeline_2 \
		--load-model pipeline_3 \
		--load-model pipeline_4 \
		--load-model pipeline_5

.PHONY: start-triton-model
start-triton-model:
	tritonserver \
		${TRITON_COMMON_PARAMS} \
		--load-model dummy_model

.PHONY: start-triton-ensemble
start-triton-ensemble:
	tritonserver \
		${TRITON_COMMON_PARAMS} \
		--load-model pipeline_${NUM_MODELS}


.PHONY: start-triton-monolithic
start-triton-monolithic:
	tritonserver \
		${TRITON_COMMON_PARAMS} \
		--load-model dummy_model_${NUM_MODELS}

.PHONY: start-triton-ensemble-with-nsys
start-triton-ensemble-with-nsys:
	nsys profile ${NSYS_COMMON_PARAMS} --output=./outputs/triton-ensemble-ES${NUM_MODELS} \
		tritonserver \
			${TRITON_COMMON_PARAMS} \
			--load-model pipeline_${NUM_MODELS}


.PHONY: start-triton-monolithic-with-nsys
start-triton-monolithic-with-nsys:
	nsys profile ${NSYS_COMMON_PARAMS} --output=./outputs/triton-monolithic-ES${NUM_MODELS} \
		tritonserver \
			${TRITON_COMMON_PARAMS} \
			--load-model dummy_model_${NUM_MODELS}



# ================ Client ================

OUTPUT_DIR := ./outputs/run0
# OUTPUT_DIR := ./outputs/run1

# NUM_REQUESTS := 100
NUM_REQUESTS := 10
# BATCH_SIZE := 1
BATCH_SIZE := 2
# BATCH_SIZE := 4
# BATCH_SIZE := 8
# BATCH_SIZE := 16
CLIENT_COMMON_ARGS := \
	-n ${NUM_REQUESTS} \
	-b ${BATCH_SIZE} \
	--use-shared-memory

.PHONY: call-model
call-model:
	python client.py -p 0 ${CLIENT_COMMON_ARGS} --logdir ${OUTPUT_DIR}

.PHONY: call-pipeline
call-pipeline:
	python client.py -p ${NUM_MODELS} ${CLIENT_COMMON_ARGS} --logdir ${OUTPUT_DIR}
