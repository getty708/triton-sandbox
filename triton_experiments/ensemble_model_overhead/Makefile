# ================ Docker ================
# Docker for GPU
.PHONY: docker-run-gpu
docker-run-gpu:
	docker compose up -d



# ================ Triton Server ================
PIPELINE_STEPS := 1
SHM_SIZE = 8589934592 # 8GB
TRITON_COMMON_PARAMS = \
	--model-repository=model_repository \
	--backend-config=onnxruntime,default-max-batch-size=8 \
	--pinned-memory-pool-byte-size ${SHM_SIZE} \
	--cuda-memory-pool-byte-size 0:${SHM_SIZE}
NSYS_COMMON_PARAMS = -t nvtx,osrt,cuda --force-overwrite=true

.PHONY: start-triton
start-triton:
	tritonserver ${TRITON_COMMON_PARAMS}

.PHONY: start-triton-monolithic
start-triton-monolithic:
	tritonserver \
		${TRITON_COMMON_PARAMS} \
		--model-control-mode=explicit \
		--load-model monolithic_${PIPELINE_STEPS}

.PHONY: start-triton-ensemble
start-triton-ensemble:
	tritonserver \
		${TRITON_COMMON_PARAMS} \
		--model-control-mode=explicit \
		--load-model ensemble_${PIPELINE_STEPS}

.PHONY: start-triton-monolithic-with-nsys
start-triton-monolithic-with-nsys:
	# nsys profile ${NSYS_COMMON_PARAMS} --output=./outputs/triton-monolithic-ES${PIPELINE_STEPS} \
	tritonserver \
		${TRITON_COMMON_PARAMS} \
		--model-control-mode=explicit \
		--load-model monolithic_${PIPELINE_STEPS}

.PHONY: start-triton-ensemble-with-nsys
start-triton-ensemble-with-nsys:
	nsys profile ${NSYS_COMMON_PARAMS} --output=./outputs/triton-ensemble-ES${PIPELINE_STEPS} \
		tritonserver \
			${TRITON_COMMON_PARAMS} \
			--model-control-mode=explicit \
			--load-model ensemble_${PIPELINE_STEPS}


# ================ Client ================

OUTPUT_DIR := ./outputs/run0
NUM_REQUESTS := 10
BATCH_SIZE := 2
CLIENT_COMMON_ARGS := \
	-n ${NUM_REQUESTS} \
	-b ${BATCH_SIZE} \
	--logdir ${OUTPUT_DIR} \
	--use-shared-memory

.PHONY: call-monolithic
call-monolithic:
	python client.py ${CLIENT_COMMON_ARGS} \
		--pipeline-architecture monolithic \
		-s ${PIPELINE_STEPS}


.PHONY: call-ensemble
call-ensemble:
	python client.py ${CLIENT_COMMON_ARGS} \
		--pipeline-architecture ensemble \
		-s ${PIPELINE_STEPS}
