.PHONY: install
install:
	cp -r ./models/tools/simple_cnn.onnx ./model_repository/simple_cnn/1/model.onnx

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

.PHONY: start-triton-with-nsys
start-triton-with-nsys:
	nsys profile ${NSYS_COMMON_PARAMS} --output=./outputs/triton-ensemble-ES${PIPELINE_STEPS} \
		tritonserver ${TRITON_COMMON_PARAMS}

# ================ Client ================
OUTPUT_DIR := ./outputs/run0
NUM_REQUESTS := 10
BATCH_SIZE := 2
CLIENT_COMMON_ARGS := \
	-n ${NUM_REQUESTS} \
	-b ${BATCH_SIZE} \
	--logdir ${OUTPUT_DIR} \
	--use-shared-memory

.PHONY: call-monolithic-cnn
call-monolithic-cnn:
	python client_cnn.py ${CLIENT_COMMON_ARGS} \
		--pipeline-architecture	monolithic \
		--pipeline-step-size 1


.PHONY: call-ensemble-cnn
call-ensemble-cnn:
	python client_cnn.py ${CLIENT_COMMON_ARGS} \
		--pipeline-architecture	ensemble \
		--pipeline-step-size 4
