.PHONY: init
init:
	@echo "Convert pytorch model into ONNX"
	$(MAKE) -C ./models/tools clean build
	@echo "Copy generated ONNX files into model_repository/"
	cp -r ./models/tools/simple_cnn.onnx ./model_repository/simple_cnn/1/model.onnx
	cp -r ./models/tools/simple_transformer_optimized.onnx ./model_repository/simple_transformer/1/simple_transformer_optimized.onnx

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

.PHONY: start-triton-transformer
start-triton-transformer:
	tritonserver ${TRITON_COMMON_PARAMS} \
		--model-control-mode=explicit \
		--load-model simple_transformer

.PHONY: start-triton-with-nsys
start-triton-with-nsys:
	nsys profile ${NSYS_COMMON_PARAMS} --output=./outputs/triton-ensemble-ES${PIPELINE_STEPS} \
		tritonserver ${TRITON_COMMON_PARAMS}

# ================ Client ================
OUTPUT_DIR := ./outputs/run0
NUM_REQUESTS := 10
BATCH_SIZE := 8
CLIENT_COMMON_ARGS := \
	-n ${NUM_REQUESTS} \
	-b ${BATCH_SIZE} \
	--logdir ${OUTPUT_DIR} \
	--use-shared-memory

# CNN
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


# Transformer
.PHONY: call-monolithic-transformer
call-monolithic-transformer:
	python client_transformer.py ${CLIENT_COMMON_ARGS} \
		--pipeline-architecture	monolithic \
		--pipeline-step-size 1


# Transformer
.PHONY: call-ensemble-transformer
call-ensemble-transformer:
	python client_transformer.py ${CLIENT_COMMON_ARGS} \
		--pipeline-architecture	ensemble \
		--pipeline-step-size 4
