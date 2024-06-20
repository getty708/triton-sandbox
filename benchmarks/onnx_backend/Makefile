.PHONY: init
init: build-simple-cnn build-simple-transformer

.PHONY: build-simple-cnn
build-simple-cnn:
	$(MAKE) -C ../common/models/simple_cnn/tools build MODEL_NUM_LAYERS=1
	cp -r ../common/models/simple_cnn/tools/outputs/simple_cnn_l3.onnx ./model_repository/simple_cnn/1/model.onnx
	cp -r ../common/models/simple_cnn/tools/outputs/simple_cnn_l3.onnx ./model_repository/simple_cnn_trt/1/model.onnx

.PHONY: build-simple-transformer
build-simple-transformer:
	$(MAKE) -C ../common/models/simple_transformer/tools build
	cp -r ../common/models/simple_transformer/tools/outputs/simple_transformer.onnx ./model_repository/simple_transformer/1/model.onnx
	cp -r ../common/models/simple_transformer/tools/outputs/simple_transformer.onnx ./model_repository/simple_transformer_trt/1/model.onnx



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

# Transformer (ORT-TRT Optimization)
.PHONY: call-monolithic-transformer-trt
call-monolithic-transformer-trt:
	python client_transformer.py ${CLIENT_COMMON_ARGS} \
		--pipeline-architecture	monolithic \
		--pipeline-step-size 1 --trt

# Transformer
.PHONY: call-ensemble-transformer
call-ensemble-transformer:
	python client_transformer.py ${CLIENT_COMMON_ARGS} \
		--pipeline-architecture	ensemble \
		--pipeline-step-size 4


OUTPUT_SHM_SIZE = 214748364

# Performance Analyzer
.PHONY: perf-transformer
perf-transformer:
	perf_analyzer \
		-m simple_transformer --shape input:${BATCH_SIZE},32,2048 \
		--shared-memory cuda --output-shared-memory-size ${OUTPUT_SHM_SIZE} \
		--percentile=95 --concurrency-range 1:4
