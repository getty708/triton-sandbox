MODEL_DIR := ../common/models/simple_cnn/
MODEL_TOOLS_DIR := ${MODEL_DIR}/tools
MODEL_NAME := default_model


.PHONY: init
init:
	@echo "Convert pytorch model into ONNX"
	$(MAKE) -C ${MODEL_TOOLS_DIR} clean build
	@echo "Copy generated ONNX files into model_repository/"
	cp -r ${MODEL_TOOLS_DIR}/outputs/simple_cnn.onnx ./model_repository/simple_cnn/1/model.onnx


# ================ Triton Server ================
SHM_SIZE = 8589934592 # 8GB
TRITON_COMMON_PARAMS = \
	--model-repository=model_repository \
	--backend-config=onnxruntime,default-max-batch-size=8 \
	--pinned-memory-pool-byte-size ${SHM_SIZE} \
	--cuda-memory-pool-byte-size 0:${SHM_SIZE}
NSYS_COMMON_PARAMS = -t nvtx,osrt,cuda --force-overwrite=true
NSYS_CMD := nsys profile ${NSYS_COMMON_PARAMS} --output=./outputs/${MODEL_NAME}
# NSYS_CMD :=

.PHONY: run
run:
	${NSYS_CMD} tritonserver ${TRITON_COMMON_PARAMS}


# ================ Client ================
OUTPUT_DIR := ./outputs/run0/${MODEL_NAME}
BATCH_SIZE := 8
CLIENT_COMMON_ARGS := \
	-b ${BATCH_SIZE}


.PHONY: call-ensemble-model
call-ensemble-model:
	python client.py ${CLIENT_COMMON_ARGS} \
		--pipeline-name ${MODEL_NAME} \
		--logdir ${OUTPUT_DIR} \

.PHONY: call-ensemble-single-onnx
call-ensemble-single-onnx:
	$(MAKE) call-ensemble-model MODEL_NAME=ensemble_single_onnx

.PHONY: call-ensemble-triple-onnx
call-ensemble-triple-onnx:
	$(MAKE) call-ensemble-model MODEL_NAME=ensemble_triple_onnx
