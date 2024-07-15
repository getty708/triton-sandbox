MODEL_DIR := ../../models/simple_cnn
MODEL_TOOLS_DIR := ${MODEL_DIR}/tools
# MODEL_NAME := ensemble_sequential
# MODEL_NAME := ensemble_partially_parallel
MODEL_NAME := ensemble_full_parallel

# ================ Setup ================
ONNX_MODELS := \
	./model_repository/simple_cnn_l1/1/model.onnx \
	./model_repository/simple_cnn_l2/1/model.onnx \
	./model_repository/simple_cnn_l3/1/model.onnx

.PHONY: build
build: ${ONNX_MODELS}

.PHONY: clean
clean: ${ONNX_MODELS}
	rm -rf ${ONNX_MODELS}

./model_repository/simple_cnn_l1/1/model.onnx:
	$(MAKE) -C ${MODEL_TOOLS_DIR} build MODEL_NUM_LAYERS=1
	cp -r ${MODEL_TOOLS_DIR}/outputs/simple_cnn_l1.onnx $@

./model_repository/simple_cnn_l2/1/model.onnx:
	$(MAKE) -C ${MODEL_TOOLS_DIR} build MODEL_NUM_LAYERS=2
	cp -r ${MODEL_TOOLS_DIR}/outputs/simple_cnn_l2.onnx $@

./model_repository/simple_cnn_l3/1/model.onnx:
	$(MAKE) -C ${MODEL_TOOLS_DIR} build MODEL_NUM_LAYERS=3
	cp -r ${MODEL_TOOLS_DIR}/outputs/simple_cnn_l3.onnx $@



# ================ Triton Server ================
SHM_SIZE = 8589934592 # 8GB
TRITON_COMMON_PARAMS = \
	--model-repository=model_repository \
	--backend-config=onnxruntime,default-max-batch-size=8 \
	--pinned-memory-pool-byte-size ${SHM_SIZE} \
	--cuda-memory-pool-byte-size 0:${SHM_SIZE}
# NSYS_CMD := nsys profile -t nvtx,osrt,cuda --force-overwrite=true --output=./outputs/${MODEL_NAME}
NSYS_CMD :=

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
