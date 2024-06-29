LOG_LEBEL := ERROR

COMMON_ENVS := \
	GST_DEBUG_DUMP_DOT_DIR=./outputs \
	GST_DEBUG_BIN_TO_DOT_FILE=./outputs/pipeline.dot \
	GST_DEBUG=${LOG_LEBEL}
# NSYS_COMMON_PARAMS = -t nvtx,osrt,cuda,cudnn,cublas --force-overwrite=true
NSYS_CMD := nsys profile  -t nvtx,osrt,cuda,cudnn,cublas --backtrace=dwarf --force-overwrite=true
# --output=./outputs/simple_cnn

.PHONY: init
init:
	-mkdir -p outputs

.PHONY: run
run: init
	${COMMON_ENVS} python run_pipeline.py -b 8

.PHONY: run-with-nsys
run-with-nsys: init
	${COMMON_ENVS} ${NSYS_CMD} python run_pipeline.py -b 8


.PHONY: plot
plot:
	dot -Tpdf ./outputs/pipeline.dot -o ./outputs/pipeline.pdf


.PHONY: build
build: build-onnx  build-trt


MODEL_DIR := ../../benchmarks/common/models/simple_cnn
MODEL_TOOLS_DIR := ${MODEL_DIR}/tools
CNN_MODELS := \
	.cache/simple_cnn_l1.onnx \
	.cache/simple_cnn_l2.onnx \
	.cache/simple_cnn_l3.onnx

.PHONY: build-onnx
build-onnx: ${CNN_MODELS}


.cache/simple_cnn_l1.onnx:
	mkdir -p .cache
	$(MAKE) -C ${MODEL_TOOLS_DIR} build-torchscript MODEL_NUM_LAYERS=1
	cp -r ${MODEL_TOOLS_DIR}/outputs/simple_cnn_l1.onnx $@

.cache/simple_cnn_l2.onnx:
	mkdir -p .cache
	$(MAKE) -C ${MODEL_TOOLS_DIR} build-torchscript MODEL_NUM_LAYERS=2
	cp -r ${MODEL_TOOLS_DIR}/outputs/simple_cnn_l2.onnx $@

.cache/simple_cnn_l3.onnx:
	mkdir -p .cache
	$(MAKE) -C ${MODEL_TOOLS_DIR} build-torchscript MODEL_NUM_LAYERS=3
	cp -r ${MODEL_TOOLS_DIR}/outputs/simple_cnn_l3.onnx $@


TRTEXEC_CMD := /usr/src/tensorrt/bin/trtexec
TARGET_TRT_MODELS := \
	.cache/simple_cnn_l1.trt \
	.cache/simple_cnn_l2.trt \
	.cache/simple_cnn_l3.trt

.PHONY: build-trt
build-trt: ${TARGET_TRT_MODELS}

.cache/simple_cnn_%.trt: .cache/simple_cnn_%.onnx
	${TRTEXEC_CMD} --onnx=$< --saveEngine=$@
