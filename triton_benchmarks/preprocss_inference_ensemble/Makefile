
.PHONY: init-triton
init-triton:
	$(MAKE) -C ./detr/huggingface init
	cp detr/huggingface/onnx/model.onnx model_repository/detr_onnx/1/model.onnx

.PHONY: install-docker-compose
install-docekr-compose:
	DOCKER_CONFIG=${DOCKER_CONFIG:-$HOME/.docker}
	mkdir -p $DOCKER_CONFIG/cli-plugins
	curl -SL https://github.com/docker/compose/releases/download/v2.4.1/docker-compose-linux-x86_64 -o $DOCKER_CONFIG/cli-plugins/docker-compose
	chmod +x $DOCKER_CONFIG/cli-plugins/docker-compose
	sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

# ================ Triton Server ================


SHM_SIZE = 4294967296 # 4GB

TRITON_COMMON_PARAMS = \
	--model-repository=model_repository \
	--model-control-mode=explicit \
	--backend-config=onnxruntime,default-max-batch-size=8 \
	--pinned-memory-pool-byte-size ${SHM_SIZE} \
	--cuda-memory-pool-byte-size 0:${SHM_SIZE}

.PHONY: start-triton
start-triton:
	tritonserver \
		${TRITON_COMMON_PARAMS} \
		--load-model detr_onnx

.PHONY: start-triton-p1
start-triton-p1:
	tritonserver \
		${TRITON_COMMON_PARAMS} \
		--load-model all_in_one_detr

.PHONY: start-triton-p10
start-triton-p10:
	tritonserver \
		${TRITON_COMMON_PARAMS} \
		--load-model pattern_1


.PHONY: start-triton-p2
start-triton-p2:
	tritonserver \
		${TRITON_COMMON_PARAMS} \
		--load-model pattern_2

.PHONY: start-triton-p3
start-triton-p3:
	tritonserver \
		${TRITON_COMMON_PARAMS} \
		--load-model pattern_3


# ================ Client ================

OUTPUT_DIR := ./outputs/run0
# OUTPUT_SUBDIR := ${OUTPUT_DIR}/run0

# CLIENT_COMMON_ARGS := -n 5 -b 8
# CLIENT_COMMON_ARGS := -n 50 -b 8
# CLIENT_COMMON_ARGS := -n 50 -b 1
# CLIENT_COMMON_ARGS := -n 50 -b 4
# CLIENT_COMMON_ARGS := -n 100 -b 8
CLIENT_COMMON_ARGS := -n 100 -b 1


.PHONY: call-p0
call-p0:
	python client.py -p 0 ${CLIENT_COMMON_ARGS} --logdir ${OUTPUT_DIR}

.PHONY: call-p1
call-p1:
	python client.py -p 1 ${CLIENT_COMMON_ARGS} --logdir ${OUTPUT_DIR}

.PHONY: call-p10
call-p10:
	python client.py -p 10 ${CLIENT_COMMON_ARGS} --logdir ${OUTPUT_DIR}

.PHONY: call-p2
call-p2:
	python client.py -p 2 ${CLIENT_COMMON_ARGS} --logdir ${OUTPUT_DIR}

.PHONY: call-p3
call-p3:
	python client.py -p 3 ${CLIENT_COMMON_ARGS} --logdir ${OUTPUT_DIR}

.PHONY: perf-analyzer
perf-analyzer:
	perf_analyzer -m pattern_3 -b 1 --shape image:1,3,1080,1920 --concurrency-range 2:4:2 --percentile=95 -f .outputs/perf.csv
