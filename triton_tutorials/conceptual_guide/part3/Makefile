MODEL_REPO := ../part2/model_repository
OUTPUT_DIR := .outputs

.PHONY: setup
setup: install

.PHONY: install
install:
	apt-get update && apt-get install -y python3-pip wkhtmltopdf
	python3 -m pip install --upgrade pip
	pip install nvidia-pyindex
	pip3 install triton-model-analyzer


.PHONY: profile
profile:
	mkdir -p ${OUTPUT_DIR}
	model-analyzer profile \
		--model-repository ${MODEL_REPO} \
		--profile-models text_recognition \
		--triton-launch-mode=local \
		--output-model-repository-path ${OUTPUT_DIR}/models \
		-f ${OUTPUT_DIR}/perf.yaml \
		--override-output-model-repository \
		--latency-budget 100 \
		--run-config-search-mode quick
