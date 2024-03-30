

.PHONY: init
init: setup-text-detection-model

.PHONY: setup-models
setup-models:
	$(MAKE) -C model_repository/text_detection/1 init
	$(MAKE) -C model_repository/text_recognition/1 init

.PHONY: start-server
start-server:
	tritonserver --model-repository=model_repository

.PHONY: init-client
init-client:
	pip install tritonclient[http] opencv-python-headless


.PHONY: clean
clean:
	rm -rf .outputs