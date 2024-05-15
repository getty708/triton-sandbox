.PHONY: init
init:
	$(MAKE) -C ./src/text_detection/ init
	$(MAKE) -C ./src/text_recognition init
	pip install torchvision opencv-python-headless

.PHONY: init-model-repository
init-model-repository:
	mkdir -p ./model_repository/text_detection/1
	cp ./src/text_detection/model.onnx ./model_repository/text_detection/1/
	mkdir -p ./model_repository/text_recognition/1
	cp ./src/text_recognition/model.onnx ./model_repository/text_recognition/1/


.PHONY: start-server
start-server:
	tritonserver --model-repository=model_repository


.PHONY: clean
clean:
	rm -rf *.png
