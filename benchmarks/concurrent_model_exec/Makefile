MODEL_DIR := ../common/models/simple_cnn/
MODEL_TOOLS_DIR := ${MODEL_DIR}/tools


.PHONY: init
init:
	@echo "Convert pytorch model into ONNX"
	$(MAKE) -C ${MODEL_TOOLS_DIR} clean build
	@echo "Copy generated ONNX files into model_repository/"
	cp -r ${MODEL_TOOLS_DIR}/outputs/simple_cnn.onnx ./model_repository/simple_cnn/1/model.onnx
