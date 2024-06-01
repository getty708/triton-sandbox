OUTPUT_DIR := ./outputs
ONNX_FILENAME := simple_cnn.onnx

.PHONY: build
build:
	python convert_to_onnx.py --onnx-path ${OUTPUT_DIR}/${ONNX_FILENAME}

.PHONY: clean
clean:
	rm -rf *.onnx
