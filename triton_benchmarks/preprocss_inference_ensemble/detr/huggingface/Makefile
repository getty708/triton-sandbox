.PHONY: init
init: onnx/model.onnx

# NOTE: Conversion with `--device cuda` makes some difference from the original model.
onnx/model.onnx:
	optimum-cli export onnx \
		--model facebook/detr-resnet-50 \
		--task object-detection \
		--device cuda \
		onnx/ 

.PHONY: clean
clean:
	rm -rf onnx/model.onnx
