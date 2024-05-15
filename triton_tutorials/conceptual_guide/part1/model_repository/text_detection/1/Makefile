
.PHONY: init
init: clean download convert

.PHONY: download
download:
	wget https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz
	tar -xvf frozen_east_text_detection.tar.gz
	pip install -U tensorflow tf2onnx

.PHONY: convert
convert:
	python -m tf2onnx.convert \
		--input frozen_east_text_detection.pb \
		--inputs "input_images:0" \
		--outputs "feature_fusion/Conv_7/Sigmoid:0","feature_fusion/concat_3:0" \
		--output model.onnx


.PHONY: clean
clean:
	rm -rf *.tar.gz *.pb *.onnx
