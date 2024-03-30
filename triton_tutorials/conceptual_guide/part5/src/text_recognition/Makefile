.PHONY: init
init: clean download convert

.PHONY: download
download: None-ResNet-None-CTC.pth

.PHONY: convert
convert: model.onnx

.PHONY: clean
clean:
	rm -rf *.pth *.onnx

None-ResNet-None-CTC.pth:
	wget https://www.dropbox.com/sh/j3xmli4di1zuv3s/AABzCC1KGbIRe2wRwa3diWKwa/None-ResNet-None-CTC.pth

model.onnx:
	python convert_text_recognition_model_into_onnx.py