.PHONY: init
init: clean download convert

.PHONY: download
download:
	wget https://www.dropbox.com/sh/j3xmli4di1zuv3s/AABzCC1KGbIRe2wRwa3diWKwa/None-ResNet-None-CTC.pth

.PHONY: convert
convert:
	python convert_text_recognition_model_into_onnx.py

.PHONY: clean
clean:
	rm -rf *.pth *.onnx
