VIDEO_PATH :=  /opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4

.PHONY: run-test3
run-test3:
	python3 deepstream_test_3.py -i file://${VIDEO_PATH} --pgie nvinfer \
		-c /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/apps/deepstream-test3/config_infer_primary_peoplenet.txt \
		--no-display


.PHONY: run-test4
run-test4:
	python3 deepstream_test_4.py -i ${VIDEO_PATH} \
		-p /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/apps/deepstream-test4/cfg_kafka.txt \
		--schema-type 0 \
		--no-display


.PHONY: debug-urisrcbin
debug-urisrcbin:
	gst-launch-1.0 nvmultiurisrcbin \
	port=9000 ip-address=localhost \
	batched-push-timeout=33333 max-batch-size=10 \
	drop-pipeline-eos=0 live-source=0  mode=0 \
	uri-list=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 width=1920 height=1080 \
	! fakesink
