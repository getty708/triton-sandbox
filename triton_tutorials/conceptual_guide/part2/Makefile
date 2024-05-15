.PHONY: setup-models
setup-models:
	$(MAKE) -C model_repository/text_recognition/1 init

.PHONY: start-server
start-server:
	tritonserver --model-repository=model_repository

.PHONY: perf-analyzer
perf-analyzer:
	perf_analyzer -m text_recognition -b 2 --shape input.1:1,32,100 --concurrency-range 2:16:2 --percentile=95 -f .outputs/perf.csv

.PHONY: clean
clean:
	rm -rf .outputs