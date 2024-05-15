# Part 2: Dynamic Batching & Concurrent Model Execution

Original Tutorial: https://github.com/triton-inference-server/tutorials/tree/main/Conceptual_Guide/Part_2-improving_resource_utilization

## Experiment

### 1-1: Apple M1 Max (CPU)

#### w/o Dynamic Batching & w/o Model Concurrency

```text
Inferences/Second vs. Client p95 Batch Latency
Concurrency: 2, throughput: 7.10843 infer/sec, latency 586378 usec
Concurrency: 4, throughput: 7.66369 infer/sec, latency 1144723 usec
Concurrency: 6, throughput: 8.99751 infer/sec, latency 1390580 usec
Concurrency: 8, throughput: 8.99192 infer/sec, latency 1906325 usec
Concurrency: 10, throughput: 8.77462 infer/sec, latency 2326160 usec
Concurrency: 12, throughput: 6.66465 infer/sec, latency 3753949 usec
Concurrency: 14, throughput: 6.7715 infer/sec, latency 4463870 usec
Concurrency: 16, throughput: 7.55383 infer/sec, latency 4329796 usec
```

## w/ Dynamic Batching (Default Setting)

- The latency was increased due to the time to wait to make a large batch.

```text
Concurrency: 2, throughput: 9.22184 infer/sec, latency 665004 usec
Concurrency: 4, throughput: 9.10787 infer/sec, latency 1358247 usec
Concurrency: 6, throughput: 8.77454 infer/sec, latency 1846793 usec
Concurrency: 8, throughput: 9.33095 infer/sec, latency 1832295 usec
Concurrency: 10, throughput: 6.66104 infer/sec, latency 3782469 usec
Concurrency: 12, throughput: 6.66504 infer/sec, latency 3677621 usec
```

## Dynamic Batching (Default Setting) & Model Concurrency

- Instance group: 2

```text
Inferences/Second vs. Client p95 Batch Latency
Concurrency: 2, throughput: 6.21969 infer/sec, latency 744706 usec
Concurrency: 4, throughput: 5.77718 infer/sec, latency 2417051 usec
```
