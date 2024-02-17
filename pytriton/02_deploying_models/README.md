# Deploying Models

- [Triton Initialization](https://triton-inference-server.github.io/pytriton/0.5.1/initialization/)
- [Binding Model to Triton](https://triton-inference-server.github.io/pytriton/0.5.1/binding_models/)
- [Binding Configuration](https://triton-inference-server.github.io/pytriton/0.5.1/binding_configuration/)
- [Triton Remote Mode](https://triton-inference-server.github.io/pytriton/0.5.1/remote_triton/)

## Init Triton

### Blocking Mode

Use `triton.serve()` to run a server in blocking mode.

```bash
python run_01_init_triton.py blocking-mode
```

### Background Mode

`triton.run()` と `triton.stop()` の組み合わせで，Triton をバックグラウンド (サブプロセス) で動かせる．

```bash
python run_01_init_triton.py background-mode
```
