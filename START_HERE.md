# Getting started with SoproSynth

The two archives are independent Git repository starters. This is a **source MVP**,
not the finished native-ONNX audio synthesizer.

```bash
cd soprosynth
python -m pip install -e '.[native,dev]'
pytest -q
python -m soprosynth speak 'Hello from Sopro' --ref path/to/reference.wav --out hello.wav
```

The native command above calls the upstream PyTorch Sopro backend and downloads weights.
For ONNX, use the sibling catalog repository:

```bash
cd ../sopro-onnx-bundles
python tools/refresh_catalog.py
python tools/verify_catalog.py catalog/bundles.json
python tools/download_bundle.py --profile wasm-uint8 --output ../onnx-model
cd ../soprosynth
python -m pip install -e '.[cpu]'
python -m soprosynth inspect ../onnx-model --profile wasm-uint8
```

The `inspect` command registers the embedded OnnxVoice Sopro adapter and enumerates
graphs without executing them. Call `graph_specs` or `run_graph` with real tensors to
exercise ONNX Runtime. End-to-end ONNX text-to-audio is **not implemented yet**.

To activate Git-tag based dynamic versioning after unzipping:

```bash
git init
# configure git user identity, then:
git add .
git commit -m 'Initial SoproSynth MVP'
git tag v0.1.0
python -m build
```
