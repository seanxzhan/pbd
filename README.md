# Implementation of Position Based Dynamics

Generate dataset and visualize:
```bash
python examples/windblown_data_gen.py \
  --obj ~/projects/proxy-asset-gen/data/9423122485_cleaned_proxy.obj \
  --axis direction \
  --train-frames 600 --test-frames 200 \
  --pin-fraction 0.10 \
  --mag-train 0.3 0.3 --mag-test 0.3 0.3 \
  --turbulence-std 0.0 --coherence 1.0 --viz \
  --out data/9423122485_cleaned_proxy
```