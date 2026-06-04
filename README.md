# Implementation of Position Based Dynamics

## Installation

```bash
conda env create -f env_mac.yml
pip install -e .
```

## Run

Examples
```bash
# Cloth hanging 
python examples/hanging_cloth.py

# Blowing wind at a cloth
python examples/windblown.py

# Cloth falling on a ball (collision by SDF)
python examples/draped_on_ball.py

# Cloth falling on a mesh (collision by mesh)
python examples/draped_on_mesh.py --obj data/spot_simplified.obj --convex-hull
```

## Test Cases

```bash
python -m pytest
```