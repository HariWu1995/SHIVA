from pathlib import Path
from glob import glob


samples_dir = Path(__file__).resolve().parents[5] / "_samples" / "camera"
EXAMPLE_MAP = []

for ex in  ['garden-4_*.jpg', 'telebooth-2_*.jpg', 
            'vgg-lab-4_*.png', 'backyard-7_*.jpg']:
    ex_mv = sorted(glob(str(samples_dir / ex)))
    EXAMPLE_MAP.append((ex_mv[0], ex_mv))

