"""Download the public ModelScope mirror of an MLX checkpoint."""
import argparse
from pathlib import Path
from modelscope import snapshot_download
from run import MODELS

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("size", choices=MODELS)
parser.add_argument("--cache-dir", default=str(Path.home() / ".cache" / "cv-vlm-models"))
args = parser.parse_args()
print(snapshot_download(MODELS[args.size], cache_dir=args.cache_dir,
      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"]))
