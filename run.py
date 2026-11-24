"""VSR zero-shot evaluation. Run prepare once, then infer for each checkpoint."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import random
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
MODELS = {
    "256M": "mlx-community/SmolVLM-256M-Instruct-4bit",
    "500M": "mlx-community/SmolVLM-500M-Instruct-4bit",
    "2B": "mlx-community/SmolVLM-Instruct-4bit",
}
SOURCE = "https://raw.githubusercontent.com/cambridgeltl/visual-spatial-reasoning/master/data/splits/zeroshot/test.jsonl"
PROMPT = 'Based only on the image, is this statement true or false? Reply with exactly True or False. Statement: "{caption}"'
RELATIONS = ('above', 'under', 'in front of', 'behind', 'next to', 'facing')

def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def download(url, path):
    if path.exists():
        return
    for attempt in range(3):
        try:
            with urlopen(url.replace("http://", "https://", 1), timeout=30) as response:
                path.write_bytes(response.read())
            return
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)

def prepare():
    from PIL import Image
    annotation = ROOT / "vsr-test.jsonl"
    download(SOURCE, annotation)
    raw = annotation.read_bytes()
    rng = random.Random(42)
    records = [dict(json.loads(line), id=index) for index, line in enumerate(raw.splitlines())]
    chosen = []
    for relation in RELATIONS:
        for label in (0, 1):
            group = [row for row in records if row['relation'] == relation and row['label'] == label]
            chosen += rng.sample(group, 5)
    chosen.sort(key=lambda row: row['id'])
    image_dir = ROOT / "images"
    image_dir.mkdir(exist_ok=True)
    def fetch(row):
        path = image_dir / Path(row['image']).name
        download(row['image_link'].replace('http://images.cocodataset.org/',
                 'https://s3.amazonaws.com/images.cocodataset.org/'), path)
        with Image.open(path) as image:
            image.verify()
        return dict(row, image_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    save(ROOT / "sample.json", dict(source=SOURCE, sha256=hashlib.sha256(raw).hexdigest(),
                                   total=len(records), seed=42, records=chosen))
    # COCO images can occur in several captions. Download each file only once.
    unique = {row['image']: row for row in chosen}
    with ThreadPoolExecutor(max_workers=10) as pool:
        hashes = {row['image']: row['image_sha256'] for row in pool.map(fetch, unique.values())}
    for row in chosen:
        row['image_sha256'] = hashes[row['image']]
    save(ROOT / "sample.json", dict(source=SOURCE, sha256=hashlib.sha256(raw).hexdigest(),
                                   total=len(records), seed=42, records=chosen))
    print(f"Prepared {len(chosen)} examples and {len(unique)} images.")

def parse(text):
    match = re.fullmatch(r'\s*(true|false)[.!]?\s*', text, re.I)
    return None if not match else int(match.group(1).lower() == 'true')

def parse_with_yes_no(text):
    result = parse(text)
    if result is not None:
        return result
    match = re.fullmatch(r'\s*(yes|no)[.!]?\s*', text, re.I)
    return None if not match else int(match.group(1).lower() == 'yes')

def infer(name, limit, model_dir, output_dir):
    from huggingface_hub import snapshot_download
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
    path = str(Path(model_dir).resolve()) if model_dir else snapshot_download(
        MODELS[name], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
    weights = {}
    for file in sorted(Path(path).glob('*.safetensors')):
        with file.open('rb') as stream:
            weights[file.name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    if not weights:
        raise ValueError(f"No model weights under {path}")
    model, processor = load(path)
    config = load_config(path)
    manifest = json.loads((ROOT / "sample.json").read_text())
    records = manifest["records"][:limit] if limit else manifest["records"]
    output = Path(output_dir) if output_dir else ROOT / ("smoke" if limit else "results")
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"{name}.jsonl"
    if target.exists():
        raise FileExistsError(f"Move {target} first to preserve the previous run.")
    save(output / f"{name}-meta.json", {
        "model": MODELS[name], "revision": None if model_dir else Path(path).name, "weights_sha256": weights,
        "sample_sha256": hashlib.sha256((ROOT / "sample.json").read_bytes()).hexdigest(),
        "prompt": PROMPT, "max_tokens": 8, "temperature": 0, "python": platform.python_version(),
        "device": platform.platform(), "packages": {name: importlib.metadata.version(name)
        for name in ("mlx", "mlx-vlm", "transformers", "huggingface-hub", "pillow")},
    })
    with target.open("x", encoding="utf-8") as stream:
        for index, row in enumerate(records, 1):
            path = ROOT / "images" / Path(row['image']).name
            if hashlib.sha256(path.read_bytes()).hexdigest() != row['image_sha256']:
                raise ValueError(f"Image checksum mismatch: {path}")
            prompt = PROMPT.format(caption=row['caption'])
            formatted = apply_chat_template(processor, config, prompt, num_images=1)
            started = time.perf_counter()
            result = generate(model, processor, formatted, [str(path)], max_tokens=8, temperature=0, verbose=False)
            seconds = time.perf_counter() - started
            prediction = parse(result.text)
            item = {key: row[key] for key in ('id', 'image', 'image_link', 'caption', 'relation', 'label', 'image_sha256')}
            item.update(model=name, response=result.text, prediction=prediction, correct=prediction == row['label'],
                        seconds=seconds, generation_tokens=result.generation_tokens)
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            stream.flush()
            if index == 1 or index % 10 == 0:
                print(f"{name} {index}/{len(records)}: {result.text!r} ({seconds:.2f}s)", flush=True)

def score(output_dir=None):
    output = Path(output_dir) if output_dir else ROOT / "results"
    manifest = json.loads((ROOT / 'sample.json').read_text())['records']
    expected = [row['id'] for row in manifest]
    all_rows, summaries = [], []
    for name in MODELS:
        path = output / f"{name}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if [row['id'] for row in rows] != expected:
            raise ValueError(f"{name}: incomplete or mismatched sample")
        for row, sample in zip(rows, manifest, strict=True):
            if any(row[k] != sample[k] for k in ('image', 'caption', 'relation', 'label', 'image_sha256')):
                raise ValueError(f"{name}: data mismatch for sample {row['id']}")
            if row['prediction'] != parse(row['response']) or row['correct'] != (row['prediction'] == row['label']):
                raise ValueError(f"{name}: scoring mismatch for sample {row['id']}")
            row['relaxed_prediction'] = parse_with_yes_no(row['response'])
            row['relaxed_correct'] = row['relaxed_prediction'] == row['label']
        all_rows.extend(rows)
        for relation in ('all', *RELATIONS):
            items = rows if relation == 'all' else [row for row in rows if row['relation'] == relation]
            valid = [row for row in items if row['prediction'] is not None]
            correct = sum(row['correct'] for row in items)
            confusion = {f'y{y}_p{p}': sum(row['label'] == y and row['prediction'] == p for row in items)
                         for y in (0, 1) for p in (0, 1, None)}
            summaries.append(dict(model=name, relation=relation, n=len(items), correct=correct,
                                  accuracy=correct / len(items), invalid=len(items)-len(valid),
                                  relaxed_correct=sum(row['relaxed_correct'] for row in items),
                                  relaxed_accuracy=sum(row['relaxed_correct'] for row in items) / len(items),
                                  relaxed_invalid=sum(row['relaxed_prediction'] is None for row in items),
                                  true_outputs=sum(row['prediction'] == 1 for row in items),
                                  median_seconds=statistics.median(row['seconds'] for row in items),
                                  max_seconds=max(row['seconds'] for row in items),
                                  mean_seconds=statistics.mean(row['seconds'] for row in items),
                                  **confusion))
    from itertools import combinations
    by_model = {model: [row['correct'] for row in all_rows if row['model'] == model] for model in MODELS}
    paired = []
    for a, b in combinations(MODELS, 2):
        pairs = list(zip(by_model[a], by_model[b], strict=True))
        paired.append(dict(model_a=a, model_b=b, both_correct=sum(x and y for x,y in pairs),
            only_a=sum(x and not y for x,y in pairs), only_b=sum(y and not x for x,y in pairs),
            both_wrong=sum(not x and not y for x,y in pairs)))
    for name, rows in [('predictions', all_rows), ('summary', summaries), ('paired', paired)]:
        with (output / f'{name}.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for row in summaries:
        if row['relation'] == 'all':
            print(row)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    subs.add_parser('prepare')
    scoring = subs.add_parser('score')
    scoring.add_argument('--output-dir')
    inference = subs.add_parser('infer')
    inference.add_argument('model', choices=MODELS)
    inference.add_argument('--limit', type=int)
    inference.add_argument('--output-dir')
    inference.add_argument('--model-dir', help='Use an existing local MLX checkpoint directory')
    args = parser.parse_args()
    if args.command == 'infer':
        infer(args.model, args.limit, args.model_dir, args.output_dir)
    elif args.command == 'score':
        score(args.output_dir)
    else:
        prepare()
