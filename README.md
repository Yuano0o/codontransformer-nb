# CodonTransformer N. benthamiana workspace

Reproducible local, CUDA/Linux, and Google Colab workflows for a
CodonTransformer baseline and later *Nicotiana benthamiana* domain adaptation.
The official full training dataset is not downloaded.

The upstream project is
[Adibvafa/CodonTransformer](https://github.com/Adibvafa/CodonTransformer),
pinned here to commit `4a447b01dab860feb81b647ff1ff88ad598517f4`
(CodonTransformer 1.6.7). This repository does not vendor the upstream checkout.

## Repository layout

```text
.github/workflows/             lightweight GitHub CI
configs/
  n_benthamiana_preprocess.json  local stage-one QC
  smoke_test.yaml                two-batch Colab/CUDA smoke test
  finetune_cuda.yaml             full CUDA/Linux starting configuration
notebooks/
  codontransformer_finetune_colab.ipynb
scripts/                       download, baseline, QC, training and verification
tests/                         lightweight unit and portability tests
data/raw/                      local-only NbeBase source data
data/processed/                local-only generated QC/training data
models/pretrained/             local-only Hugging Face snapshot
models/finetuned/              local-only checkpoints/exports
results/                       local-only inference results and logs
upstream/                      local-only official Git clone
```

Only code, documentation, notebooks, tests, and parameter files belong in
GitHub. `.gitignore` excludes the virtual environment, upstream clone, raw
NbeBase files, processed CSV/JSONL, model weights, checkpoints, logs, tokens,
and other credentials.

## Local installation

Use Python 3.11. From the project root:

```bash
git clone https://github.com/Adibvafa/CodonTransformer.git upstream
git -C upstream checkout 4a447b01dab860feb81b647ff1ff88ad598517f4

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -r upstream/requirements.txt
python -m pip install --no-deps -e upstream
python -m pip install 'huggingface_hub>=0.23,<1.0' 'PyYAML>=6,<7'

python scripts/download_pretrained.py \
  --repo-id adibvafa/CodonTransformer \
  --output-dir models/pretrained
```

The local pretrained snapshot is immutable input. Training outputs must go to
`models/finetuned/`, `outputs/`, or another explicitly supplied output path.
Never use `models/pretrained/` as a checkpoint directory.

CodonTransformer 1.6.7 package metadata constrains `setuptools<71`, while newer
PyTorch builds may require a newer setuptools. The verified local environment
keeps the version required by PyTorch; this metadata-only conflict does not
block baseline inference.

## Local CPU/MPS baseline

The baseline script resolves defaults from the project root and runs completely
offline after the model download:

```bash
source .venv/bin/activate

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/baseline_inference.py --device cpu

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/baseline_inference.py --device mps
```

Both CPU and Apple MPS have been verified with the default protein input and
produce the same deterministic DNA sequence. A restricted process may hide MPS;
run the MPS command from a normal macOS terminal.

## N. benthamiana stage-one preprocessing

Keep the untouched NbeBase v1.1 inputs at:

```text
data/raw/n_benthamiana/nbenbase_v1.1/
├── Nbe_v1.1_cds_HC.fa.gz
├── Nbe_v1.1_pep_HC.fa.gz
├── Nbe_v1.1_HC_ann.xlsx.gz
└── Nbe_v1.1.2_HC.fixed.gff3.gz
```

These files are ignored by Git. The deterministic stage-one configuration uses
seed 23 and fixed labels:

```text
organism        = Nicotiana tabacum
source_organism = Nicotiana benthamiana
```

Run:

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
python scripts/preprocess_n_benthamiana.py \
  --config configs/n_benthamiana_preprocess.json
```

Local-only outputs:

```text
data/processed/n_benthamiana/
├── accepted_all_hc.csv
├── rejected.csv
└── qc_summary.json
```

Acceptance requires an A/T/C/G-only CDS, length divisible by three, exact
standard-code protein translation, ATG start, TAA/TAG/TGA stop, and protein
length at most 2045 amino acids. Lowercase soft-masking is evaluated
case-insensitively but preserved. Failed sequences are never repaired; all
applicable reasons are retained in `rejected.csv`.

The current local result is 56,487 accepted and 1,096 rejected records. Do not
convert `accepted_all_hc.csv` directly into final training data. The next stage
must compute CSI/CAI and codon-use metrics, cluster proteins by similarity,
perform cluster-aware splits, and define `all_clean_hc` and `high_csi_hc`.

## CUDA/Linux training configuration

`configs/finetune_cuda.yaml` is the portable full-training starting point;
`configs/smoke_test.yaml` is limited to two batches. Both use relative placeholder
paths and accept CLI overrides:

```bash
python scripts/finetune_codontransformer.py \
  --config configs/smoke_test.yaml \
  --model-dir /path/to/pretrained \
  --dataset-path /path/to/cluster_split_train.jsonl \
  --output-dir /path/to/persistent/run
```

The training JSONL must already use the actual upstream format, one object per
line:

```json
{"idx": 0, "codons": "M_ATG K_AAG __TAA", "organism": 80}
```

No formal training has been run by this repository preparation step.

## Run on Google Colab

Open `notebooks/codontransformer_finetune_colab.ipynb` in a GPU Colab runtime.
Before running it:

1. Create the GitHub repository and edit `REPO_URL` in the notebook.
2. Upload the later cluster-aware training JSONL to
   `MyDrive/CodonTransformer/data/clustered/train.jsonl`, or edit the configured
   Drive-relative path.
3. Select a GPU runtime.

The notebook then:

1. runs `nvidia-smi` and asserts CUDA is available;
2. clones this GitHub repository and the pinned official upstream repository;
3. installs `requirements-colab.txt` without replacing Colab's CUDA PyTorch;
4. mounts Google Drive;
5. downloads `adibvafa/CodonTransformer` from Hugging Face into Colab temporary
   storage;
6. reads the cleaned, cluster-split JSONL directly from Google Drive;
7. runs the two-batch configuration in `configs/smoke_test.yaml`;
8. writes one final `last.ckpt`, Lightning CSV logs, `training.log`,
   `resolved_config.yaml`, and `runtime.json` directly to Google Drive;
9. reloads `last.ckpt`, performs one `Nicotiana tabacum` conditioned inference,
   translates the predicted DNA, and asserts it matches the input protein.

GitHub stores code only. Hugging Face provides the pretrained model. Google
Drive stores training data, checkpoints, configuration snapshots, logs, and
validation results. Colab temporary storage disappears when the runtime
disconnects, so every checkpoint must be written to the Drive run directory.

## Checkpoint reload outside Colab

```bash
python scripts/validate_checkpoint_inference.py \
  --model-dir /path/to/pretrained \
  --checkpoint /path/to/last.ckpt \
  --output /path/to/checkpoint_validation.json \
  --device cuda \
  --organism 'Nicotiana tabacum'
```

The command loads the base snapshot read-only, applies the checkpoint state,
runs inference, and verifies the generated DNA by translation.

## Repository checks

```bash
python scripts/check_repository_hygiene.py
python -m py_compile scripts/*.py tests/*.py
python -m unittest discover -s tests -v
python -m json.tool notebooks/codontransformer_finetune_colab.ipynb > /dev/null
```

GitHub Actions runs the same lightweight checks without downloading models or
biological data.

## References and attribution

- [CodonTransformer project](https://adibvafa.github.io/CodonTransformer/)
- [GitHub upstream](https://github.com/Adibvafa/CodonTransformer)
- [Hugging Face model](https://huggingface.co/adibvafa/CodonTransformer)
- [Official dataset (not downloaded)](https://huggingface.co/datasets/adibvafa/CodonTransformer)
- [Official Colab](https://colab.research.google.com/drive/1WZqXrw49bk3ZDTroY709HwCTabMNOCfL)

See `THIRD_PARTY_NOTICES.md` for the upstream Apache 2.0 attribution.
