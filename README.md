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
  n_benthamiana_dataset.yaml     metrics, clustering, and split parameters
  smoke_test.yaml                two-batch Colab/CUDA smoke test
  finetune_cuda.yaml             full CUDA/Linux starting configuration
  finetune_cuda_csi_top10.yaml   primary CSI-top-10% experiment
  finetune_cuda_csi_top25.yaml   supplementary CSI-top-25% experiment
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
convert `accepted_all_hc.csv` directly into training data without the next
cluster-aware stage.

## N. benthamiana metrics, clustering, and leak-resistant splits

The complete second-stage workflow is controlled by
`configs/n_benthamiana_dataset.yaml` and uses seed 23. Install MMseqs2 outside
the Python environment (`brew install mmseqs2` on macOS, or use the Linux
system/package environment available on the CUDA server), then verify
`mmseqs version` works.

Run the stages in order:

```bash
source .venv/bin/activate

python scripts/compute_codon_metrics.py \
  --config configs/n_benthamiana_dataset.yaml

python scripts/cluster_proteins.py \
  --config configs/n_benthamiana_dataset.yaml

python scripts/build_cluster_splits.py \
  --config configs/n_benthamiana_dataset.yaml

python scripts/validate_cluster_splits.py \
  --config configs/n_benthamiana_dataset.yaml
```

The metrics stage computes CSI, CAI, GC1/GC2/GC3/GC3s, rare- and optimal-codon
fractions, mean relative codon weight, synonymous-codon entropy, and L1 codon
usage distance. CSI uses all accepted CDS as the *N. benthamiana* reference.
Without expression measurements, CAI uses the top 10% by CSI as an explicit
proxy reference; it must not be interpreted as expression-grounded CAI.
The cohort names encode their selection unambiguously: `csi_top10_hc` is the
primary experiment, `csi_top25_hc` is the supplementary experiment, and
`all_clean_hc` is full-data domain adaptation. The current CSI thresholds are
`0.80074785` for the top 10% and `0.77651441` for the top 25%.

The strict MMseqs2 configuration uses 30% minimum protein identity, 80%
bidirectional coverage, connected-component clustering, sensitivity 7.5, and
up to 1,000 candidates per query. Whole clusters, never individual records, are
assigned to 80/10/10 train/validation/test splits.

Current local-only outputs:

```text
data/processed/n_benthamiana/stage2/
├── accepted_all_hc_metrics.csv
├── codon_reference.json
├── metrics_summary_csi_cohorts.json
├── cluster_membership_strict.csv
├── cluster_summary_strict.json
└── final_csi_cohorts/
    ├── split_manifest.csv
    ├── dataset_summary.json
    ├── validation_report.json
    └── experiments/
        ├── all_clean_hc/{train,validation,test}.{csv,jsonl}
        ├── csi_top10_hc/{train,validation,test}.{csv,jsonl}
        └── csi_top25_hc/{train,validation,test}.{csv,jsonl}
```

The strict result contains 26,984 clusters. The seeded cluster shuffle balances
both record counts and cluster-size distributions. Train/validation/test counts
are 45,190/5,649/5,648 for `all_clean_hc`, 4,524/531/594 for
`csi_top10_hc`, and 11,318/1,383/1,421 for `csi_top25_hc`. Independent
validation confirms zero cluster intersection across splits and validates every
exported amino-acid/codon token. Generated data and logs remain ignored by Git.

## CUDA/Linux training configuration

`configs/finetune_cuda.yaml` targets the `all_clean_hc` training JSONL;
`configs/finetune_cuda_csi_top10.yaml` targets the primary `csi_top10_hc`
experiment; `configs/finetune_cuda_csi_top25.yaml` targets the supplementary
`csi_top25_hc` experiment;
`configs/smoke_test.yaml` is limited to two batches. All use relative paths and
accept CLI overrides:

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
{"idx": 0, "codons": "M_ATG K_AAG __TAA", "organism": 78}
```

No formal training has been run by this repository preparation step.

## Run on Google Colab

Open `notebooks/codontransformer_finetune_colab.ipynb` in a GPU Colab runtime.
Before running it:

1. Because this code repository is private, add a fine-grained GitHub token with
   read access as a Colab secret named `GITHUB_TOKEN`. Never paste or save the
   token in the notebook, GitHub, or Google Drive.
2. Upload the selected cluster-aware training JSONL to
   `MyDrive/CodonTransformer/data/stage2/final_csi_cohorts/experiments/csi_top10_hc/train.jsonl`,
   or edit `DRIVE_TRAINING_DATA` to use `csi_top25_hc` or `all_clean_hc`.
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
