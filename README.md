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
  smoke_test_cuda_csi_top10.yaml eight-step csi_top10_hc CUDA smoke test
  finetune_cuda.yaml             full CUDA/Linux starting configuration
  finetune_cuda_csi_top10.yaml   primary CSI-top-10% experiment
  finetune_cuda_csi_top25.yaml   supplementary CSI-top-25% experiment
  evaluate_validation_refined_v2.yaml frozen validation evaluation inputs/rules
  diagnose_validation_decoding.yaml frozen E/K/L/S decoding diagnosis
  check_validation_hybrid_decoding.yaml decoder-only gate before v2
notebooks/
  codontransformer_finetune_colab.ipynb
  codontransformer_finetune_csi_top10_colab.ipynb
  codontransformer_biological_evaluation_colab.ipynb
  codontransformer_validation_evaluation_colab.ipynb
  codontransformer_validation_decoding_colab.ipynb
  codontransformer_validation_hybrid_decoding_colab.ipynb
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

`configs/finetune_cuda_csi_top10.yaml` is the current formal primary experiment.
It uses all 4,524 training records and all 531 validation records, runs on one
T4 with mixed FP16, uses physical batch size 1 and gradient accumulation 8
(effective batch size 8), and allows at most five epochs. Fixed-mask validation
drives one best checkpoint, one last checkpoint, and early stopping with
patience 2. Gradient norm clipping is 1.0. The independent 594-record test split
is recorded in the configuration but is never passed to `Trainer.fit`. The
training entry checks all three configured record counts before fitting.

The `all_clean_hc` and `csi_top25_hc` configurations remain inactive comparison
plans; do not launch them during the primary experiment. Smoke configurations
remain separate. All paths accept CLI overrides:

```bash
python scripts/finetune_codontransformer.py \
  --config configs/finetune_cuda_csi_top10.yaml \
  --model-dir /path/to/pretrained \
  --dataset-path /path/to/csi_top10_hc/train.jsonl \
  --validation-dataset-path /path/to/csi_top10_hc/validation.jsonl \
  --test-dataset-path /path/to/csi_top10_hc/test.jsonl \
  --output-dir /path/to/persistent/formal_run \
  --resume-from-checkpoint /path/to/persistent/formal_run/checkpoints/last.ckpt
```

The training JSONL must already use the actual upstream format, one object per
line:

```json
{"idx": 0, "codons": "M_ATG K_AAG __TAA", "organism": 78}
```

No formal training has been run by this repository preparation step.

## Run the CUDA smoke test on Google Colab

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
7. deterministically selects eight records from `csi_top10_hc` and runs the
   eight-step CUDA configuration in `configs/smoke_test_cuda_csi_top10.yaml`;
8. writes one final `last.ckpt`, Lightning CSV logs, `training.log`,
   `resolved_config.yaml`, and `runtime.json` directly to Google Drive;
9. reloads `last.ckpt`, performs one `Nicotiana tabacum` conditioned inference,
   translates the predicted DNA, and asserts it matches the input protein.

GitHub stores code only. Hugging Face provides the pretrained model. Google
Drive stores training data, checkpoints, configuration snapshots, logs, and
validation results. Colab temporary storage disappears when the runtime
disconnects, so every checkpoint must be written to the Drive run directory.

## Run formal csi_top10_hc fine-tuning on Google Colab

Open `notebooks/codontransformer_finetune_csi_top10_colab.ipynb` in a T4 GPU
runtime. This is a separate entry from the completed eight-step smoke test. It
uses the exact verified official Hugging Face revision
`9744dcc920d813066391fc828d7a590207f148e8` and never writes into the pretrained
model directory or the smoke-test run directory.

Upload all three primary-experiment files to Google Drive:

```text
MyDrive/CodonTransformer/data/stage2/final_csi_cohorts/experiments/csi_top10_hc/
├── train.jsonl       # exactly 4,524 records
├── validation.jsonl  # exactly   531 records
└── test.jsonl        # exactly   594 records
```

All persistent formal outputs go only to:

```text
MyDrive/CodonTransformer/runs/finetune_csi_top10_hc_formal_v1/
```

The notebook checks all split counts before training. If that formal directory
already contains `checkpoints/last.ckpt`, `AUTO_RESUME = True` resumes optimizer,
scheduler, callback, epoch, and global-step state. The training script writes
`best-*.ckpt`, `last.ckpt`, Lightning CSV logs, `training.log`,
`resolved_config.yaml`, `runtime.json`, and `training_result.json` directly to
Drive.

Keep several GB of free Drive capacity: `best-*.ckpt` and `last.ckpt` are full
Lightning checkpoints with model, optimizer, scheduler, callback, epoch, and
global-step state.

After training and model selection by validation loss, the notebook evaluates
the immutable baseline and best fine-tuned checkpoint on all 594 test records
using identical deterministic masks. `test_baseline_vs_finetuned.json` reports
masked-token NLL, perplexity, top-1 accuracy, top-3 accuracy, and signed deltas.
These fixed-mask metrics are a reproducible held-out language-model comparison;
they are not a substitute for downstream biological validation.
The final inference cell also reloads the best checkpoint, generates one DNA
sequence, and verifies its translation. Test data must not be used for early
stopping or checkpoint selection.

## Paired biological evaluation on the independent test split

`notebooks/codontransformer_biological_evaluation_colab.ipynb` performs no
training. It compares the official pretrained baseline, the formal
`best-epoch04-val_loss0.842573.ckpt`, and the true CDS reconstructed directly
from all 594 records in the independent `csi_top10_hc/test.jsonl`.

Upload the local reference file
`data/processed/n_benthamiana/stage2/codon_reference.json` to:

```text
MyDrive/CodonTransformer/data/stage2/codon_reference.json
```

The evaluator uses deterministic synonymous-codon-constrained decoding and
reports translation correctness, sequence validity, CSI, top-10%-reference CAI,
GC and GC3 error versus the true CDS, Jensen-Shannon codon-frequency distance,
rare-codon fraction, exact positional codon match, and short/medium/long protein
strata. Fine-tuned versus baseline comparisons use paired bootstrap confidence
intervals and two-sided Wilcoxon tests with Benjamini-Hochberg correction.

Persistent outputs are written below the completed formal run without modifying
its checkpoints:

```text
MyDrive/CodonTransformer/runs/finetune_csi_top10_hc_formal_v1/
└── biological_evaluation_v1/
    ├── evaluation_manifest.json
    ├── prediction_cache/{baseline,finetuned}_predictions.jsonl
    ├── per_sequence_metrics.csv
    ├── biological_evaluation_summary.json
    └── biological_evaluation_report.md
```

Prediction caches are flushed to Drive every 25 records. Re-running after a
free-Colab interruption resumes only missing predictions; it never starts or
resumes training.

### Refine a completed biological evaluation without inference

After the v1 evaluator has produced `per_sequence_metrics.csv`, the paired
analysis can be refined locally without loading PyTorch, a model, or a
checkpoint:

```bash
python scripts/refine_biological_evaluation.py \
  --per-sequence-csv results/csi_top10_hc_formal_v1/biological_evaluation_v1/per_sequence_metrics.csv \
  --reference-json data/processed/n_benthamiana/stage2/codon_reference.json \
  --output-dir results/csi_top10_hc_formal_v1/biological_evaluation_v1/refined_analysis_v2 \
  --bootstrap-samples 10000 \
  --seed 23
```

The command preserves all v1 files and refuses to overwrite an existing v2
output directory unless `--force` is explicitly supplied. It adds
amino-acid-family-conditional Jensen-Shannon and RSCU distances, paired
bootstrap confidence intervals and BH-adjusted Wilcoxon tests for each protein
length stratum, and codon-family attribution. Outputs are:

```text
refined_analysis_v2/
├── refined_biological_evaluation_report.md
├── refined_biological_evaluation_summary.json
├── per_sequence_refined_metrics.csv
└── synonymous_codon_family_attribution.csv
```

The stricter decision requires at least 99.9% translation correctness and
sequence validity, stable improvements in at least two distinct target
categories (preference, composition, or synonymous-codon distribution), and no
significant target regression overall or within short, medium, or long protein
strata. Because the 594-record test set has now been inspected, subsequent
hyperparameter or checkpoint selection must use validation data or a new
external holdout—not this test report.

### Three-way validation evaluation with frozen refined-v2 rules

`notebooks/codontransformer_validation_evaluation_colab.ipynb` compares true
CDS, the exact official pretrained baseline, and
`best-epoch04-val_loss0.842573.ckpt` on all 531 validation records. It performs
no training and writes to a separate Drive directory:

```text
MyDrive/CodonTransformer/runs/finetune_csi_top10_hc_formal_v1/
└── validation_biological_evaluation_v1/
    ├── evaluation_manifest.json
    ├── prediction_cache/{baseline,finetuned}_predictions.jsonl
    ├── per_sequence_metrics.csv
    ├── biological_evaluation_summary.json
    ├── biological_evaluation_report.md
    └── refined_analysis_v2/
        ├── refined_biological_evaluation_report.md
        ├── refined_biological_evaluation_summary.json
        ├── per_sequence_refined_metrics.csv
        └── synonymous_codon_family_attribution.csv
```

All immutable inputs and rules are recorded in
`configs/evaluate_validation_refined_v2.yaml`: 531 records, the validation and
reference SHA256 values, the pretrained snapshot SHA256, the formal checkpoint
name and size, bootstrap seed/sample count, and the test-derived length bins
`short <= 140 aa`, `medium <= 280.3333333333333 aa`, and `long` above that.
The validation boundaries are never recomputed. Prediction caches are flushed
to Drive every 25 records and resume after a Colab interruption. The final
analysis cell deliberately regenerates only the small refined-v2 tables; it
does not alter checkpoints, model weights, test outputs, or prediction caches.

The validation JSONL must already be present at:

```text
MyDrive/CodonTransformer/data/stage2/final_csi_cohorts/experiments/csi_top10_hc/validation.jsonl
```

Keep this validation analysis for diagnosis and model selection under the
frozen rules. The previously inspected 594-record test results remain a
separate final report and must not be used for further tuning.

### Validation-only E/K/L/S probability and decoding diagnosis

`notebooks/codontransformer_validation_decoding_colab.ipynb` diagnoses whether
the E, K, L, and S family regressions originate in the model probabilities or
are amplified by greedy decoding. It runs no training and accepts only the
SHA256-pinned 531-record validation JSONL. The script refuses a non-validation
dataset and never reads the test split or its prediction caches.

For pretrained and v1 separately, the experiment records the T=1 probability
distribution conditional on the correct synonymous family at every E/K/L/S
position. It reports mean probability by codon, position entropy, normalized
entropy, maximum probability, argmax frequency, argmax concentration, and
argmax HHI. These diagnostics and the paired strategy comparisons are reported
both overall and within the frozen refined-v2 short/medium/long boundaries. It
then evaluates three translation-preserving strategies:

- `greedy`: synonymous-family argmax;
- `temperature_sampling`: family-masked T=0.5, top-p=0.95 sampling;
- `synonymous_family_sampling`: family-masked T=1, top-p=1 sampling.

All three strategies mask non-synonymous tokens before decoding, including the
terminal stop family. Every generated DNA is rejected immediately unless it
passes the same strict sequence-validity checks and translates exactly to the
input protein. Random seeds are derived deterministically from the frozen seed,
model, strategy, and record idx, so resuming an interrupted Colab run cannot
change previously generated or future records.

The runner also computes the full v1 checkpoint SHA256 before creating any
cache and stores it in `evaluation_manifest.json`; later sessions refuse to
reuse caches if the checkpoint content changes.

The fixed experiment definition is
`configs/diagnose_validation_decoding.yaml`. Persistent output is isolated from
the earlier test and validation evaluations:

```text
MyDrive/CodonTransformer/runs/finetune_csi_top10_hc_formal_v1/
└── validation_decoding_diagnostics_v1/
    ├── evaluation_manifest.json
    ├── record_cache/{baseline,finetuned}.jsonl
    ├── family_probability_summary.csv
    ├── per_sequence_strategy_metrics.csv
    ├── decoding_diagnostic_summary.json
    └── decoding_diagnostic_report.md
```

Sampling strategies are compared with their same-model greedy output using
paired validation statistics. Strategy selection must use these validation
results only; the test split remains out of scope.

### Final decoder-only gate before v2 fine-tuning

`notebooks/codontransformer_validation_hybrid_decoding_colab.ipynb` is the last
check before changing the training objective. It requires no GPU and performs
no model download, model/checkpoint load, forward pass, or training. It reads
the frozen validation JSONL/reference plus the completed finetuned
`validation_decoding_diagnostics_v1/record_cache/finetuned.jsonl` and evaluates
four predeclared hybrid candidates:

- S-only T=0.5 sampling;
- S-only T=1 synonymous-family sampling;
- E/K/L/S T=0.5 sampling only at high-entropy, low-maximum-probability positions;
- S at T=1 plus high-entropy-gated E/K/L T=0.5 sampling.

All non-selected positions, start ATG, and terminal stops remain exactly v1
greedy. Every candidate DNA must pass strict sequence validity and translate
exactly to its input protein. The analysis recomputes the full frozen biological
metric set and compares each candidate with v1 greedy using paired bootstrap,
two-sided Wilcoxon, and global BH correction across all candidates, both overall
and within the fixed short/medium/long strata.

The decoder-only gate passes only if a candidate has stable E/K/L/S improvement
in at least one JSD and one RSCU metric, while CSI, CAI, GC/GC3 error, positional
codon match, and full synonymous-family distribution show no significant
regression overall or in any length stratum. If no candidate passes, the report
sets `proceed_to_v2_finetuning: true`.

Persistent cache-only results are written to:

```text
MyDrive/CodonTransformer/runs/finetune_csi_top10_hc_formal_v1/
└── validation_hybrid_decoding_check_v1/
    ├── evaluation_manifest.json
    ├── candidate_predictions.jsonl
    ├── per_sequence_hybrid_metrics.csv
    ├── hybrid_decoding_summary.json
    └── hybrid_decoding_report.md
```

The test split remains prohibited. If the decoder-only gate fails, v2
fine-tuning still uses validation for selection and requires a new external
holdout for final assessment.

### v2 synonymous-family calibrated fine-tuning

The final decoder-only gate found no strategy that improved E/K/L/S JSD and
RSCU without significantly reducing CSI, CAI, or positional codon match.
`scripts/finetune_codontransformer_v2.py` therefore changes the training
objective while leaving the verified v1 workflow untouched. The v2 objective
is:

```text
total loss = MLM loss + 0.2 × synonymous-family distribution JSD
```

The auxiliary term is evaluated only at masked E/K/L/S positions. For each
family it compares the mean predicted within-family probability distribution
with a frozen 50:50 mixture of the whole-species CSI reference and the Top10
proxy-CAI reference in `codon_reference.json`. MLM remains the dominant term.
Train and validation file counts and SHA256 hashes are frozen; no test path is
accepted by the v2 trainer.

The v2 experiment starts again from the same immutable official pretrained
snapshot used by v1. This gives a controlled comparison of objectives and does
not inherit the already concentrated v1 output distribution. It writes to new
directories and never overwrites v1, the CUDA smoke test, or pretrained files:

```text
MyDrive/CodonTransformer/runs/
├── finetune_csi_top10_hc_v2_smoke_v1/
└── finetune_csi_top10_hc_formal_v2/
```

Open `notebooks/codontransformer_finetune_csi_top10_v2_colab.ipynb` with
`RUN_MODE = "smoke"` first. The smoke configuration runs eight train and eight
validation batches and verifies finite MLM/JSD/total losses, changed weights,
checkpoint reload, DNA generation, and exact back-translation. After those
checks pass, change only `RUN_MODE` to `"formal"`. Formal settings retain batch
size 1, gradient accumulation 8, `16-mixed`, five epochs, early stopping, and
best-checkpoint selection by `val_total_loss`. One Drive `last.ckpt` is replaced
every 200 optimizer steps so a later Colab session can resume without restarting
the completed epoch.

The formal run still uses validation only for model selection. Do not read the
existing test split while choosing the v2 objective, regularization weight,
epoch, or checkpoint. After v2 selection is frozen, use a new external holdout
for the final biological claim because the current test split has already been
inspected during v1 development.

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

## License

Unless otherwise noted, the code, configuration, and documentation authored in
this repository are licensed under the [Apache License 2.0](LICENSE).
CodonTransformer itself remains a separate upstream dependency; its attribution
is retained in `THIRD_PARTY_NOTICES.md`. NbenBase source data, locally derived
datasets, pretrained weights, and fine-tuned checkpoints are not distributed by
this repository and are not covered by this repository's license.
