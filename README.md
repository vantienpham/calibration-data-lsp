# When Calibration Data Become Training Data

Code and experiment records for the paper

> Van Tien Pham. **When Calibration Data Become Training Data: Specialization and
> Self-Generated Text in Learned Low-Rank Compression of Large Language Models.**
> 2026.

Low-rank compression of a language model is guided by a small calibration set. When
the compression is learned by distillation, as in learned subspace projections (LSP;
[Bini et al., 2026](https://arxiv.org/abs/2609.40127)), the calibration set becomes
the training set. The paper compresses OPT-1.3B, Llama-3.2-1B and Qwen3-1.7B by 30%
and 50%, and Llama-3.1-8B by 30%, with LSP calibrated on six sources at a matched
budget (WikiText-2, C4, SlimPajama, Alpaca, and two kinds of text sampled from the
dense model), and measures what each source preserves. In short:

- Before training, the calibration source changes perplexity up to a hundredfold.
  Training removes most of this gap but turns it into specialization: each source
  best preserves its own text, WikiText-2 gives the least general models, and
  zero-shot accuracy favors instruction data.
- Calibrating on exact samples of the dense model makes the distillation objective
  the sequence-level divergence between the two models, but long samples drift.
  Independent self-generated documents give the most uniformly faithful compressed
  models, without any external data.
- The domain of the initialization survives training, more distinct calibration
  windows keep helping at a fixed number of steps, and on Llama-3.2 the
  beginning-of-sequence token of the calibration windows decides which input formats
  the compressed model handles.

This repository contains the reimplementation of LSP, the definition of every job
behind the paper, the records of all 182 runs, and the scripts that turn the records
into every table, figure and number of the paper.

## Contents

| Path | What it holds |
|---|---|
| `src/fsc/` | the library: compressible units, calibration data, Gram matrices, whitened initialization, rank allocation, projectors and their exact merge into low-rank factors, distillation, evaluation, self-generated text, pinned Hub revisions |
| `scripts/` | entry points: one compression run (`compress.py`), self-generated calibration (`generate_calib.py`), the experiment plans and their runner (`plans.py`, `submit.py`), reporting (`aggregate.py`, `make_report.py`, `claims.py`) |
| `records/` | the raw measurements: one directory per run, plus the per-source statistics and the subspace overlaps |
| `results/` | everything derived from `records/`: run tables, the LaTeX tables and PDF figures of the paper, and the numbers its text quotes |
| `slurm/` | environment setup and the Slurm job runner |
| `tests/` | CPU tests: exact merging, initialization identities, gradients, plans against records, results against records |

## Installation

Requirements: Linux, [uv](https://docs.astral.sh/uv/), and for GPU runs an NVIDIA GPU
with bfloat16 support and a driver >= 525.

```bash
git clone https://github.com/vantienpham/calibration-data-lsp.git
cd calibration-data-lsp
make setup          # uv sync from uv.lock, then the CPU tests
```

`uv.lock` pins every package, including torch 2.6.0+cu124, transformers 4.57.6 and
lm-eval 0.4.9.2, the versions the experiments ran with. Commands below use
`uv run --no-sync`, which runs in that environment without re-resolving it.

## Reproducing the paper from the records (CPU, about a minute)

```bash
make results        # records/ -> results/
make check          # results/ is byte-identical to what records/ produce
```

`make results` runs `aggregate.py` (run tables), `make_report.py` (tables and figures)
and `claims.py` (every number the Results text states, written to
`results/claims.txt`). The files map to the paper as follows.

| Paper | File in `results/` | Paper | File in `results/` |
|---|---|---|---|
| Table 1 | `tables/sources.tex` | Figure 1 | `figures/entropy.pdf` |
| Table 2 | `tables/oblique.tex` | Figure 2 | `figures/dumbbell_r30.pdf` |
| Table 3 | `tables/main.tex` | Figure 3 | `figures/kl_matrix_r50.pdf` |
| Table 4 | `tables/klsummary.tex` | Figure 4 | `figures/domains_r30.pdf` |
| Table 5 | `tables/domain.tex` | Figure 5 | `figures/size.pdf` |
| Table 6 | `tables/ablations.tex` | Figure 6 | `figures/domains_8b.pdf` |
| Table 7 | `tables/alloc.tex` | Figure A1 | `figures/dumbbell_r50.pdf` |
| Table 8 | `tables/factors.tex` | Figure A2 | `figures/kl_matrix_r30.pdf` |
| Table 9 | `tables/seeds.tex` | Figure A3 | `figures/domains_r50.pdf` |
| Table 10 | `tables/scale.tex` | Figure A4 | `figures/subspaces_r50.pdf` |
| Tables A1-A4 | `tables/repro.tex`, `lr.tex`, `bos.tex`, `tasks.tex` | | |

The tables are the LaTeX bodies inserted into the paper's `tabular` environments.
`results/runs.csv` has one row per run with its configuration and every metric
(`init_*`: training-free initialization, `final_*`: trained model; `ppl_*`: perplexity
on BOS-prefixed windows, `pplstream_*`: GPTQ-style windows, `sp_*`: SlimPajama domains,
`zs_*`: zero-shot accuracy, `kl_*`: divergence from the dense model on held-out sets).

## Re-running the experiments (GPU)

The 182 runs took about 80 GPU-hours on NVIDIA A100 (40 and 80 GB), H100 and H200 GPUs,
plus a few hours to sample the self-generated calibration sets. Every job needs one
GPU with bfloat16 and at least 40 GB of memory; the Llama-3.1-8B jobs need 80 GB.

**1. Data and models.** Accept the license of the gated
[Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) and
[Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) repositories, log in
(`uv run --no-sync huggingface-cli login`), then, on a machine with internet access:

```bash
uv run --no-sync python scripts/prefetch.py    # models, corpora and benchmarks, ~25 GB
```

Models and corpora are loaded at the exact Hub commits the experiments used, pinned
in `src/fsc/hub.py`, so later changes to the Hub repositories do not affect a re-run.

**2. Check the GPU and the pipeline.**

```bash
uv run --no-sync python scripts/verify_gpu.py              # the torch build runs kernels here
uv run --no-sync python scripts/submit.py smoke --local    # a 2-minute end-to-end run
```

**3. Run the plans.** `scripts/plans.py` defines every job of the paper as named plans;
`all` runs them in dependency order. On a Slurm cluster:

```bash
export FSC_PARTITIONS=<partitions with >= 40 GB GPUs>     # optional; Slurm's default otherwise
export FSC_PARTITIONS_80G=<partitions with >= 80 GB GPUs> # for Llama-3.1-8B
uv run --no-sync python scripts/submit.py all --list      # what is done, blocked, to run
uv run --no-sync python scripts/submit.py all             # submit; re-run to top up the queue
```

`submit.py` keeps at most `--max-jobs` (default 12) of your jobs queued, skips finished
jobs, and holds a job until the files it needs exist (for example a self-generated
calibration set), so running it again is always safe. Jobs run through
`slurm/run.slurm` with `HF_HUB_OFFLINE=1`, for compute nodes without internet access.
Without Slurm, add `--local` to run the jobs one after another on this machine.
`--only` selects jobs by name; a blocked job names the file it is waiting for. For
example, one run of the main grid, after the held-out self-samples it is evaluated on:

```bash
uv run --no-sync python scripts/submit.py selfgen --only llama-3.2-1b-T1.0-s1000 --local
uv run --no-sync python scripts/submit.py main --only llama-3.2-1b-proj-c4-r50 --local
```

**4. Compare.** Each run writes `out/runs/<model>/<run>/`, in the layout of
`records/runs/`; source statistics go to `out/stats/` and subspace overlaps to
`out/subspaces/`. Compare your runs with the records, and rebuild the paper's
tables and figures from them:

```bash
uv run --no-sync python scripts/compare_records.py            # run by run
make results RECORDS=out RESULTS=out/results                 # tables, figures, claims
```

A run that stops (time limit, preemption) resumes from its last completed phase when
resubmitted; training checkpoints every epoch.

| Plan | Jobs | Paper | Plan | Jobs | Paper |
|---|---|---|---|---|---|
| `dense` | 4 | reference rows | `size`, `fresh` | 9 + 2 | Figure 5 |
| `selfgen`, `selfdoc` | 11 + 5 | self-generated sets | `temperature` | 2 | Table 6 |
| `stats` | 3 | Table 1, Figure 1 | `factors` | 12 | Table 8 |
| `pilot-lr` | 12 | Table A2 | `klalloc` | 6 | Table 7 |
| `pilot-opt125m` | 12 | Appendices A-B | `domain` | 3 | Table 5 |
| `main` | 36 | Tables 3-4, Figures 2-4 | `crossover`, `bos` | 4 + 5 | Table 6, Table A3 |
| `oblique` | 36 | Table 2 | `repro`, `diag-oblique`, `repro-init` | 6 + 3 + 9 | Table A1 |
| `subspaces` | 6 | Figure A4 | `scale` | 12 | Table 10, Figure 6 |
| `seeds` | 12 | Table 9 | `smoke` | 1 | installation check |

### What to expect from a re-run

`tests/test_plans.py` checks that the plans define exactly the published runs, with the
configuration recorded in each `config.json`. Re-running five records from a fresh
clone of this repository gave these differences from the records
(`scripts/compare_records.py` prints the same comparison for your own runs):

| Run | WikiText-2 perplexity | C4 perplexity | SlimPajama perplexity | Mean accuracy |
|---|---|---|---|---|
| `opt-125m/dense` | +0.00% | +0.00% | +0.00% | +0.07 points |
| `opt-1.3b/dense` | +0.00% | +0.00% | +0.00% | -0.11 points |
| `llama-3.2-1b/dense` | +0.00% | +0.01% | +0.00% | -0.04 points |
| `qwen3-1.7b/dense` | +0.00% | +0.00% | +0.00% | +0.00 points |
| `llama-3.2-1b/proj-c4-r50-s0` | +4.33% | -0.22% | +0.88% | -0.14 points |

Evaluation reproduces exactly on the same GPU architecture (the Qwen3 record and its
re-run both ran on A100 GPUs), and within about 0.1 accuracy points across
architectures (the OPT and Llama records were evaluated on H100 or H200 GPUs, the
re-runs on A100 GPUs), with perplexities equal to 0.01%. Training is not bit-for-bit
reproducible on GPUs, whose backward kernels are not deterministic: on the same GPU
type as its record, the trained run reached a best validation KL within 0.2% of the
record and matches it on its calibration domain (C4), while its WikiText-2
perplexity differs by 4.3%, of the order of the seed-to-seed variation reported in
the paper (Table 9).

## Records

`records/runs/<model>/<run>/` holds, for every run:

| File | Content |
|---|---|
| `config.json` | every argument of `scripts/compress.py` |
| `metrics.json` | `init` and `final` metrics (perplexities, SlimPajama domains, zero-shot accuracy, KL to the dense model), the training summary and history, `merge_check_kl` (projected vs merged model), parameter counts, time and peak memory per phase |
| `alloc.json` | the number of directions removed from every unit, and the realized ratio |
| `train_log.jsonl` | train and validation KL after every epoch |
| `kl_curves.json` | measured-KL allocation only: the KL of every unit at every removal fraction |
| `env.json` | software versions, GPU, driver |

The zero-shot results of 33 runs, first evaluated with an lm-eval batch size of 32,
were re-evaluated at the final size of 16 (`scripts/compress.py --reeval-zs`, which
rebuilds the initialization and the trained model and reruns only the zero-shot
tasks), so that every record uses the same evaluation setting; their `metrics.json`
keeps the earlier results under `zs_reevaluated`.

Run names read `<mode>-<source>-r<ratio>-s<seed>[-<variant>]`: `proj` learned
projectors, `none` training-free truncation, `factors` free low-rank factors, `dense`
the uncompressed model. `records/stats/` holds the per-source statistics of Table 1 and
Figure 1, `records/subspaces/` the overlaps of Figure A4. The trained projectors
(`best_params.pt`) and initial bases are not included: they are regenerated by the
runs, and every reported number is in the records.

## Code

| Module | Role |
|---|---|
| `fsc.arch` | which linear layers form units (tied query/key/value and gate/up groups), and their parameter costs |
| `fsc.data` | calibration windows for every source, evaluation streams, held-out KL sets |
| `fsc.stats` | input and output Gram matrices of every unit |
| `fsc.init` | whitened bases: the orthogonal recast that initializes LSP, and the oblique truncation of SVD-LLM |
| `fsc.alloc` | uniform and measured-KL rank allocation |
| `fsc.project` | projectors, the projector bank, the exact merge into low-rank factors, free factors |
| `fsc.train` | distillation of the projectors against the dense model |
| `fsc.evaluate` | perplexity and zero-shot accuracy (lm-eval) |
| `fsc.selfgen` | text sampled from the dense model: exact sequences and independent documents |
| `fsc.hub` | pinned Hugging Face revisions of every model and corpus |

`scripts/compress.py --help` documents a single run. `scripts/status.py` prints the
validation-KL trajectory of runs in progress, and `scripts/compare_records.py`
compares finished runs with the records.

## Citation

```bibtex
@misc{pham2026calibration,
  title   = {When Calibration Data Become Training Data: Specialization and Self-Generated Text in Learned Low-Rank Compression of Large Language Models},
  author  = {Pham, Van Tien},
  year    = {2026},
}
```

The compression method is learned subspace projections, reimplemented here:
M. Bini et al., *Learning Functional Subspaces for Neural Network Compression*,
arXiv:2609.40127, 2026.

## License

The code and the records are released under the [Apache License 2.0](LICENSE). The
models and datasets used by the experiments are not redistributed here and keep their
own licenses (in particular the Llama 3.1 and 3.2 community licenses).
