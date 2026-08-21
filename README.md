# EpiScore — measuring redundancy per operator in video training data

Measures what share of a video corpus duplicates content it already contains,
**broken down by who captured it**, and how much of it is too motion-blurred
to train on.

**[Read the findings](https://eitanp214.github.io/episcore/)** — the numbers,
the charts, the motion-blur confound that inverted the similarity ranking, and
both validation gates.

## Where this sits

**Near-duplicate detection is a solved, shipped feature.** Before using
anything here, know what already exists and is likely enough:

| tool | what it does | availability |
|---|---|---|
| [FiftyOne](https://docs.voxel51.com/) | `find_duplicates()`, image-dedup plugin, `compute_uniqueness()` | open source, 2.8M installs |
| [Cleanlab Datalab](https://docs.cleanlab.ai/) | near-duplicate detection with a defined distance criterion | open source |
| [Encord Active](https://docs.encord.com/) | Uniqueness metric, flags near-duplicates side by side | commercial |
| [Lightly](https://www.lightly.ai/) | curation and selection, claims 50%+ training-cost cuts | commercial |

If you want to deduplicate an image dataset, use one of those. They are
mature, better supported, and free or funded.

### What this adds

Three things I could not find elsewhere, all narrow:

1. **Per-operator breakdown.** Existing per-contributor quality reporting
   measures *label correctness* — inter-annotator agreement, gold-set
   accuracy. This measures *content redundancy* per capturer, which applies
   to collection rather than annotation. On the corpus below, that figure
   ranged **7.4% to 46.0% between operators** — a 6.2x spread the corpus
   average hides completely.
2. **Two gates that refuse to report.** Most tools hand back a number
   regardless. These exit non-zero when the embedding cannot support one.
3. **A published figure on a named open corpus**, with the sensitivity sweep
   and the confounds documented rather than omitted.

None of this is a moat. A vendor already reporting per-annotator agreement
would find per-operator redundancy a natural extension and could ship it in a
sprint. It is a measurement and a finding, not a product.

## Status

**Method validated across factories; figure measured on one.** The redundancy
number comes from factory 051 of Egocentric-10K (8 workers, 416 clips). The
embedding it relies on was checked against **six physically distinct
factories**: separation +0.175 across all 15 factory pairs, and **zero of
20,000 cross-factory pairs** clear the duplicate threshold. Whether the
*figure itself* generalises to other plants is untested.

## Result

Across 7,993 sampled keyframes from 7 workers in factory 051 of
**Egocentric-10K** (Build AI, Apache-2.0):

| blur gate | frames | redundant @0.92 |
|---|---|---|
| none | 7,993 | 23.6% |
| drop blurriest 10% | 7,193 | 25.7% |
| drop blurriest 20% | 6,394 | 27.2% |
| drop blurriest 30% | 5,595 | 27.3% |
| drop blurriest 40% | 4,796 | 27.4% |

**Roughly a quarter of frames duplicate earlier content** — 24% to 27%
depending on blur filtering, 95% CI [24.6%, 29.9%] by jackknife. The figure
moves 3.8 points across the whole sweep and plateaus past 20%, so the cutoff
is not carrying the claim.

**The corpus average is the least useful number here.** Per operator it runs
7.4% to 46.0%. That spread is ~39 points wide, an order of magnitude larger
than the sampling error on the mean.

## Method

A frame is **redundant at threshold t** if its cosine similarity to any
*earlier* frame in the corpus is at least t.

Computed threshold-independently: one pass records each frame's best match
against everything preceding it, and the whole curve derives from that array.
The alternative — greedy keep/drop — must be re-run per threshold and its
answer depends on how drops cascade. That hidden knob would make a published
figure indefensible.

### Bootstrap is invalid for this statistic

Resampling clips with replacement puts the same clip in the corpus twice, and
this metric measures duplication, so every repeated clip scores as a perfect
match against its own copy. A first attempt returned a 95% interval of
[49.2%, 53.4%] around a 27.2% estimate — an interval that cannot contain its
own point. The published figure uses a jackknife.

## Motion blur is a confound, not just noise

Visual inspection of sampled pairs found a 0.906 pair that matched purely on
shared blur, ranking *above* a 0.872 pair that was genuinely the same
workstation with the same parts bin. Blur destroys the detail that
distinguishes frames, so blurred frames collapse toward each other in
embedding space and register as duplicates of one another.

Frames are scored for sharpness (variance of the Laplacian) and the blurriest
fraction dropped before measurement. Filtering blur *raises* measured
redundancy, which is why the sensitivity sweep is published alongside the
number rather than after someone asks.

## The two gates

**Both exit non-zero rather than annotate a bad number**, because a
redundancy figure from a broken embedding is not an approximation — it is a
fabrication.

### Saturation

How often do *random* frame pairs already clear the duplicate threshold? Real
footage measured 0.03% at a 0.647 median. A synthetic corpus that broke the
metric measured 26.2% at 0.904.

> This gate exists because of a real failure. An end-to-end run on synthetic
> footage reported 97.4% redundancy and ranked a deliberately varied operator
> as the *most* repetitive. Separability passed it at 2.5x chance — operators
> were still coarsely distinguishable — but unrelated frames were scoring
> 0.904 against a 0.92 threshold. Coarse separability is not enough; the
> scale itself has to be intact.

### Separability

Can a linear model name the operator from one frame? Below 2x chance the
embedding is not resolving the footage.

Measured across six factories, sampling **all 15 factory pairs** rather than
one factory against the rest:

| scope | mean | p95 | p99 | ≥0.92 |
|---|---|---|---|---|
| same worker | 0.767 | 0.907 | 0.935 | 2.62% |
| different factory | 0.612 | 0.751 | 0.798 | **0.00%** |

Separation **+0.175**. Pass criteria were written into `validate.py` before
the gated data was reachable: above 0.10, and under 1% of cross-factory pairs
clearing the threshold.

Leave-one-factory-out spans +0.152 to +0.194 (`factory_stability.py`), so the
pass does not depend on which factories were drawn.

A linear classifier also identifies which of 7 workstations *inside one
factory* a frame came from at **92.9% against 14.3% chance**, split by clip so
adjacent frames cannot leak the answer.

## Score your own corpus

```bash
pip install -r requirements.txt
python src/score.py /path/to/clips --json report.json
```

Operators are inferred from immediate subdirectories, or from a filename
pattern: `--pattern "(?P<operator>op\d+)"`.

## Reproduce the measurement

```bash
python src/pipeline.py --tag myrun --source subset
```

`--source subset` reads the open [Voxel51 mirror][mirror] (factory 051, 8
workers, 37 GB). `--source gated` reads the full 85-factory repo and needs a
Hugging Face token with access granted; `verify_access.py` diagnoses the 401s.

[mirror]: https://huggingface.co/datasets/Voxel51/Egocentric_10K_subset

## Tests

```bash
python -m pytest tests -q
```

18 tests over the measurement core, sharpness, both gates, and corpus
discovery. `tests/make_fixture.py` generates a synthetic corpus with known
structure — note that CLIP cannot resolve abstract synthetic patterns, so that
fixture exercises the plumbing and the saturation gate, not ranking accuracy.

## Known limitations

- **One factory measured.** Factory work is the easy case: repetitive manual
  labour produces repetitive footage. Says nothing about kitchen or household
  corpora without re-running.
- **Fisheye uncorrected.** Per-worker intrinsics differ, so part of the
  workstation-identification signal may be lens rather than scene.
- **Keyframe sampling** ties the rate to encoder settings rather than a fixed
  interval. Fine for scene-level redundancy, wrong for motion-sensitive work.
- **No resume in `score.py`.** A large corpus that dies mid-run starts over.
  `run_embed.py` checkpoints; the product path does not.
- **Reports a percentage, not a file list.** It tells you how much repeats,
  not which clips to drop.
- **Redundant is not worthless.** Repetition carries real signal for
  robustness, and failure footage is valuable for recovery training. The
  argument is that buyers should be able to *price* redundancy.

## Prior art

a16z's [ARES](https://github.com/jacobphillips99/ares/) does ingestion, VLM
annotation, and FAISS similarity search over task-instruction text and
trajectory space — not visual near-duplicate detection. The commercial tools
in [Where this sits](#where-this-sits) do handle visual dedup, on images.

## Hardware

Developed on a GTX 1050 Ti (4 GB, Pascal). fp16 runs at 1/64 rate on GP107, so
compute stays fp32 deliberately — half precision would be slower here. A full
8-worker pass takes about an hour, network-bound.

## Licence

MIT.
