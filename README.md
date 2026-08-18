# EpiScore — corpus redundancy measurement for egocentric training data

Measures what share of a video training corpus is content that already
appeared earlier in that same corpus, and how much of it is too motion-blurred
to train on.

Built against **Egocentric-10K** (Build AI, Apache-2.0): 85 factories, 2,138
workers, 192,900 clips, 1080p/30fps H.265, ~18 TB.

## Status

**Method validated across factories; figure measured on one.** The redundancy
number comes from factory 051 (8 workers, 416 clips). The embedding it relies
on has now been checked against **six physically distinct factories**:
separation +0.155 and **zero of 20,000 cross-factory pairs** clear the
duplicate threshold, so the figure is not an artifact of CLIP collapsing
industrial interiors. What remains untested is whether the *figure itself*
generalises to other plants. See [The validation gate](#the-validation-gate).

## Result so far

Across 7,993 sampled keyframes from 7 workers in factory 051:

| blur gate | frames | redundant @0.92 |
|---|---|---|
| none | 7,993 | 23.6% |
| drop blurriest 10% | 7,193 | 25.7% |
| drop blurriest 20% | 6,394 | 27.2% |
| drop blurriest 30% | 5,595 | 27.3% |
| drop blurriest 40% | 4,796 | 27.4% |

**Roughly a quarter of frames duplicate earlier content** — 24% to 27%
depending on how aggressively blur is filtered. The figure moves 3.8 points
across the whole sweep and plateaus past 20%, so the cutoff is not carrying
the claim.

Redundancy is almost entirely *self*-redundancy: within-worker measures 27.2%
and the global figure is also 27.2%. Workers repeat themselves; they do not
duplicate each other.

## Method

A frame is **redundant at threshold t** if its cosine similarity to any
*earlier* frame in the corpus is at least t.

Computed threshold-independently: one pass records each frame's best match
against everything preceding it, and the whole curve derives from that array.
The alternative — greedy keep/drop — must be re-run per threshold and its
answer depends on how drops cascade. That hidden knob would make a published
figure indefensible.

Reported at three scopes, never blended: within worker, within factory, global.

## Motion blur is a confound, not just noise

Visual inspection of sampled pairs found a 0.906 pair that matched purely on
shared blur, ranking *above* a 0.872 pair that was genuinely the same
workstation with the same parts bin. Blur destroys the detail that
distinguishes frames, so blurred frames collapse toward each other in
embedding space and register as duplicates of one another.

Frames are therefore scored for sharpness (variance of the Laplacian) and the
blurriest fraction is dropped before measurement. Filtering blur *raises*
measured redundancy, because it removes low-information noise rather than real
repeats — which is why the sensitivity sweep above is published alongside the
number rather than after someone asks for it.

Blur share is independently useful: frames too blurred to distinguish are also
too blurred to train on, whether or not they repeat.

## The validation gate

**`validate.py` must pass before any number leaves this repo.**

CLIP was trained on web photography, not fisheye industrial video. If factory
interiors collapse into a narrow region of the embedding space, everything
looks alike and redundancy is inflated — the metric would be measuring CLIP's
blind spot rather than the corpus.

The corpus spans 85 physically distinct factories, which is a free control:
frames from *different factories* must be clearly less similar than frames
from the *same worker*. The gate fails if separation is under 0.10, or if more
than 1% of different-factory pairs clear the threshold.

Measured on factory 051 (single-factory fallback, cross-*worker* control):

| scope | mean | p95 | ≥0.92 |
|---|---|---|---|
| same worker | 0.709 | 0.854 | 0.16% |
| different worker | 0.603 | 0.700 | 0.00% |

Separation +0.107, zero false duplicates. CLIP does discriminate here, and
0.92 sits above the 99th percentile of random same-worker pairs — a
conservative threshold.

### Cross-factory control (`validate.py`, 6 factories)

The decisive test. Its pass criteria were written into the code before the
gated data was reachable: separation above 0.10, and under 1% of
cross-factory pairs clearing the threshold.

| scope | mean | p95 | p99 | >=0.92 |
|---|---|---|---|---|
| same worker | 0.767 | 0.907 | 0.935 | 2.62% |
| different factory | 0.612 | 0.751 | 0.798 | **0.00%** |

Separation **+0.155**. Not one pair in twenty thousand. PASS.

### Workstation identification (`separability.py`)

Mean separation is blunt. The direct test: if the embedding had collapsed
this footage, it could not tell workstations *inside one factory* apart.

A linear classifier on the raw embeddings identifies which of 7 workstations
a frame came from with **92.9% accuracy against 14.3% chance** (6.5x), with
per-worker recall from 86.1% to 99.7%. The split is **by clip**, not by
frame — frames seconds apart would otherwise leak the answer.

This is arguably a harder test than the cross-factory one, since stations
inside one plant resemble each other far more than two plants do.

**Two caveats.** Per-worker fisheye intrinsics differ, so part of that signal
may be lens rather than scene. And the method being validated does not
validate the *scope*: one factory is still one factory. The cross-factory run
tests generalisation.

## Score your own corpus

```bash
pip install -r requirements.txt
python src/score.py /path/to/clips --json report.json
```

Operators are inferred from immediate subdirectories, or from a filename
pattern: `--pattern "(?P<operator>op\d+)"`. Output is a per-operator
scorecard — per-operator because measured redundancy varies ~6x between
operators, so a single corpus figure averages away the thing worth acting on.

### Two gates run before any number is reported

**Saturation.** How often do *random* frame pairs already clear the duplicate
threshold? Real footage sits near zero (0.03%, median pair similarity 0.647).
If unrelated frames routinely score as duplicates, "duplicate" has stopped
meaning anything on that corpus and no figure is reported.

**Operator separability.** Can a linear model name the operator from a frame?
Below 2x chance, the embedding is not resolving the footage.

Both gates exit non-zero rather than annotating a bad number, because a
redundancy figure from a broken embedding is not an approximation — it is a
fabrication.

> The saturation gate exists because of a real failure. An end-to-end run on
> synthetic footage reported 97.4% redundancy and ranked a deliberately varied
> operator as the *most* repetitive. Separability passed it at 2.5x chance —
> operators were still coarsely distinguishable — but unrelated frames were
> scoring 0.904 against a 0.92 threshold. Coarse separability is not enough;
> the scale itself has to be intact.

## Research pipeline

For reproducing the Egocentric-10K measurement specifically:

```bash
python src/pipeline.py --tag myrun --source subset
```

## Tests

```bash
python -m pytest tests -q
```

18 tests over the measurement core, sharpness, both gates, and corpus
discovery. `tests/make_fixture.py` generates a synthetic corpus with known
structure; note that CLIP cannot resolve abstract synthetic patterns, so that
fixture exercises the plumbing and the saturation gate, not ranking accuracy.

`--source subset` reads the open [Voxel51 mirror][mirror] (factory 051, 8
workers, 37 GB). `--source gated` reads the full 85-factory repo and needs a
Hugging Face token with access granted.

Individual steps:

```bash
python src/run_embed.py --tag myrun --source subset --workers 8
python src/validate.py --tag myrun        # gate — must pass
python src/run_report.py --tag myrun --blur-percentile 20
python src/sensitivity.py --tag myrun
python src/extract_pairs.py --tag myrun   # renders pairs for eyeballing
```

[mirror]: https://huggingface.co/datasets/Voxel51/Egocentric_10K_subset

## Pipeline

```
list_shards      breadth-first across factories, not depth-first
stream_shard     tar streamed from HF; raw video never hits disk
sample_frames    keyframe-only decode (H.265 I-frames land every 1-4s)
                 + Laplacian-variance sharpness per frame
Embedder         CLIP ViT-B/32, fp32 compute, fp16 storage
measure          FAISS IndexFlatIP, blocked, strictly-earlier matches
```

Runs checkpoint after every worker: a long run is network-bound and can die on
any single clip.

## Known limitations

- **One factory measured.** Factory footage is also the easy case — repetitive
  manual work should show high redundancy. This does not generalise to kitchen
  or household corpora without re-running.
- **Fisheye is not corrected.** Per-worker intrinsics differ, so cross-worker
  comparison is confounded. Within-worker numbers are the trustworthy ones
  until undistortion lands.
- **Keyframe sampling** ties the sample rate to encoder settings rather than a
  fixed interval. Fine for scene-level redundancy, wrong for anything
  motion-sensitive.
- **Redundant is not worthless.** Repetition carries real signal for
  robustness. The claim is that buyers should be able to *price* redundancy,
  not that it should be deleted.

## Prior art

a16z's [ARES](https://github.com/jacobphillips99/ares/) does ingestion, VLM
annotation, and FAISS similarity search over *task-instruction text* and
*trajectory* space. It does not do visual near-duplicate detection or
redundancy scoring; those are listed as future work.

## Hardware notes

Developed on a GTX 1050 Ti (4 GB, Pascal). fp16 arithmetic runs at 1/64 rate
on GP107, so compute stays fp32 deliberately — half precision would be slower
here, not faster. A full 8-worker pass takes about an hour, network-bound.
