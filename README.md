# CS2-LeWM

Latent world model experiments for Counter-Strike 2, trained from POV gameplay clips using a LeWorldModel-style setup.

This repository focuses on learning compact latent dynamics from unlabelled gameplay video and analyzing whether the learned state space supports prediction, retrieval, and planning.

## Highlights

- End-to-end JEPA-style world model training on CS2 POV data.
- DeltaTok-inspired change-centric prediction using residual latent updates.
- Latent diagnostics and visual analysis outputs in the assets directory.
- Static project report in index.html for GitHub Pages style presentation.

## Project Layout

```text
cs2lewm/
|- train.py
|- cs2_showcase_tests.py
|- index.html
|- README.md
`- assets/
   |- CEM_planner.png
   |- delta_analysis.png
   |- per_key_analysis.png
   |- S1_embedding_health.png
   |- S2_tsne.png
   |- S3_reconstructions.png
   |- S4_action_sensitivity.png
   |- S5_direction_showcase.png
   |- V1_action_sensitivity.png
   |- V2_embedding_health.png
   |- V3_tsne.png
   |- V4_reconstruction.png
   |- V5_interpolation.png
   |- V6_nn_retrieval.png
   |- V7_prediction_quality.png
   `- V9_action_trajectories.png
```

## Quick Start

Install dependencies:

```bash
pip install torch torchvision transformers einops opencv-contrib-python-headless scikit-learn matplotlib numpy
```

Run smoke check:

```bash
python train.py --stage smoke
```

Run full training budget:

```bash
python train.py --stage all --budget-hours 3.4
```

Run evaluation only:

```bash
python train.py --stage eval
```

## Core Notes

- `train.py` is the cleaned main training and evaluation script.
- `cs2_showcase_tests.py` contains analysis notebook cells and now imports from `train`.
- `assets/` stores all test and result images used by `index.html` and this README.
- `index.html` is a light-themed academic project page suitable for GitHub Pages.

## Delta-Token Inspired Usage (DeltaTok)

This project explicitly applies the DeltaTok intuition from arXiv:2604.04913 in the predictor path:

- Instead of directly predicting the full next latent state, the model predicts the latent change.
- The residual update is implemented as `z_hat = z_t + predictor(z_t, action)`.
- Training uses a delta-style objective where `delta = z_{t+1} - z_t` and `delta_hat = z_hat - z_t`.
- This reduces collapse toward the trivial no-change predictor on dense 10 fps gameplay data.

In short: we keep the LeWorldModel training setup, but make prediction explicitly change-centric in the spirit of DeltaTok.

## Project Structure

- train.py: Main training, decoder, and evaluation pipeline.
- cs2_showcase_tests.py: Additional diagnostics and visualization routines.
- index.html: Project report page.
- assets/: Generated figures used by README and index page.

## Showcase Tests and Why These Results Are Good

The showcase outputs are not just visuals; each one verifies a specific world-model property from the training run.

- Embedding Health (assets/V2_embedding_health.png): all 256 latent dimensions remain active, and the latent distribution tracks the SIGReg Gaussian target. This is strong evidence that representation collapse was avoided.
- Latent Space Structure (assets/V3_tsne.png and assets/S2_tsne.png): t-SNE projections separate scoped and non-scoped gameplay states without labels, which indicates meaningful semantic organization in latent space.
- Nearest-Neighbour Retrieval (assets/V6_nn_retrieval.png): latent neighbors are visually and semantically related to the query frame, showing that learned embeddings preserve gameplay-relevant similarity.
- Decoder Reconstruction (assets/V4_reconstruction.png): decoded frames recover HUD layout, scope circle, and scene geometry from 256-d latents, confirming that latent codes retain useful visual state information.
- Multi-Step Prediction Quality (assets/V7_prediction_quality.png): cosine similarity stays high across rollout steps, indicating stable forward dynamics prediction and low drift over short horizons.
- Residual Delta Analysis (assets/delta_analysis.png): predicted latent deltas align with real latent changes, and the model beats the null baseline by focusing on action-driven change.
- Action Sensitivity and Direction Tests (assets/S4_action_sensitivity.png and assets/S5_direction_showcase.png): pitch and yaw dominate latent movement, and action-conditioned rollouts diverge in consistent directions, which confirms genuine action conditioning rather than static memorization.

Together, these tests show that the model learns a stable latent space, predicts temporal dynamics coherently, and responds to actions in a physically plausible way.

## References

- LeWorldModel paper: https://arxiv.org/abs/2603.19312
- DeltaTok paper: https://arxiv.org/pdf/2604.04913
- Original implementation reference: https://github.com/lucas-maes/le-wm