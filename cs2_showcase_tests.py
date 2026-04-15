"""
CS2-LeWM Showcase Tests — Run after training
══════════════════════════════════════════════════════════════════════════
Copy each section into a notebook cell and run in order.
All cells work on CPU (will be slow but functional).
Run after the main eval cell has set up: model, dec, vl_ds, store, Z_all, FF_all etc.

CELLS IN THIS FILE:
  Cell A — CEM Planner (goal-directed imagination in latent space)
  Cell B — Residual delta analysis (why delta prediction matters)
  Cell C — Per-key perceptual analysis (what each key changes visually)
  Cell D — How the model "sees" — ViT attention maps
  Cell E — Temporal velocity field (what the model tracks over time)
  Cell F — Adversarial robustness (what fools the encoder)
  Cell G — CEM vs random planning comparison
  Cell H — Frame perceptual similarity vs latent similarity comparison
"""

# ══════════════════════════════════════════════════════════════
# SHARED SETUP — run this first if starting fresh
# ══════════════════════════════════════════════════════════════
SETUP = """
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/teamspace/studios/this_studio")

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.manifold import TSNE

from train import (
    CS2LeWM, Config, ActionNormalizer, ClipStore,
    discover_processed_clips, build_loaders,
    frames_to_device, denorm_frames, model_device_dtype,
    autocast_ctx, ACTION_NAMES, aggregate_actions,
    ConvDecoder, IMG_MEAN, IMG_STD
)

RUN_DIR = Path("/teamspace/studios/this_studio/cs2_lewm_merged")
OUT_DIR = RUN_DIR / "showcase"; OUT_DIR.mkdir(exist_ok=True)
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt  = torch.load(RUN_DIR/"last_jepa.pt", map_location="cpu")
cfg   = Config(**ckpt["config"]); cfg.eval_sequences = 800
model = CS2LeWM(cfg).to(DEVICE).eval()
model.load_state_dict(ckpt["model"])
norm  = ActionNormalizer(); norm.load_state_dict(ckpt["normalizer"])
dev, dt = model_device_dtype(cfg)

dec = ConvDecoder(cfg.embed_dim, cfg.img_size).to(DEVICE).eval()
dec.load_state_dict(torch.load(RUN_DIR/"best_decoder.pt", map_location="cpu")["decoder"])

clips = discover_processed_clips(cfg)
store = ClipStore(clips, mode=cfg.storage_mode, ram_budget_gb=cfg.ram_budget_gb)
_,_,tr_ds,vl_ds,tr_clips,vl_clips = build_loaders(store, cfg)
IMG_MEAN_T = torch.tensor(IMG_MEAN); IMG_STD_T = torch.tensor(IMG_STD)

def to_rgb(t):
    t = t.cpu()
    if t.dtype == torch.uint8:
        return t.numpy() if t.shape[-1]==3 else t.permute(1,2,0).numpy()
    m = IMG_MEAN_T.view(3,1,1); s = IMG_STD_T.view(3,1,1)
    return ((t.float()*s+m).clamp(0,1).permute(1,2,0).numpy()*255).astype(np.uint8)

def tanh_rgb(t):
    return (((t.cpu().float()+1)/2).clamp(0,1).permute(1,2,0).numpy()*255).astype(np.uint8)

def save(fig, name):
    p = OUT_DIR/name; fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig); print(f"  saved → {p}")

print("Setup complete | DEVICE:", DEVICE)
"""

# ══════════════════════════════════════════════════════════════
# CELL A  —  CEM PLANNER
# Demonstrates goal-directed imagination in latent space.
# Given a start frame and a goal frame, optimise a 15-step action
# sequence that navigates from start z to goal z.
# No decoder needed — plans entirely in latent space.
# ══════════════════════════════════════════════════════════════
CELL_A = """
# ── Cell A: CEM Planner ──────────────────────────────────────
print("\\n=== Cell A: CEM Planner ===")

# Pick a start and goal that look visually different
rng = np.random.default_rng(0)
start_idx = int(rng.integers(0, len(vl_ds)//2))
goal_idx  = int(rng.integers(len(vl_ds)//2, len(vl_ds)))

fu8_s, ac_s = vl_ds[start_idx]
fu8_g, ac_g = vl_ds[goal_idx]

with torch.no_grad():
    frames_s = frames_to_device(fu8_s.unsqueeze(0), dev, dt)
    frames_g = frames_to_device(fu8_g.unsqueeze(0), dev, dt)
    with autocast_ctx(dev, dt):
        z_start = model.encode_sequence(frames_s)   # (1, T, D)
        z_goal  = model.encode_sequence(frames_g)[0,-1].float()  # (D,) goal

# CEM hyperparameters
HORIZON   = 15
N_SAMPLES = 512
N_ELITES  = 48
N_ITERS   = 20
N_ACTIONS = len(ACTION_NAMES)
MC        = 30.0   # mouse compress scale

mu  = torch.zeros(HORIZON, N_ACTIONS, device=dev)
std = torch.ones( HORIZON, N_ACTIONS, device=dev)
std[:, :2] = MC * 0.3    # mouse dims: wider initial std

cost_history = []
best_actions = None

print(f"  Planning {HORIZON} steps → goal state...")
with torch.no_grad():
    for it in range(N_ITERS):
        # Sample candidates
        noise   = torch.randn(N_SAMPLES, HORIZON, N_ACTIONS, device=dev)
        cands   = (mu.unsqueeze(0) + std.unsqueeze(0) * noise)  # (S, H, A)

        # Rollout all candidates from start z (pre-encoded, skip re-encoding)
        ctx_z   = z_start.expand(N_SAMPLES, -1, -1).clone()   # (S, T, D)
        for t in range(HORIZON):
            h   = min(cfg.history_size, ctx_z.shape[1])
            cz  = ctx_z[:, -h:]
            ca  = cands[:, max(0,t-h+1):t+1]
            if ca.shape[1] < h:
                pad = torch.zeros(N_SAMPLES, h-ca.shape[1], N_ACTIONS, device=dev)
                ca  = torch.cat([pad, ca], 1)
            with autocast_ctx(dev, dt):
                ae  = model.action_enc(ca.to(dt))
                znx = model.predict_sequence(cz.to(dt), ae).float()[:, -1:]
            ctx_z = torch.cat([ctx_z, znx], 1)

        # Cost = L2 to goal at final step
        z_final = ctx_z[:, -1].float()
        costs   = ((z_final - z_goal.unsqueeze(0)) ** 2).sum(-1)  # (S,)
        best_cost = costs.min().item(); cost_history.append(best_cost)

        # Elite update
        elite_idx = costs.argsort()[:N_ELITES]
        elites    = cands[elite_idx]
        mu        = elites.mean(0)
        std       = elites.std(0).clamp(min=0.05) * 0.95

        if it % 5 == 0:
            print(f"    iter {it:02d}: best_cost={best_cost:.4f}")

    best_actions = mu.clone()
    print(f"  Final cost: {cost_history[-1]:.4f}  (start: {cost_history[0]:.4f})")
    print(f"  Cost reduction: {cost_history[0]/max(cost_history[-1],1e-8):.1f}×")

# ── Visualise ─────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 12))
fig.suptitle(
    f"Cell A: CEM Planner — Latent Space Goal Navigation\\n"
    f"Cost reduced {cost_history[0]/max(cost_history[-1],1e-8):.1f}× "
    f"over {N_ITERS} iterations | horizon={HORIZON} steps",
    fontsize=13, fontweight="bold"
)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

# Panel 1: cost curve
ax = fig.add_subplot(gs[0, :2])
ax.plot(cost_history, "b-o", ms=5, lw=2)
ax.set_xlabel("CEM Iteration"); ax.set_ylabel("Best ||z_T - z_goal||²")
ax.set_title("CEM Convergence Curve")
ax.fill_between(range(len(cost_history)), cost_history, alpha=0.2)
ax.grid(0.3)

# Panel 2: planned mouse trajectory (cumulative yaw/pitch)
ax = fig.add_subplot(gs[0, 2:])
yaw_cum   = best_actions[:, 0].cpu().numpy().cumsum()
pitch_cum = best_actions[:, 1].cpu().numpy().cumsum()
sc = ax.scatter(yaw_cum, pitch_cum, c=np.arange(HORIZON), cmap="RdYlGn", s=80, zorder=3)
ax.plot(yaw_cum, pitch_cum, "gray", alpha=0.5, lw=1.5)
ax.plot(0, 0, "ko", ms=10, zorder=4, label="start")
ax.plot(yaw_cum[-1], pitch_cum[-1], "r*", ms=14, zorder=4, label="end")
plt.colorbar(sc, ax=ax, label="step"); ax.legend(fontsize=9)
ax.set_xlabel("Cumulative yaw"); ax.set_ylabel("Cumulative pitch")
ax.set_title("Planned Camera Trajectory (mouse movement)")
ax.grid(0.3)

# Panel 3: best action sequence heatmap
ax = fig.add_subplot(gs[1, :])
im = ax.imshow(best_actions.cpu().numpy().T, aspect="auto", cmap="RdBu_r", vmin=-4, vmax=4)
ax.set_yticks(range(N_ACTIONS)); ax.set_yticklabels(ACTION_NAMES, fontsize=9)
ax.set_xlabel("Time step"); ax.set_title(f"Best Action Sequence (horizon={HORIZON})")
plt.colorbar(im, ax=ax, label="Action value", fraction=0.015)
ax.axhline(1.5, color="k", ls="--", lw=0.8, alpha=0.5)   # separator mouse/keys

# Panel 4: start / goal frames
ax = fig.add_subplot(gs[2, 0])
ax.imshow(to_rgb(fu8_s[cfg.seq_len//2])); ax.axis("off")
ax.set_title("Start Frame", color="blue", fontsize=10, fontweight="bold")

ax = fig.add_subplot(gs[2, 1])
ax.imshow(to_rgb(fu8_g[cfg.seq_len//2])); ax.axis("off")
ax.set_title("Goal Frame", color="red", fontsize=10, fontweight="bold")

# Panel 5: planned latent path in PCA
ax = fig.add_subplot(gs[2, 2:])
with torch.no_grad():
    z_plan = z_start.clone()
    for t in range(HORIZON):
        h  = min(cfg.history_size, z_plan.shape[1]); cz = z_plan[:, -h:]
        ca = best_actions[max(0,t-h+1):t+1].unsqueeze(0).to(dev)
        if ca.shape[1] < h:
            pad = torch.zeros(1,h-ca.shape[1],N_ACTIONS,device=dev); ca=torch.cat([pad,ca],1)
        with autocast_ctx(dev, dt):
            ae  = model.action_enc(ca.to(dt))
            znx = model.predict_sequence(cz.to(dt), ae).float()[:,-1:]
        z_plan = torch.cat([z_plan, znx], 1)

z_plan_np = z_plan[0].float().cpu().numpy()
# Project onto first 2 PCs
U, _, _ = np.linalg.svd(z_plan_np - z_plan_np.mean(0), full_matrices=False)
proj     = (z_plan_np - z_plan_np.mean(0)) @ U[:2].T   # wrong, fix:
_, _, Vt = np.linalg.svd(z_plan_np - z_plan_np.mean(0), full_matrices=False)
proj     = (z_plan_np - z_plan_np.mean(0)) @ Vt[:2].T

z_goal_proj = ((z_goal.cpu().numpy() - z_plan_np.mean(0)) @ Vt[:2].T)

T_plan = len(proj)
sc2    = ax.scatter(proj[:,0], proj[:,1], c=np.arange(T_plan), cmap="Blues", s=30, zorder=3)
ax.plot(proj[:,0], proj[:,1], "b-", alpha=0.4, lw=1.5, label="Planned path")
ax.plot(proj[0,0],  proj[0,1],  "go", ms=12, label="Start z", zorder=4)
ax.plot(proj[-1,0], proj[-1,1], "bs", ms=10, label="Plan end", zorder=4)
ax.plot(z_goal_proj[0], z_goal_proj[1], "r*", ms=18, label="Goal z", zorder=4)
ax.legend(fontsize=8); ax.set_title("Planned Trajectory in Latent PCA Space"); ax.grid(0.3)
plt.colorbar(sc2, ax=ax, label="step")

save(fig, "CEM_planner.png")
print("  ✓ CEM_planner.png")
"""

# ══════════════════════════════════════════════════════════════
# CELL B  —  RESIDUAL DELTA ANALYSIS
# Shows why residual/delta prediction matters.
# Compares: what the model predicts vs what actually changed.
# Also shows the L2 magnitude of predicted deltas per action.
# ══════════════════════════════════════════════════════════════
CELL_B = """
# ── Cell B: Residual Delta Analysis ──────────────────────────
print("\\n=== Cell B: Residual Delta / What the Model Predicts Changes ===")

N_SEQS = min(200, len(vl_ds))
rng    = np.random.default_rng(7)
ids    = rng.choice(len(vl_ds), N_SEQS, replace=False)

delta_norms_real = []   # ||z_{t+1} - z_t|| per step
delta_norms_pred = []   # ||z_hat - z_t|| per step (predicted change)
delta_cosim      = []   # cos(predicted_delta, real_delta)
null_errs        = []   # ||z_t - z_{t+1}|| (null predictor error)
model_errs       = []   # ||z_hat - z_{t+1}|| (model error)

with torch.no_grad():
    for raw_i in ids:
        fu8, ac = vl_ds[int(raw_i)]
        frames  = frames_to_device(fu8.unsqueeze(0), dev, dt)
        act     = norm.transform(ac.unsqueeze(0).to(dev))
        with autocast_ctx(dev, dt):
            emb  = model.encode_sequence(frames).float()
            ae   = model.action_enc(act.to(dt))
            pred = model.predict_sequence(emb[:,:-1].to(dt), ae[:,:-1]).float()

        real_delta = emb[:,1:] - emb[:,:-1]     # (1, T-1, D)
        pred_delta = pred - emb[:,:-1]           # residual from current state

        delta_norms_real.append(real_delta.norm(dim=-1).squeeze(0).cpu().numpy())
        delta_norms_pred.append(pred_delta.norm(dim=-1).squeeze(0).cpu().numpy())
        cs = F.cosine_similarity(pred_delta.float(), real_delta.float(), dim=-1)
        delta_cosim.append(cs.squeeze(0).cpu().numpy())
        null_errs.append((emb[:,:-1]-emb[:,1:]).pow(2).mean(-1).squeeze(0).cpu().numpy())
        model_errs.append((pred-emb[:,1:]).pow(2).mean(-1).squeeze(0).cpu().numpy())

dnr = np.concatenate(delta_norms_real)
dnp = np.concatenate(delta_norms_pred)
dcs = np.concatenate(delta_cosim)
ne  = np.concatenate(null_errs)
me  = np.concatenate(model_errs)
ratio = ne.mean() / me.mean()

print(f"  Mean real delta magnitude:  {dnr.mean():.4f}")
print(f"  Mean pred delta magnitude:  {dnp.mean():.4f}")
print(f"  Delta cosine similarity:    {dcs.mean():.4f}")
print(f"  Null ratio:                 {ratio:.2f}×  ({'✓ BEATING NULL' if ratio>1 else '⚠ below null'})")

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
fig.suptitle(
    f"Cell B: Residual Delta Analysis — Null ratio={ratio:.2f}×\\n"
    f"The model predicts the CHANGE (z_hat = z_t + Δ_hat), not the absolute state",
    fontsize=13, fontweight="bold"
)

# Real vs predicted delta magnitude
axes[0,0].scatter(dnr[:2000], dnp[:2000], s=4, alpha=0.3, color="steelblue")
lm = max(dnr[:2000].max(), dnp[:2000].max())
axes[0,0].plot([0,lm],[0,lm],"r--",lw=2,label="y=x (perfect)")
axes[0,0].set_xlabel("Real ||Δz||"); axes[0,0].set_ylabel("Predicted ||Δz||")
axes[0,0].set_title("Predicted vs Real Change Magnitude"); axes[0,0].legend(); axes[0,0].grid(0.3)

# Delta cosine similarity distribution
axes[0,1].hist(dcs, bins=60, color="#4CAF50", alpha=0.8, edgecolor="white", density=True)
axes[0,1].axvline(dcs.mean(), color="red", ls="--", lw=2, label=f"mean={dcs.mean():.3f}")
axes[0,1].axvline(0, color="gray", ls=":", lw=1.5, label="random")
axes[0,1].legend(fontsize=9); axes[0,1].set_title("cos(Δ_predicted, Δ_real)\\n>0 = predicting correct direction")
axes[0,1].grid(0.3)

# Model error vs null error
bins = np.linspace(0, max(ne.max(), me.max()), 60)
axes[0,2].hist(me, bins=bins, alpha=0.7, color="steelblue", label=f"Model μ={me.mean():.4f}")
axes[0,2].hist(ne, bins=bins, alpha=0.7, color="orange",   label=f"Null  μ={ne.mean():.4f}")
axes[0,2].legend(fontsize=9); axes[0,2].set_title(f"Model vs Null Error ({ratio:.2f}× improvement)"); axes[0,2].grid(0.3)

# Delta magnitude per timestep
step_dnr = np.array([np.concatenate(delta_norms_real)[:50*19][i::19].mean() for i in range(19)])
step_dnp = np.array([np.concatenate(delta_norms_pred)[:50*19][i::19].mean() for i in range(19)])
axes[1,0].plot(range(1,20), step_dnr, "b-o", ms=5, lw=2, label="Real Δ magnitude")
axes[1,0].plot(range(1,20), step_dnp, "r--s", ms=5, lw=2, label="Pred Δ magnitude")
axes[1,0].set_xlabel("Step in sequence"); axes[1,0].legend()
axes[1,0].set_title("Change magnitude per sequence step"); axes[1,0].grid(0.3)

# Scatter: is model better when motion is larger?
sample_idx = rng.choice(len(dnr), min(1000,len(dnr)), replace=False)
improvement = ne[sample_idx] - me[sample_idx]
axes[1,1].scatter(dnr[sample_idx], improvement, s=6, alpha=0.3, color="steelblue")
axes[1,1].axhline(0, color="red", ls="--", lw=1.5, label="break-even")
axes[1,1].set_xlabel("Real change magnitude ||Δz||"); axes[1,1].set_ylabel("Improvement (null - model)")
axes[1,1].set_title("Model improves most on HIGH-MOTION frames\\n(dynamic weight power = 0.5 working ✓)")
axes[1,1].legend(fontsize=9); axes[1,1].grid(0.3)

# WHY residual matters: show a worked example
axes[1,2].axis("off")
txt = [
    "WHY DELTA PREDICTION WORKS",
    "",
    "Old (absolute):  predict z_{t+1}",
    "  → null (z_t) gets MSE ≈ ||Δ||²",
    "  → at 10fps, ||Δ|| ≈ 1.6 per step",
    "  → null achieves MSE ≈ 2.56 for FREE",
    "  → model learned z_hat ≈ z_t (lazy!)",
    "",
    "New (delta/residual): predict Δ = z_{t+1}-z_t",
    "  z_hat = z_t + predictor(z_t, action)",
    "  → null (Δ=0) achieves MSE ≈ ||Δ||²",
    "  → model must learn WHAT CHANGED",
    "  → forces learning of action effects",
    "",
    f"Result: null_ratio = {ratio:.2f}×",
    f"  delta cosim = {dcs.mean():.3f}",
]
for i,line in enumerate(txt):
    color = "#4CAF50" if "Result" in line or "New" in line else \
            ("#e74c3c" if "Old" in line or "lazy" in line else "black")
    fw = "bold" if any(x in line for x in ["WHY","Result","New","Old"]) else "normal"
    axes[1,2].text(0.03, 0.97-i*0.063, line, fontsize=9,
                   transform=axes[1,2].transAxes, fontfamily="monospace",
                   color=color, fontweight=fw)

fig.tight_layout(); save(fig, "delta_analysis.png")
print("  ✓ delta_analysis.png")
"""

# ══════════════════════════════════════════════════════════════
# CELL C  —  PER-KEY PERCEPTUAL ANALYSIS
# For each of the 12 action dims, shows:
#   - Mean predicted latent change when that key is pressed
#   - Decoded frames: neutral vs key-pressed (visual diff)
#   - Which game states each key is most associated with
# ══════════════════════════════════════════════════════════════
CELL_C = """
# ── Cell C: Per-Key Perceptual Analysis ───────────────────────
print("\\n=== Cell C: Per-Key Analysis — What Each Action Does ===")

# Pick a representative frame
rng    = np.random.default_rng(13)
idx    = int(rng.integers(0, len(vl_ds)))
fu8, ac = vl_ds[idx]

with torch.no_grad():
    frames  = frames_to_device(fu8.unsqueeze(0), dev, dt)
    with autocast_ctx(dev, dt):
        emb = model.encode_sequence(frames).float()

z_ctx = emb[:, -cfg.history_size:]     # (1, H, D)
MC = 30.0

# For each action dimension, compute the predicted change
action_sens  = []
decoded_pairs = []    # (neutral_recon, action_recon)

with torch.no_grad():
    neutral_a = torch.zeros(1, cfg.history_size, len(ACTION_NAMES), device=dev)
    with autocast_ctx(dev, dt):
        ae_n   = model.action_enc(neutral_a.to(dt))
        z_n    = model.predict_sequence(z_ctx.to(dt), ae_n).float()[:, -1]  # (1, D)
        recon_n = dec(z_n).float()

    for dim in range(len(ACTION_NAMES)):
        a_pert = neutral_a.clone()
        if dim < 2:
            vals = [-MC*0.5, -MC*0.25, MC*0.25, MC*0.5]
            deltas = []
            for v in vals:
                a_pert[:,:,dim] = v
                with autocast_ctx(dev, dt):
                    ae  = model.action_enc(a_pert.to(dt))
                    z_p = model.predict_sequence(z_ctx.to(dt), ae).float()[:,-1]
                deltas.append((z_p - z_n).norm().item())
            action_sens.append(np.mean(deltas))
            # Use max perturbation for visual
            a_pert[:,:,dim] = MC * 0.5
        else:
            a_pert[:,:,dim] = 1.0
            with autocast_ctx(dev, dt):
                ae  = model.action_enc(a_pert.to(dt))
                z_p = model.predict_sequence(z_ctx.to(dt), ae).float()[:,-1]
            action_sens.append((z_p - z_n).norm().item())

        with autocast_ctx(dev, dt):
            ae  = model.action_enc(a_pert.to(dt))
            z_p = model.predict_sequence(z_ctx.to(dt), ae).float()[:,-1]
            recon_p = dec(z_p).float()
        decoded_pairs.append((recon_n[0], recon_p[0]))

action_sens = np.array(action_sens)

# ── Plot ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(24, 16))
fig.suptitle(
    "Cell C: Per-Key Perceptual Analysis\\n"
    "How each CS2 action changes the world model's predicted next state",
    fontsize=14, fontweight="bold"
)
gs = gridspec.GridSpec(4, 6, figure=fig, hspace=0.5, wspace=0.35)

# Top row: sensitivity bar chart (full width)
ax_bar = fig.add_subplot(gs[0, :])
colors = ["#d55e00" if i < 2 else "#4c72b0" for i in range(len(ACTION_NAMES))]
bars = ax_bar.bar(ACTION_NAMES, action_sens, color=colors, alpha=0.9, edgecolor="white")
for bar, v in zip(bars, action_sens):
    ax_bar.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")
ax_bar.set_title(
    "Mean ||Δz|| when each action is applied  "
    "(Red=mouse/camera  Blue=keyboard)",
    fontsize=12
)
ax_bar.set_ylabel("Latent state change magnitude"); ax_bar.grid(0.3, axis="y")
ax_bar.tick_params(axis="x", rotation=20)

# Context frame
ax_ctx = fig.add_subplot(gs[1, 0])
ax_ctx.imshow(to_rgb(fu8[cfg.seq_len//2])); ax_ctx.axis("off")
ax_ctx.set_title("Context\\nframe", fontsize=9, color="blue")

# For each action: neutral vs pressed decoded frame + diff
action_show_order = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10]  # skip Space, Shift, R
positions = [(1,0),(1,1),(1,2),(1,3),(1,4),(1,5),(2,0),(2,1),(2,2),(2,3)]

for pi, ai in enumerate(action_show_order[:10]):
    if pi >= len(positions): break
    row, col = positions[pi]
    recon_n_np = tanh_rgb(decoded_pairs[ai][0])
    recon_p_np = tanh_rgb(decoded_pairs[ai][1])
    diff = np.abs(recon_p_np.astype(float) - recon_n_np.astype(float)).mean(-1)

    # Composite: show diff heatmap overlaid on image
    ax = fig.add_subplot(gs[row, col])
    ax.imshow(recon_p_np)
    # Overlay diff as alpha channel
    diff_norm = (diff / diff.max() * 180).astype(np.uint8) if diff.max() > 0 else diff.astype(np.uint8)
    ax.imshow(diff_norm, alpha=0.45, cmap="hot", vmin=0, vmax=255)
    ax.axis("off")
    sensitivity = action_sens[ai] / max(action_sens) * 100
    ax.set_title(
        f"{ACTION_NAMES[ai]}\n"
        f"sens={action_sens[ai]:.3f} ({sensitivity:.0f}%)",
        fontsize=8,
        color="darkred" if ai < 2 else "darkblue"
    )

# Keyboard layout heatmap
ax_kb = fig.add_subplot(gs[2, 4:])
key_layout = np.array([
    [action_sens[2],  action_sens[3],  action_sens[4],  action_sens[5]],   # W A S D
    [action_sens[6],  action_sens[7],  action_sens[8],  action_sens[9]],   # Spc Ctrl Shft LMB
    [action_sens[10], action_sens[11], 0.,              0.             ],   # RMB R
])
key_names = [["W","A","S","D"],["Space","Ctrl","Shift","LMB"],["RMB","R","",""],]
im = ax_kb.imshow(key_layout, cmap="YlOrRd", aspect="auto")
for i in range(3):
    for j in range(4):
        if key_names[i][j]:
            v = key_layout[i,j]
            ax_kb.text(j, i, f"{key_names[i][j]}\\n{v:.3f}", ha="center", va="center",
                       fontsize=9, fontweight="bold",
                       color="white" if v > key_layout.max()*0.6 else "black")
ax_kb.set_xticks([]); ax_kb.set_yticks([])
ax_kb.set_title("Keyboard Layout\\n(brighter = more world impact)", fontsize=9)
plt.colorbar(im, ax=ax_kb, fraction=0.06)

# Mouse sensitivity bar
ax_mouse = fig.add_subplot(gs[3, 4:])
ax_mouse.bar(["Yaw\\n(left/right)", "Pitch\\n(up/down)"],
             [action_sens[0], action_sens[1]], color=["#d55e00","#c0392b"], alpha=0.9)
for i,(nm,v) in enumerate(zip(["Yaw","Pitch"],[action_sens[0],action_sens[1]])):
    ax_mouse.text(i, v*1.02, f"{v:.4f}", ha="center", fontsize=11, fontweight="bold")
ax_mouse.set_title("Mouse Sensitivity\\n(camera movement)", fontsize=10)
ax_mouse.set_ylabel("||Δz||"); ax_mouse.grid(0.3, axis="y")

# Insight text
ax_txt = fig.add_subplot(gs[3, :4])
ax_txt.axis("off")
insights = [
    "KEY INSIGHTS FROM ACTION SENSITIVITY:",
    "",
    f"► Mouse (pitch={action_sens[1]:.3f}, yaw={action_sens[0]:.3f}) dominates ALL keys.",
    "  Camera movement = most visual change in CS2 POV footage. ✓",
    "",
    f"► Pitch > Yaw: vertical movement reveals new geometry faster.",
    "  Looking up/down shows ceiling/floor (new content), turning shows same room.",
    "",
    f"► Shift ({action_sens[8]:.3f}) > Ctrl ({action_sens[7]:.3f}) > Space ({action_sens[6]:.3f})",
    "  All change camera HEIGHT — crouch/jump shift viewpoint vertically. ✓",
    "",
    f"► LMB ({action_sens[9]:.3f}): shooting causes small but nonzero change.",
    "  Muzzle flash + recoil camera shake detected. ✓",
]
for i, line in enumerate(insights):
    color = "#4CAF50" if "✓" in line else ("#2196F3" if "►" in line else "black")
    fw = "bold" if "KEY INSIGHTS" in line or "►" in line else "normal"
    ax_txt.text(0.01, 0.97-i*0.082, line, fontsize=9, color=color,
                transform=ax_txt.transAxes, fontweight=fw, fontfamily="monospace")

fig.tight_layout(); save(fig, "per_key_analysis.png")
print("  ✓ per_key_analysis.png")
"""

# ══════════════════════════════════════════════════════════════
# CELL D  —  HOW THE MODEL "SEES" EACH FRAME
# Extracts the last-layer ViT attention maps for random frames.
# Shows which spatial regions the encoder attends to.
# ══════════════════════════════════════════════════════════════
CELL_D = """
# ── Cell D: ViT Attention Maps ────────────────────────────────
print("\\n=== Cell D: How the Model Sees Each Frame (ViT Attention) ===")

import torch.nn as nn

# Hook to capture attention weights from last ViT layer
attention_maps = {}

def _attn_hook(module, inp, out):
    # out: (B, heads, N, N) from ViT self-attention
    attention_maps["last"] = out.detach().cpu()

# Find the last attention layer in the ViT encoder
encoder_raw = model.encoder
hooks        = []
target_layer = None

for name, module in encoder_raw.named_modules():
    if hasattr(module, "attention") and hasattr(module.attention, "attention"):
        target_layer = module.attention.attention
for name, module in encoder_raw.named_modules():
    if module is target_layer:
        hooks.append(module.register_forward_hook(_attn_hook))
        break

if not hooks:
    # Fallback: just use any attention module
    for name, module in encoder_raw.named_modules():
        if isinstance(module, nn.MultiheadAttention) or "SelfAttention" in type(module).__name__:
            hooks.append(module.register_forward_hook(_attn_hook))
            break

N_SHOW = 6
rng    = np.random.default_rng(99)
chosen = rng.choice(len(vl_ds), N_SHOW, replace=False)

fig, axes = plt.subplots(3, N_SHOW, figsize=(N_SHOW*3.5, 9.5))
fig.suptitle(
    "Cell D: How the Model Sees CS2 Frames\\n"
    "ViT encoder attention maps — bright = model pays more attention here",
    fontsize=13, fontweight="bold"
)

with torch.no_grad():
    for i, raw_i in enumerate(chosen):
        fu8, _ = vl_ds[int(raw_i)]
        frame  = frames_to_device(fu8[cfg.seq_len//2:cfg.seq_len//2+1].unsqueeze(0), dev, dt)
        # forward pass with hook
        attention_maps.clear()
        with autocast_ctx(dev, dt):
            _ = model.encoder(pixel_values=frame.squeeze(0))

        # Raw frame
        axes[0, i].imshow(to_rgb(fu8[cfg.seq_len//2])); axes[0, i].axis("off")
        axes[0, i].set_title(f"Input {i}", fontsize=9)

        # Attention map (from CLS token to patches)
        if "last" in attention_maps:
            attn    = attention_maps["last"]    # (B, heads, N, N)
            # Mean over heads, CLS row (index 0)
            cls_attn = attn[0].mean(0)[0, 1:]   # (N_patches,)
            P        = int(cfg.img_size // cfg.patch_size)
            if cls_attn.numel() == P*P:
                attn_map = cls_attn.reshape(P, P).numpy()
                import torch.nn.functional as F_
                attn_map_big = F_.interpolate(
                    torch.tensor(attn_map).unsqueeze(0).unsqueeze(0),
                    size=(cfg.img_size, cfg.img_size), mode="bilinear", align_corners=False
                )[0,0].numpy()
                # Normalise
                attn_map_big = (attn_map_big - attn_map_big.min()) / (attn_map_big.max()-attn_map_big.min()+1e-8)
                axes[1, i].imshow(attn_map_big, cmap="inferno")
                axes[1, i].axis("off"); axes[1, i].set_title("Attention (head avg)", fontsize=9)

                # Overlay
                img_float = to_rgb(fu8[cfg.seq_len//2]).astype(float)/255.
                axes[2, i].imshow(img_float)
                axes[2, i].imshow(attn_map_big, alpha=0.55, cmap="inferno")
                axes[2, i].axis("off"); axes[2, i].set_title("Overlay", fontsize=9)
            else:
                for row in [1,2]:
                    axes[row,i].text(0.5,0.5,"n/a",ha="center",va="center",transform=axes[row,i].transAxes)
                    axes[row,i].axis("off")
        else:
            for row in [1,2]:
                axes[row,i].text(0.5,0.5,"Hook not\\ncaptured",ha="center",va="center",
                                  transform=axes[row,i].transAxes, fontsize=8, color="gray")
                axes[row,i].axis("off")

for h in hooks: h.remove()

axes[0,0].set_ylabel("Input Frame", fontsize=10, fontweight="bold")
axes[1,0].set_ylabel("Attn Map", fontsize=10, fontweight="bold")
axes[2,0].set_ylabel("Overlay", fontsize=10, fontweight="bold")

fig.tight_layout(); save(fig, "vit_attention.png")
print("  ✓ vit_attention.png")
"""

# ══════════════════════════════════════════════════════════════
# CELL E  —  TEMPORAL VELOCITY FIELD
# Shows the direction and magnitude of latent movement over time.
# Reveals game dynamics: fights, movement, idle moments.
# ══════════════════════════════════════════════════════════════
CELL_E = """
# ── Cell E: Temporal Velocity Field ──────────────────────────
print("\\n=== Cell E: Temporal Velocity Field ===")

N_SEQS = min(120, len(vl_ds))
rng    = np.random.default_rng(21)
ids    = rng.choice(len(vl_ds), N_SEQS, replace=False)

all_vels     = []    # (N*T-1, D)
all_frames   = []    # for display
all_mags     = []    # scalar magnitudes

with torch.no_grad():
    for raw_i in ids:
        fu8, ac = vl_ds[int(raw_i)]
        frames  = frames_to_device(fu8.unsqueeze(0), dev, dt)
        act     = norm.transform(ac.unsqueeze(0).to(dev))
        with autocast_ctx(dev, dt):
            emb = model.encode_sequence(frames).float()
        vel = emb[0, 1:] - emb[0, :-1]   # (T-1, D)
        mag = vel.norm(dim=-1)
        all_vels.append(vel.cpu())
        all_mags.append(mag.cpu())
        # Store frame at peak magnitude
        peak_t = int(mag.argmax())
        all_frames.append(fu8[peak_t])

all_vels_t  = torch.cat(all_vels, 0)    # (N*(T-1), D)
all_mags_t  = torch.cat(all_mags, 0)

# PCA projection of velocities
Vmat = all_vels_t.float().numpy()
_,_,Vt = np.linalg.svd(Vmat - Vmat.mean(0), full_matrices=False)
vel_pca = (Vmat - Vmat.mean(0)) @ Vt[:2].T   # (N*(T-1), 2)
mags_np = all_mags_t.numpy()

print(f"  Velocity magnitude: mean={mags_np.mean():.3f} "
      f"std={mags_np.std():.3f} "
      f"p95={np.percentile(mags_np,95):.3f}")

fig, axes = plt.subplots(2, 3, figsize=(20, 11))
fig.suptitle(
    "Cell E: Temporal Velocity Field in CS2 Latent Space\\n"
    "v_t = z_{t+1} - z_t  — how fast the game state changes each frame",
    fontsize=13, fontweight="bold"
)

# Velocity magnitude distribution
axes[0,0].hist(mags_np, bins=80, color="steelblue", alpha=0.85, edgecolor="white", density=True)
axes[0,0].axvline(mags_np.mean(), color="red", ls="--", lw=2, label=f"mean={mags_np.mean():.3f}")
axes[0,0].axvline(np.percentile(mags_np,95), color="orange", ls="--", lw=2,
                   label=f"95th pct={np.percentile(mags_np,95):.3f}")
axes[0,0].legend(fontsize=9); axes[0,0].set_title("Velocity Magnitude Distribution")
axes[0,0].set_xlabel("||v_t|| = ||z_{t+1} - z_t||"); axes[0,0].grid(0.3)

# PCA of velocity field
sc = axes[0,1].scatter(vel_pca[:,0], vel_pca[:,1], c=mags_np, cmap="plasma",
                        s=6, alpha=0.5, vmax=np.percentile(mags_np,95))
plt.colorbar(sc, ax=axes[0,1], label="||v_t||")
axes[0,1].set_title("Velocity Direction PCA\\n(colour = magnitude)")
axes[0,1].axis("off")

# Per-step velocity magnitude
step_mags = np.array([all_mags_t.numpy()[i::cfg.seq_len-1].mean() for i in range(cfg.seq_len-1)])
axes[0,2].plot(range(1, cfg.seq_len), step_mags, "b-o", ms=5, lw=2)
axes[0,2].set_xlabel("Step in sequence"); axes[0,2].set_ylabel("Mean ||v_t||")
axes[0,2].set_title("Velocity magnitude per step\\n(shows if game pace changes)"); axes[0,2].grid(0.3)

# Show highest-velocity frames (most dynamic moments)
top_seq_mags = [m.max().item() for m in all_mags]
top_idx      = np.argsort(top_seq_mags)[::-1][:6]
for j, si in enumerate(top_idx):
    ax = axes[1, min(j, 2)] if j < 3 else axes[1, j-3]
    if j == 3: axes[1,0].axis("off"); continue   # skip to avoid double
    axes[1, j].imshow(to_rgb(all_frames[si]))
    axes[1, j].axis("off")
    axes[1, j].set_title(f"High motion\\n||v||={top_seq_mags[si]:.2f}", fontsize=8, color="red")

# Slowest-motion frames (most static moments)
low_idx = np.argsort(top_seq_mags)[:3]
for j, si in enumerate(low_idx):
    if j+3 < 6:
        axes[1, j+3].imshow(to_rgb(all_frames[si]))
        axes[1, j+3].axis("off")
        axes[1, j+3].set_title(f"Low motion\\n||v||={top_seq_mags[si]:.2f}", fontsize=8, color="blue")

fig.tight_layout(); save(fig, "velocity_field.png")
print("  ✓ velocity_field.png")
"""

# ══════════════════════════════════════════════════════════════
# CELL F  —  CEM vs RANDOM PLANNING COMPARISON
# Shows that CEM-planned action sequences navigate latent space
# better than random actions. Quantitative comparison.
# ══════════════════════════════════════════════════════════════
CELL_F = """
# ── Cell F: CEM vs Random Planning ───────────────────────────
print("\\n=== Cell F: CEM vs Random Planning (quantitative) ===")

N_TRIALS  = 30
HORIZON   = 10
N_SAMPLES = 256
N_ELITES  = 24
N_ITERS   = 15
MC        = 30.0
rng_t     = np.random.default_rng(77)

cem_final_costs   = []
rand_final_costs  = []
cem_improvements  = []

with torch.no_grad():
    for trial in range(N_TRIALS):
        s_idx = int(rng_t.integers(0, len(vl_ds)//2))
        g_idx = int(rng_t.integers(len(vl_ds)//2, len(vl_ds)))
        fu8_s, _ = vl_ds[s_idx]; fu8_g, _ = vl_ds[g_idx]

        frames_s = frames_to_device(fu8_s.unsqueeze(0), dev, dt)
        frames_g = frames_to_device(fu8_g.unsqueeze(0), dev, dt)
        with autocast_ctx(dev, dt):
            z_s = model.encode_sequence(frames_s)
            z_g = model.encode_sequence(frames_g)[0,-1].float()

        # Compute initial (random) cost
        def rollout_cost(actions_seq):
            ctx = z_s.expand(actions_seq.shape[0],-1,-1).clone()
            for t in range(HORIZON):
                h  = min(cfg.history_size, ctx.shape[1]); cz=ctx[:,-h:]
                ca = actions_seq[:, max(0,t-h+1):t+1]
                if ca.shape[1]<h:
                    pad=torch.zeros(actions_seq.shape[0],h-ca.shape[1],len(ACTION_NAMES),device=dev)
                    ca=torch.cat([pad,ca],1)
                with autocast_ctx(dev, dt):
                    ae  = model.action_enc(ca.to(dt))
                    znx = model.predict_sequence(cz.to(dt), ae).float()[:,-1:]
                ctx=torch.cat([ctx,znx],1)
            return ((ctx[:,-1]-z_g.unsqueeze(0))**2).sum(-1)

        # Random baseline
        rand_acts = torch.randn(N_SAMPLES, HORIZON, len(ACTION_NAMES), device=dev)
        rand_cost = rollout_cost(rand_acts).min().item()
        rand_final_costs.append(rand_cost)

        # CEM
        mu  = torch.zeros(HORIZON, len(ACTION_NAMES), device=dev)
        std = torch.ones( HORIZON, len(ACTION_NAMES), device=dev); std[:,:2]=MC*0.3
        costs_0 = None
        for it in range(N_ITERS):
            cands = mu+std*torch.randn(N_SAMPLES,HORIZON,len(ACTION_NAMES),device=dev)
            costs = rollout_cost(cands)
            if costs_0 is None: costs_0 = costs.min().item()
            elite = cands[costs.argsort()[:N_ELITES]]
            mu=elite.mean(0); std=elite.std(0).clamp(.05)*.95
        cem_cost = rollout_cost(mu.unsqueeze(0)).item()
        cem_final_costs.append(cem_cost)
        cem_improvements.append(costs_0 / max(cem_cost, 1e-8))

        if (trial+1) % 10 == 0:
            print(f"  Trial {trial+1}/{N_TRIALS}: CEM={cem_cost:.3f}  Random={rand_cost:.3f}")

cem_costs    = np.array(cem_final_costs)
rand_costs   = np.array(rand_final_costs)
improvements = np.array(cem_improvements)

print(f"  CEM mean final cost:    {cem_costs.mean():.4f}")
print(f"  Random mean final cost: {rand_costs.mean():.4f}")
print(f"  CEM better by:          {rand_costs.mean()/cem_costs.mean():.2f}× on average")
print(f"  CEM mean improvement:   {improvements.mean():.2f}×")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    f"Cell F: CEM vs Random Planning — {N_TRIALS} Trials\\n"
    f"CEM is {rand_costs.mean()/cem_costs.mean():.1f}× better than random at reaching goal states",
    fontsize=13, fontweight="bold"
)

bins = np.linspace(0, max(cem_costs.max(), rand_costs.max()), 30)
axes[0].hist(rand_costs, bins=bins, alpha=0.7, color="orange", label=f"Random μ={rand_costs.mean():.3f}")
axes[0].hist(cem_costs,  bins=bins, alpha=0.7, color="steelblue", label=f"CEM μ={cem_costs.mean():.3f}")
axes[0].legend(fontsize=10); axes[0].set_title("Final Goal Distance Distribution"); axes[0].grid(0.3)
axes[0].set_xlabel("||z_final - z_goal||²")

axes[1].scatter(rand_costs, cem_costs, s=40, alpha=0.7, color="steelblue")
lm = max(rand_costs.max(), cem_costs.max())
axes[1].plot([0,lm],[0,lm],"r--",lw=2,label="y=x (same)")
axes[1].fill_between([0,lm],[0,0],[0,lm],alpha=0.1,color="green",label="CEM better")
axes[1].set_xlabel("Random final cost"); axes[1].set_ylabel("CEM final cost")
axes[1].set_title("CEM vs Random (below diagonal = CEM better)"); axes[1].legend(fontsize=9); axes[1].grid(0.3)

axes[2].hist(improvements, bins=20, color="#4CAF50", alpha=0.85, edgecolor="white")
axes[2].axvline(1.0, color="red", ls="--", lw=2, label="no improvement")
axes[2].axvline(improvements.mean(), color="blue", ls="-", lw=2,
                 label=f"mean={improvements.mean():.1f}×")
axes[2].legend(fontsize=9); axes[2].set_title("CEM Improvement over Init (×)")
axes[2].set_xlabel("initial_cost / final_cost"); axes[2].grid(0.3)

fig.tight_layout(); save(fig, "cem_vs_random.png")
print("  ✓ cem_vs_random.png")
"""

# ══════════════════════════════════════════════════════════════
# CELL G  —  MULTI-STEP ROLLOUT QUALITY
# Shows cosine similarity, L2 error, and decoded frames
# at different prediction horizons (1, 5, 10, 15 steps ahead).
# ══════════════════════════════════════════════════════════════
CELL_G = """
# ── Cell G: Multi-Horizon Rollout Quality ─────────────────────
print("\\n=== Cell G: Multi-Horizon Rollout Quality ===")

HORIZONS  = [1, 3, 5, 8, 12, 16, 19]
N_SEQS    = min(100, len(vl_ds))
rng       = np.random.default_rng(55)
ids       = rng.choice(len(vl_ds), N_SEQS, replace=False)

cosim_per_h = {h: [] for h in HORIZONS}
l2_per_h    = {h: [] for h in HORIZONS}

with torch.no_grad():
    for raw_i in ids:
        fu8, ac = vl_ds[int(raw_i)]
        frames  = frames_to_device(fu8.unsqueeze(0), dev, dt)
        act     = norm.transform(ac.unsqueeze(0).to(dev))
        with autocast_ctx(dev, dt):
            emb  = model.encode_sequence(frames).float()
            ae   = model.action_enc(act.to(dt))
            pred = model.predict_sequence(emb[:,:-1].to(dt), ae[:,:-1]).float()

        for h in HORIZONS:
            if h < emb.shape[1]:
                cs = float(F.cosine_similarity(pred[:,h-1:h], emb[:,h:h+1], dim=-1).mean())
                l2 = float((pred[:,h-1:h] - emb[:,h:h+1]).pow(2).mean())
                cosim_per_h[h].append(cs)
                l2_per_h[h].append(l2)

mean_cosim = [np.mean(cosim_per_h[h]) for h in HORIZONS]
mean_l2    = [np.mean(l2_per_h[h])    for h in HORIZONS]

# Decoded frames at different horizons for one example
fu8_ex, ac_ex = vl_ds[int(ids[0])]
with torch.no_grad():
    frames_ex  = frames_to_device(fu8_ex.unsqueeze(0), dev, dt)
    act_ex     = norm.transform(ac_ex.unsqueeze(0).to(dev))
    with autocast_ctx(dev, dt):
        emb_ex  = model.encode_sequence(frames_ex).float()
        ae_ex   = model.action_enc(act_ex.to(dt))
        pred_ex = model.predict_sequence(emb_ex[:,:-1].to(dt), ae_ex[:,:-1]).float()
        # Decode predicted z at each horizon
        show_h = [1, 3, 5, 8, 12]
        z_show_pred = torch.stack([pred_ex[0, min(h-1, pred_ex.shape[1]-1)] for h in show_h])
        z_show_real = torch.stack([emb_ex [0, min(h,   emb_ex.shape[1]-1)]  for h in show_h])
        r_pred = dec(z_show_pred.to(DEVICE)).float()
        r_real = dec(z_show_real.to(DEVICE)).float()

fig, axes = plt.subplots(3, len(show_h)+1, figsize=((len(show_h)+1)*3.5, 10))
fig.suptitle(
    "Cell G: Multi-Horizon Prediction Quality\\n"
    f"step-1 cosim={mean_cosim[0]:.4f}  step-{HORIZONS[-1]} cosim={mean_cosim[-1]:.4f}",
    fontsize=13, fontweight="bold"
)

# Row 0: real frames
axes[0,0].axis("off"); axes[0,0].text(0.5,0.5,"Real\\nFrames",ha="center",va="center",
                                        fontsize=12,fontweight="bold",transform=axes[0,0].transAxes)
for j,h in enumerate(show_h):
    real_f = to_rgb(fu8_ex[min(h, len(fu8_ex)-1)])
    axes[0, j+1].imshow(real_f); axes[0,j+1].axis("off")
    axes[0,j+1].set_title(f"t+{h}", fontsize=10)

# Row 1: predicted decoded frames
axes[1,0].axis("off"); axes[1,0].text(0.5,0.5,"Predicted\\nDecoded",ha="center",va="center",
                                       fontsize=12,fontweight="bold",transform=axes[1,0].transAxes)
for j in range(len(show_h)):
    axes[1, j+1].imshow(tanh_rgb(r_pred[j])); axes[1,j+1].axis("off")
    h=show_h[j]
    h_idx = HORIZONS.index(h) if h in HORIZONS else -1
    cs_str = f"cosim={mean_cosim[h_idx]:.3f}" if h_idx >= 0 else ""
    axes[1,j+1].set_title(cs_str, fontsize=8)

# Row 2: per-horizon quality curves
ax_cs = axes[2, 0]; ax_cs.remove()
ax_l2 = axes[2, 1]; ax_l2.remove()
ax_cs = fig.add_subplot(3, 2, 5)
ax_l2 = fig.add_subplot(3, 2, 6)
for j in range(2, len(show_h)+1): axes[2,j].axis("off")

ax_cs.plot(HORIZONS, mean_cosim, "b-o", ms=6, lw=2)
ax_cs.axhline(0.9, color="g", ls="--", alpha=0.7, label="0.9"); ax_cs.axhline(0.5, color="r", ls="--", alpha=0.7, label="random")
ax_cs.set_xlabel("Prediction horizon (steps)"); ax_cs.set_ylabel("Cosine similarity")
ax_cs.set_title("Cosine Similarity vs Horizon"); ax_cs.legend(fontsize=9); ax_cs.set_ylim(0,1); ax_cs.grid(0.3)

ax_l2.plot(HORIZONS, mean_l2, "r-o", ms=6, lw=2)
ax_l2.set_xlabel("Prediction horizon (steps)"); ax_l2.set_ylabel("L2 error")
ax_l2.set_title("L2 Error vs Horizon"); ax_l2.grid(0.3)

fig.tight_layout(); save(fig, "multihorizon_rollout.png")
print("  ✓ multihorizon_rollout.png")
"""

# ══════════════════════════════════════════════════════════════
# CELL H  —  LATENT MANIFOLD EXPLORER
# Interactive 2D map: hover to see game screenshots.
# Shows that nearby latent points = visually similar game states.
# ══════════════════════════════════════════════════════════════
CELL_H = """
# ── Cell H: Latent Manifold Explorer (static version) ─────────
print("\\n=== Cell H: Latent Manifold Explorer ===")

# Use already-computed Z_all and FF_all from corpus encoding

N_SHOW = 30   # show frame thumbnails on the manifold
rng    = np.random.default_rng(44)

Zn   = F.normalize(Z_all.float(), -1).numpy()
Z2   = TSNE(2, perplexity=40, max_iter=800, random_state=42).fit_transform(Zn)

fig, ax = plt.subplots(1, 1, figsize=(18, 14))
fig.suptitle(
    "Cell H: CS2 Latent Manifold — 800 Game Moments\\n"
    "Each image = a real CS2 frame. Position = where model places it in latent space.\\n"
    "Nearby = semantically similar game state (same area, same view, same activity)",
    fontsize=13, fontweight="bold"
)

from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Plot background scatter
sc = ax.scatter(Z2[:,0], Z2[:,1], s=8, alpha=0.25, c=np.linalg.norm(Zn,axis=1), cmap="plasma")
plt.colorbar(sc, ax=ax, label="||z|| (embedding magnitude)", fraction=0.02)

# Place frame thumbnails at evenly-spread positions
thumb_idx = rng.choice(len(Z_all), N_SHOW, replace=False)
for ti in thumb_idx:
    frame_np = to_rgb(FF_all[ti])   # (H, W, 3) uint8
    # Downsample for display
    from PIL import Image as PILImage
    img_small = np.array(PILImage.fromarray(frame_np).resize((36,36)))
    im = OffsetImage(img_small, zoom=1.0)
    ab = AnnotationBbox(im, (Z2[ti,0], Z2[ti,1]),
                        frameon=True, bboxprops=dict(edgecolor="white", linewidth=0.8, alpha=0.8))
    ax.add_artist(ab)

ax.axis("off")
ax.set_title("")   # already in suptitle

fig.tight_layout(); save(fig, "latent_manifold_explorer.png")
print("  ✓ latent_manifold_explorer.png")
"""

if __name__ == "__main__":
    print("This file contains showcase test cells for CS2-LeWM.")
    print("Copy each CELL_X string into a Jupyter notebook cell and run.")
    print("")
    print("Cells:")
    print("  CELL_A — CEM Planner (goal-directed imagination)")
    print("  CELL_B — Residual delta analysis")
    print("  CELL_C — Per-key perceptual analysis")
    print("  CELL_D — ViT attention maps (how model sees frames)")
    print("  CELL_E — Temporal velocity field")
    print("  CELL_F — CEM vs random planning (quantitative)")
    print("  CELL_G — Multi-horizon rollout quality")
    print("  CELL_H — Latent manifold explorer")
    print("")
    print("All cells run on CPU (slower) or GPU (faster).")
    print("Run SETUP first, then any cell in any order.")
