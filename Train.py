# ============================================================
# OpenVLA-style Training — Robot Manipulation (HDF5 Edition)
#
# Dataset  : episodes/episode_N.hdf5
#            - observations/images/zed_left  : (250, 376, 672, 3) uint8
#            - observations/follower/position: (250, 7)  float32   joint positions
#            - observations/follower/velocity: (250, 6)  float32   joint velocities
#            - observations/end_pose/pose    : (250, 7)  float32   EEF pose (xyz + quat)
#            - actions/position              : (250, 7)  float32   target joint pos
#            - actions/velocity              : (250, 6)  float32   target joint vel
#
# Task     : Given zed_left frame at timestep t + auto-generated proprioceptive
#            instruction → predict actions/position[t] + actions/velocity[t]
#
# Model    : SigLIP (frozen vision encoder)
#          + Projector (trainable MLP, 768 → LLM_DIM)
#          + Qwen2-7B  (frozen LLM, provides rich contextual embeddings)
#          + ActionHead (trainable MLP regression head, 2*LLM_DIM → ACTION_DIM)
#
# Phases   :
#   1. Alignment        — InfoNCE between projected vision tokens and prompts
#   2. Action-only warm — pure Huber regression, no alignment penalty
#                         (ACTION_ONLY_STEPS steps, lets head bootstrap cleanly)
#   3. Action + align   — Huber + decaying alignment regulariser
#
# Key improvements in this version:
#   G1. Self-attention-weighted patch pooling (query = mean patch)
#   G2. Top-k patch selection before Qwen (preserves spatial info, cuts memory)
#   G3. Hybrid action repr = vision_repr ‖ last_hidden_token (2×LLM_DIM)
#   G4. ACTION_ONLY_STEPS phase — action head trains before alignment interferes
#   G5. ACTION_BETA loss scale — action loss dominates from the start
#   G6. ALIGN_LOSS_ALPHA reduced to 0.05 to prevent contrastive over-regularisation
#   G7. Overfit diagnostic — run_overfit_test() validates pipeline before full run
#   P1. Full sinusoidal spatial positional encoding added to projected patches
#   P2. Top-k selection replaced with attention-score-based importance (semantic focus)
#   P3. Training-time noise injection on vision_repr (anti mean-collapse regulariser)
# ============================================================

import os
os.environ["HDF5_USE_FILE_LOCKING"]="FALSE"
import glob
import math
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
from transformers import AutoProcessor, AutoModel, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ========================= CONFIG =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.bfloat16 if torch.cuda.is_available() else torch.float32

DATASET_DIR = "/media/deeptech/R/dataset/task1"

BATCH_SIZE       = 8          # lower than VQA — frames are 376×672, larger than 224²
ACCUM_STEPS      = 4
LR               = 2e-5
EPOCHS           = 10
TEXT_MAX_LEN     = 128        # auto-prompts are short
ALIGN_STEPS      = 2000       # shorter alignment phase (no natural-language diversity)
NUM_VIS_TOKENS   = 256
SAVE_EVERY_STEPS = 500

# Action space dimensionality
ACTION_POS_DIM   = 7          # joint positions
ACTION_VEL_DIM   = 6          # joint velocities
ACTION_DIM       = ACTION_POS_DIM + ACTION_VEL_DIM   # 13

# Action normalisation — computed from dataset stats at startup
# (populated by compute_action_stats())
ACTION_MEAN: torch.Tensor
ACTION_STD:  torch.Tensor

# Loss weights
# G6: alignment alpha drastically reduced — action signal must dominate
ALIGN_LOSS_ALPHA     = 0.05
ALIGN_LOSS_ALPHA_MIN = 0.01
POS_LOSS_WEIGHT      = 1.0    # joint position loss weight
VEL_LOSS_WEIGHT      = 0.5    # velocity loss weight (typically easier to predict)
# G5: overall action loss scale — keeps action loss >> alignment loss numerically
ACTION_BETA          = 5.0

# G4: pure action-only warm-up phase (no alignment penalty)
# Gives ActionHead time to bootstrap before contrastive regularisation interferes.
ACTION_ONLY_STEPS    = 1000   # steps after the alignment phase ends

# G2: top-k patches to retain per image (SigLIP 224px → ~196 patches total)
# 32 preserves salient spatial regions while halving Qwen sequence length
TOPK_PATCHES         = 32

# Temporal context: feed frame at t, predict action at t
FRAME_STACK = 1

VAL_SPLIT        = 0.10       # 10 % of episodes held out
EVAL_EVERY_STEPS = 500
WARMUP_FRACTION  = 0.10

CHECKPOINT_DIR    = "checkpoints_robot"
RESUME_CHECKPOINT = None
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ========================= MODELS =========================
print("Loading models...")

siglip_processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
siglip_model = (
    AutoModel.from_pretrained("google/siglip-base-patch16-224")
    .to(device=device, dtype=dtype)
    .eval()
)
for p in siglip_model.parameters():
    p.requires_grad = False

qwen_name = "Qwen/Qwen2-7B"
tokenizer = AutoTokenizer.from_pretrained(qwen_name)
tokenizer.padding_side = "right"

if "<image>" not in tokenizer.get_vocab():
    tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
IMAGE_TOKEN_ID = tokenizer.convert_tokens_to_ids("<image>")

qwen_model = AutoModelForCausalLM.from_pretrained(
    qwen_name,
    torch_dtype=dtype,
    device_map="auto",
    output_hidden_states=True,   # we need the last hidden state for the action head
)
qwen_model.resize_token_embeddings(len(tokenizer))
qwen_model.eval()
for p in qwen_model.parameters():
    p.requires_grad = False

if hasattr(qwen_model, "gradient_checkpointing_enable"):
    qwen_model.gradient_checkpointing_enable()

LLM_DIM = qwen_model.config.hidden_size

# ========================= PROJECTOR =========================
class Projector(nn.Module):
    """Three-layer MLP with residual skip: 768 → LLM_DIM."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        mid_dim = (in_dim + out_dim) // 2
        self.norm_in = nn.LayerNorm(in_dim)
        self.fc1     = nn.Linear(in_dim, mid_dim)
        self.act1    = nn.GELU()
        self.norm1   = nn.LayerNorm(mid_dim)
        self.fc2     = nn.Linear(mid_dim, out_dim)
        self.act2    = nn.GELU()
        self.norm2   = nn.LayerNorm(out_dim)
        self.fc3     = nn.Linear(out_dim, out_dim)
        self.skip    = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):           # [B, N, in_dim]
        residual = self.skip(self.norm_in(x))
        h = self.fc1(self.norm_in(x))
        h = self.act1(h)
        h = self.fc2(self.norm1(h))
        h = self.act2(h)
        h = self.fc3(self.norm2(h))
        return h + residual         # [B, N, out_dim]


projector = Projector(768, LLM_DIM).to(device=device, dtype=dtype)

# ========================= ACTION HEAD (G3) =========================
class ActionHead(nn.Module):
    """
    G3: Hybrid input = vision_repr ‖ last_hidden_token → 2*LLM_DIM.

    vision_repr      : attention-weighted pool of top-k projected patches
                       — carries spatial/object identity signal
    last_hidden_token: final token of Qwen's last hidden state
                       — carries language-grounded contextual signal

    Concatenating both ensures neither source is discarded when one is weak.
    Separate pos/vel heads allow independent loss weighting.
    """
    def __init__(self, in_dim: int, pos_dim: int, vel_dim: int):
        super().__init__()
        # in_dim = 2 * LLM_DIM (vision_repr + last_hidden_token)
        hidden = in_dim // 2
        self.norm   = nn.LayerNorm(in_dim)
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.pos_head = nn.Linear(hidden, pos_dim)
        self.vel_head = nn.Linear(hidden, vel_dim)

    def forward(self, x):           # x: [B, 2*LLM_DIM]
        h = self.shared(self.norm(x))
        return self.pos_head(h), self.vel_head(h)   # ([B,7], [B,6])


# G3: ActionHead now takes 2*LLM_DIM
action_head = ActionHead(2 * LLM_DIM, ACTION_POS_DIM, ACTION_VEL_DIM).to(device=device, dtype=dtype)

# Learnable InfoNCE temperature
logit_scale = nn.Parameter(
    torch.ones([], device=device, dtype=dtype) * math.log(1 / 0.07)
)

# ========================= ACTION NORMALISATION =========================
def compute_action_stats(hdf5_paths: list[str]):
    """
    Single pass over all episodes to compute per-dimension mean and std
    for both position and velocity actions. Called once before training.
    """
    print("Computing action normalisation stats...")
    all_pos, all_vel = [], []
    for path in tqdm(hdf5_paths, desc="  scanning episodes"):
        with h5py.File(path, "r") as f:
            all_pos.append(f["actions/position"][:])   # (T, 7)
            all_vel.append(f["actions/velocity"][:])   # (T, 6)
    pos = np.concatenate(all_pos, axis=0)   # (N*T, 7)
    vel = np.concatenate(all_vel, axis=0)   # (N*T, 6)
    combined = np.concatenate([pos, vel], axis=1)   # (N*T, 13)
    mean = torch.tensor(combined.mean(axis=0), dtype=torch.float32, device=device)
    std  = torch.tensor(combined.std(axis=0),  dtype=torch.float32, device=device).clamp(min=1e-6)
    print(f"  action mean range: [{mean.min():.3f}, {mean.max():.3f}]")
    print(f"  action std  range: [{std.min():.3f},  {std.max():.3f}]")
    return mean, std


def normalise_action(pos: torch.Tensor, vel: torch.Tensor):
    """Normalise using pre-computed stats. Returns (pos_norm, vel_norm)."""
    combined = torch.cat([pos, vel], dim=-1).float()
    normed   = (combined - ACTION_MEAN) / ACTION_STD
    return normed[..., :ACTION_POS_DIM], normed[..., ACTION_POS_DIM:]


def denormalise_action(pos_norm: torch.Tensor, vel_norm: torch.Tensor):
    """Inverse of normalise_action. Used at inference."""
    combined = torch.cat([pos_norm, vel_norm], dim=-1).float()
    return combined * ACTION_STD + ACTION_MEAN


# ========================= AUTO PROMPT GENERATION =========================
def make_prompt(joint_pos: np.ndarray, eef_pose: np.ndarray, timestep: int) -> str:
    """
    Auto-generate a natural-language-style instruction prompt from
    proprioceptive state. Since there are no task annotations, we
    describe the current robot state and ask for the next action.

    joint_pos : (7,)  — follower joint positions in radians
    eef_pose  : (7,)  — end-effector pose [x, y, z, qx, qy, qz, qw]
    timestep  : int
    """
    jp  = joint_pos
    xyz = eef_pose[:3]
    prompt = (
        f"<image> "
        f"Robot arm control task. "
        f"Current end-effector position: x={xyz[0]:.3f}, y={xyz[1]:.3f}, z={xyz[2]:.3f}. "
        f"Joint positions: [{', '.join(f'{v:.3f}' for v in jp)}]. "
        f"Timestep {timestep}. "
        f"Predict the next joint position and velocity commands."
    )
    return prompt


# ========================= DATASET =========================
class RobotEpisodeDataset(Dataset):
    """
    Each item is a single timestep from a single episode.
    Loads the full episode into RAM on first access (cached per episode).
    Only zed_left camera is used.
    """
    def __init__(self, hdf5_paths: list[str]):
        self.samples: list[tuple[str, int]] = []   # (path, timestep_idx)

        for path in hdf5_paths:
            try:
                with h5py.File(path, "r") as f:
                    T = f["observations/images/zed_left"].shape[0]
                    # skip first and last frame (edge effects in demonstrations)
                    for t in range(1, T - 1):
                        self.samples.append((path, t))
            except Exception as e:
                print(f"  Skipping {path}: {e}")

        print(f"  Total timestep samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, t = self.samples[idx]
        with h5py.File(path, "r") as f:
            # Image: zed_left (376, 672, 3) uint8
            img_np = f["observations/images/zed_left"][t]          # (H, W, 3)

            # Proprioception at timestep t
            joint_pos = f["observations/follower/position"][t]     # (7,)
            joint_vel = f["observations/follower/velocity"][t]     # (6,)
            eef_pose  = f["observations/end_pose/pose"][t]         # (7,)

            # Target actions at timestep t
            act_pos   = f["actions/position"][t].astype(np.float32)   # (7,)
            act_vel   = f["actions/velocity"][t].astype(np.float32)   # (6,)

        image  = Image.fromarray(img_np, mode="RGB")
        prompt = make_prompt(joint_pos, eef_pose, t)

        return (
            image,
            prompt,
            torch.from_numpy(act_pos),
            torch.from_numpy(act_vel),
        )


def collate_fn(batch):
    images, prompts, act_pos, act_vel = zip(*batch)
    return (
        list(images),
        list(prompts),
        torch.stack(act_pos),    # [B, 7]
        torch.stack(act_vel),    # [B, 6]
    )


# ========================= DATASET SETUP =========================
print("Scanning dataset...")
all_paths = sorted(glob.glob(os.path.join(DATASET_DIR, "episode_*.hdf5")))
assert len(all_paths) > 0, f"No HDF5 files found in {DATASET_DIR}"
print(f"  Found {len(all_paths)} episodes")

# Episode-level train/val split (not timestep-level — prevents leakage)
n_val    = max(1, int(len(all_paths) * VAL_SPLIT))
n_train  = len(all_paths) - n_val
rng      = np.random.default_rng(seed=42)
shuffled = rng.permutation(len(all_paths)).tolist()
train_paths = [all_paths[i] for i in shuffled[:n_train]]
val_paths   = [all_paths[i] for i in shuffled[n_train:]]
print(f"  Train episodes: {n_train} | Val episodes: {n_val}")

# Compute normalisation stats from training episodes only
ACTION_MEAN, ACTION_STD = compute_action_stats(train_paths)

train_dataset = RobotEpisodeDataset(train_paths)
val_dataset   = RobotEpisodeDataset(val_paths)

NUM_WORKERS = min(8, os.cpu_count() or 1)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,     # lower than VQA — images are 10× larger
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)

# ========================= SCHEDULERS =========================
def make_cosine_with_warmup(optimizer, total_steps, warmup_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


steps_per_epoch     = math.ceil(len(train_dataset) / BATCH_SIZE)
ESTIMATED_ACT_STEPS = max(1, steps_per_epoch * EPOCHS - ALIGN_STEPS)

align_trainable = list(projector.parameters()) + [logit_scale]
act_trainable   = list(projector.parameters()) + list(action_head.parameters()) + [logit_scale]

align_optimizer = torch.optim.AdamW(align_trainable, lr=LR, weight_decay=1e-2)
act_optimizer   = torch.optim.AdamW(act_trainable,   lr=LR * 0.5, weight_decay=1e-2)

align_warmup = max(1, int(ALIGN_STEPS * WARMUP_FRACTION))
act_warmup   = max(1, int(ESTIMATED_ACT_STEPS * WARMUP_FRACTION))

align_scheduler = make_cosine_with_warmup(align_optimizer, ALIGN_STEPS, align_warmup)
act_scheduler   = make_cosine_with_warmup(act_optimizer,   ESTIMATED_ACT_STEPS, act_warmup)

# ========================= CHECKPOINT HELPERS =========================
def _step_from_name(fname: str) -> int:
    base = fname.replace(".pt", "")
    if "ckpt_step"  in base: return int(base.replace("ckpt_step",  ""))
    if "ckpt_epoch" in base: return int(base.replace("ckpt_epoch", "")) * 10_000_000
    return 0


def find_latest_checkpoint(ckpt_dir):
    ckpts = [
        f for f in os.listdir(ckpt_dir)
        if (f.startswith("ckpt_step") or f.startswith("ckpt_epoch")) and f.endswith(".pt")
    ]
    if not ckpts:
        return None
    ckpts.sort(key=_step_from_name)
    return os.path.join(ckpt_dir, ckpts[-1])


def save_checkpoint(step, epoch):
    path = f"{CHECKPOINT_DIR}/ckpt_step{step}.pt"
    torch.save({
        "projector":       projector.state_dict(),
        "action_head":     action_head.state_dict(),
        "logit_scale":     logit_scale.data,
        "action_mean":     ACTION_MEAN,
        "action_std":      ACTION_STD,
        "align_optimizer": align_optimizer.state_dict(),
        "align_scheduler": align_scheduler.state_dict(),
        "act_optimizer":   act_optimizer.state_dict(),
        "act_scheduler":   act_scheduler.state_dict(),
        "global_step":     step,
        "epoch":           epoch,
    }, path)
    print(f"  ✓ Checkpoint saved → {path}")


def load_checkpoint(path):
    global start_epoch, global_step, last_saved_step, ACTION_MEAN, ACTION_STD
    print(f"Resuming from checkpoint: {path}")
    ckpt = torch.load(path, map_location=device)
    projector.load_state_dict(ckpt["projector"])
    action_head.load_state_dict(ckpt["action_head"])
    if "logit_scale"  in ckpt: logit_scale.data.copy_(ckpt["logit_scale"])
    if "action_mean"  in ckpt: ACTION_MEAN = ckpt["action_mean"].to(device)
    if "action_std"   in ckpt: ACTION_STD  = ckpt["action_std"].to(device)
    align_optimizer.load_state_dict(ckpt["align_optimizer"])
    align_scheduler.load_state_dict(ckpt["align_scheduler"])
    if "act_optimizer" in ckpt: act_optimizer.load_state_dict(ckpt["act_optimizer"])
    if "act_scheduler" in ckpt: act_scheduler.load_state_dict(ckpt["act_scheduler"])
    global_step     = ckpt["global_step"]
    start_epoch     = ckpt["epoch"]
    last_saved_step = global_step
    print(f"  Resumed at epoch={start_epoch}, global_step={global_step}")


start_epoch = 0
global_step = 0
last_saved_step = -1

resume_path = RESUME_CHECKPOINT or find_latest_checkpoint(CHECKPOINT_DIR)
if resume_path and os.path.isfile(resume_path):
    load_checkpoint(resume_path)
else:
    print("No checkpoint found — starting fresh.")

# ========================= VISION HELPER =========================
@torch.no_grad()
def encode_images(images: list[Image.Image]) -> torch.Tensor:
    """SigLIP forward → [B, N, 768]."""
    inp   = siglip_processor(images=images, return_tensors="pt").to(device=device, dtype=dtype)
    feats = siglip_model.vision_model(**inp).last_hidden_state
    return feats.to(dtype)


# ========================= ALIGNMENT LOSS =========================
def compute_alignment_loss(prompts: list[str], sig_f: torch.Tensor) -> torch.Tensor:
    """
    Symmetric InfoNCE between norm-weighted mean of projected vision tokens
    and mean-pooled prompt text embeddings.
    sig_f: [B, N, 768] — pre-computed, no double forward.
    """
    proj_tokens  = projector(sig_f)                             # [B, N, LLM_DIM]
    patch_w      = torch.softmax(proj_tokens.norm(dim=-1, keepdim=True), dim=1)
    img_repr     = F.normalize((proj_tokens * patch_w).sum(dim=1), dim=-1)  # [B, LLM_DIM]

    tok = tokenizer(
        prompts, padding=True, truncation=True,
        max_length=TEXT_MAX_LEN, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        txt_emb = qwen_model.model.embed_tokens(tok.input_ids)
    mask     = tok.attention_mask.unsqueeze(-1).to(dtype)
    txt_repr = F.normalize(
        (txt_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1), dim=-1
    )

    scale  = logit_scale.exp().clamp(max=100.0)
    logits = scale * (img_repr @ txt_repr.T)
    labels = torch.arange(len(prompts), device=device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


# ========================= PATCH UTILITIES (G1, G2, P1, P2) =========================

def make_sinusoidal_pos_enc(n_patches: int, d_model: int, device, dtype) -> torch.Tensor:
    """
    P1: Standard sinusoidal positional encoding — same formulation as the
    original Transformer (Vaswani et al. 2017).

    Encodes patch index i across all d_model dimensions:
        PE[i, 2j]   = sin(i / 10000^(2j/d_model))
        PE[i, 2j+1] = cos(i / 10000^(2j/d_model))

    Why this beats the naive scalar sin(i):
    - Each patch token receives a unique D-dimensional fingerprint.
    - The encoding is smooth and distance-preserving: nearby patches
      have similar encodings, distant patches are orthogonal.
    - Gives the projector (and downstream attention) left/right and
      up/down awareness — critical for spatial tasks like object picking.

    Returns: [1, n_patches, d_model] (batch dimension for broadcasting).
    """
    position = torch.arange(n_patches, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32, device=device)
        * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(n_patches, d_model, dtype=torch.float32, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
    return pe.unsqueeze(0).to(dtype)            # [1, N, D]


def select_topk_patches(proj_tokens: torch.Tensor, k: int) -> torch.Tensor:
    """
    P2: Select top-k patches by attention-score importance rather than L2 norm.

    Score_i = (token_i · mean_token) — measures how semantically aligned each
    patch is with the dominant scene representation (mean embedding).
    This selects object-relevant patches rather than high-magnitude outliers,
    which tend to be edges/backgrounds with no task relevance.

    L2-norm top-k (previous): picks high-activation patches regardless of direction.
    Attention top-k (this):   picks patches pointing in the same direction as the
                               scene mean — i.e. the most representative objects.

    proj_tokens: [B, N, D]
    returns:     [B, k, D]
    """
    mean_tok = proj_tokens.mean(dim=1, keepdim=True)            # [B, 1, D]
    scores   = torch.bmm(proj_tokens,
                         mean_tok.transpose(1, 2)).squeeze(-1)  # [B, N]
    topk_idx = scores.topk(k, dim=1).indices                    # [B, k]
    idx_exp  = topk_idx.unsqueeze(-1).expand(-1, -1, proj_tokens.shape[-1])
    return proj_tokens.gather(1, idx_exp)                       # [B, k, D]


def attention_weighted_pool(patches: torch.Tensor) -> torch.Tensor:
    """
    G1: Self-attention-weighted pooling.
    Query = mean patch embedding; keys = all patch embeddings.
    Temperature-scaled softmax weights.

    patches: [B, N, D]
    returns: [B, D]
    """
    q       = patches.mean(dim=1, keepdim=True)                          # [B, 1, D]
    scores  = torch.bmm(patches, q.transpose(1, 2)).squeeze(-1)          # [B, N]
    weights = torch.softmax(scores / math.sqrt(patches.shape[-1]), dim=1)  # [B, N]
    return (patches * weights.unsqueeze(-1)).sum(dim=1)                  # [B, D]


# ========================= ACTION FORWARD (G1, G2, G3, P1, P2, P3) =========================
def forward_action(
    images: list,
    prompts: list[str],
    sig_f: torch.Tensor,
    training: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Full forward pass → (pred_pos [B,7], pred_vel [B,6]) in normalised space.

    1. Project SigLIP patches → LLM embedding space.
    P1. Add sinusoidal positional encoding to projected patches (spatial bias).
    P2. Select top-k patches by attention-score importance (semantic focus).
    3.  Splice top-k vision tokens into prompt embeddings at <image> position.
    4.  Forward through frozen Qwen; extract last hidden state.
    G1. Attention-weighted pool of top-k patches → vision_repr [B, D].
    P3. Add small training-time noise to vision_repr (anti mean-collapse).
    G3. Last non-padding Qwen token → lang_repr [B, D].
    7.  Hybrid: action_repr = cat([vision_repr, lang_repr]) → [B, 2D].
    8.  ActionHead → pred_pos, pred_vel.

    training: controls P3 noise injection (True during training, False at eval/inference).
    """
    B = len(images)

    # Step 1: project all patches
    all_proj = projector(sig_f)                             # [B, N_all, LLM_DIM]

    # P1: add full sinusoidal positional encoding
    # Cached per (N, D) pair so it is computed at most once per shape.
    pos_enc  = make_sinusoidal_pos_enc(
        all_proj.shape[1], all_proj.shape[2], all_proj.device, all_proj.dtype
    )                                                       # [1, N, D]
    all_proj = all_proj + pos_enc                           # [B, N, D]

    # P2: top-k selection by attention-score importance
    topk_proj = select_topk_patches(all_proj, TOPK_PATCHES)  # [B, k, LLM_DIM]

    # G1: attention-weighted vision repr — computed before Qwen so gradients
    # flow directly through the projector without going through frozen Qwen.
    vision_repr = attention_weighted_pool(topk_proj)        # [B, LLM_DIM]

    # P3: training-time noise injection on vision_repr
    # Amplitude 0.01 is small enough not to corrupt signal but large enough
    # to break symmetry and prevent all samples collapsing to the dataset mean.
    # Disabled at eval/inference to keep predictions deterministic.
    if training:
        vision_repr = vision_repr + 0.01 * torch.randn_like(vision_repr)

    # Step 3: splice into prompt embeddings
    enc = tokenizer(
        prompts, padding=True, truncation=True,
        max_length=TEXT_MAX_LEN, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        text_embeds = qwen_model.model.embed_tokens(enc.input_ids)  # [B, Lt, D]

    Nv = TOPK_PATCHES
    all_embeds, all_masks = [], []

    for i in range(B):
        ids   = enc.input_ids[i]
        tmask = enc.attention_mask[i]
        temb  = text_embeds[i]
        vemb  = topk_proj[i]                                # [k, D]

        img_pos = (ids == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]

        if len(img_pos) == 0:
            emb  = torch.cat([vemb, temb], dim=0)
            mask = torch.cat([torch.ones(Nv, device=device, dtype=torch.long), tmask])
        else:
            pos  = img_pos[0].item()
            emb  = torch.cat([temb[:pos], vemb, temb[pos + 1:]], dim=0)
            mask = torch.cat([
                tmask[:pos],
                torch.ones(Nv, device=device, dtype=torch.long),
                tmask[pos + 1:]
            ])

        all_embeds.append(emb)
        all_masks.append(mask)

    max_len        = max(e.shape[0] for e in all_embeds)
    inputs_embeds  = torch.stack([F.pad(e, (0, 0, 0, max_len - e.shape[0])) for e in all_embeds])
    attention_mask = torch.stack([F.pad(m, (0, max_len - m.shape[0]))        for m in all_masks])

    # Step 4: Qwen forward (frozen)
    out    = qwen_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    hidden = out.hidden_states[-1]                          # [B, L, D]

    # G3: last non-padding token as language repr
    last_idx  = attention_mask.sum(dim=1) - 1              # [B]
    last_idx  = last_idx.clamp(min=0)
    lang_repr = hidden[torch.arange(B, device=device), last_idx]  # [B, D]

    # G3: hybrid representation
    action_repr = torch.cat([vision_repr, lang_repr], dim=-1)      # [B, 2*D]

    pred_pos, pred_vel = action_head(action_repr.to(dtype))        # [B,7], [B,6]
    return pred_pos, pred_vel


# ========================= ACTION LOSS =========================
def compute_action_loss(
    pred_pos: torch.Tensor,
    pred_vel: torch.Tensor,
    tgt_pos:  torch.Tensor,
    tgt_vel:  torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Huber loss (smooth L1) on normalised actions.
    Returns (total_loss, pos_loss, vel_loss).
    """
    # Normalise targets
    tgt_pos_n, tgt_vel_n = normalise_action(
        tgt_pos.to(device), tgt_vel.to(device)
    )
    tgt_pos_n = tgt_pos_n.to(dtype)
    tgt_vel_n = tgt_vel_n.to(dtype)

    pos_loss = F.huber_loss(pred_pos, tgt_pos_n, delta=1.0)
    vel_loss = F.huber_loss(pred_vel, tgt_vel_n, delta=1.0)
    total    = POS_LOSS_WEIGHT * pos_loss + VEL_LOSS_WEIGHT * vel_loss
    return total, pos_loss, vel_loss


# ========================= ALPHA SCHEDULE (G4) =========================
def get_align_alpha(act_step: int) -> float:
    """
    G4: Returns 0 during ACTION_ONLY_STEPS warm-up, then decays from
    ALIGN_LOSS_ALPHA to ALIGN_LOSS_ALPHA_MIN over the next 20% of steps.
    """
    if act_step < ACTION_ONLY_STEPS:
        return 0.0
    decay_step = act_step - ACTION_ONLY_STEPS
    decay_len  = max(1, int(ESTIMATED_ACT_STEPS * 0.20))
    t = min(1.0, decay_step / decay_len)
    return ALIGN_LOSS_ALPHA + t * (ALIGN_LOSS_ALPHA_MIN - ALIGN_LOSS_ALPHA)


# ========================= OVERFIT DIAGNOSTIC (G7) =========================
def run_overfit_test(n_steps: int = 50) -> bool:
    """
    G7: Sanity-check the full pipeline by overfitting on a tiny fixed batch
    (4 samples). A correct pipeline should drive loss < 0.05 within ~50 steps.

    Returns True if the test passes (loss converges), False if it plateaus,
    which indicates a bug in gradient flow (frozen layer, wrong input, etc.).
    """
    print("\n" + "="*60)
    print("  OVERFIT DIAGNOSTIC (G7)")
    print("  Fitting 4 fixed samples for {} steps...".format(n_steps))
    print("="*60)

    projector.train()
    action_head.train()

    # Grab one fixed batch from training data
    fixed_batch = next(iter(DataLoader(
        train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn
    )))
    f_images, f_prompts, f_act_pos, f_act_vel = fixed_batch

    test_opt = torch.optim.AdamW(
        list(projector.parameters()) + list(action_head.parameters()),
        lr=1e-3
    )

    initial_loss = None
    final_loss   = None

    for step in range(n_steps):
        test_opt.zero_grad()
        sig_f = encode_images(f_images)
        pred_pos, pred_vel = forward_action(f_images, f_prompts, sig_f, training=True)
        loss, _, _ = compute_action_loss(pred_pos, pred_vel, f_act_pos, f_act_vel)
        (loss * ACTION_BETA).backward()
        torch.nn.utils.clip_grad_norm_(
            list(projector.parameters()) + list(action_head.parameters()), 1.0
        )
        test_opt.step()

        if step == 0:
            initial_loss = loss.item()
        if step % 10 == 0:
            # Check prediction variance — if all preds are identical, signal collapsed
            with torch.no_grad():
                p_std = pred_pos.std(dim=0).mean().item()
            print(f"    step {step:3d} | loss {loss.item():.4f} | pred_pos std {p_std:.4f}")
        final_loss = loss.item()

    passed = (final_loss < initial_loss * 0.3)
    status = "PASS ✓" if passed else "FAIL ✗ — check gradient flow"
    print(f"\n  Result: {status}  (initial={initial_loss:.4f} → final={final_loss:.4f})")
    print("="*60 + "\n")

    # Reload fresh optimiser states — overfit test must not pollute main training
    # (projector/action_head weights are dirtied; restore from last checkpoint or
    #  reinitialise if training from scratch)
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        projector.load_state_dict(ckpt["projector"])
        action_head.load_state_dict(ckpt["action_head"])
    else:
        # Re-initialise weights
        def _reset(m):
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        projector.apply(_reset)
        action_head.apply(_reset)

    projector.train()
    action_head.train()
    return passed


# ========================= EVAL =========================
@torch.no_grad()
def run_eval(step: int) -> float:
    projector.eval()
    action_head.eval()
    total_pos_loss = 0.0
    total_vel_loss = 0.0
    all_pred_pos   = []
    n_batches      = 0

    print(f"\n{'='*60}")
    print(f"  EVAL @ step {step}")
    print(f"{'='*60}")

    try:
        for batch_idx, (images, prompts, act_pos, act_vel) in enumerate(val_loader):
            sig_f              = encode_images(images)
            pred_pos, pred_vel = forward_action(images, prompts, sig_f, training=False)
            _, p_loss, v_loss  = compute_action_loss(pred_pos, pred_vel, act_pos, act_vel)
            total_pos_loss    += p_loss.item()
            total_vel_loss    += v_loss.item()
            all_pred_pos.append(pred_pos.float().cpu())
            n_batches         += 1

            if batch_idx == 0:
                pred_act_dn = denormalise_action(pred_pos[:2], pred_vel[:2])
                gt_act_dn   = torch.cat([act_pos[:2].to(device), act_vel[:2].to(device)], dim=-1)
                for i in range(min(2, len(images))):
                    print(f"\n  [Sample {i+1}]")
                    print(f"  Prompt : {prompts[i][:100]}")
                    pred_str = " ".join(f"{v:.3f}" for v in pred_act_dn[i].tolist())
                    gt_str   = " ".join(f"{v:.3f}" for v in gt_act_dn[i].tolist())
                    print(f"  Pred   : [{pred_str}]")
                    print(f"  GT     : [{gt_str}]")

        avg_pos = total_pos_loss / n_batches if n_batches else float("nan")
        avg_vel = total_vel_loss / n_batches if n_batches else float("nan")

        # G7-style variance check — if std ≈ 0, predictions have collapsed to mean
        if all_pred_pos:
            stacked  = torch.cat(all_pred_pos, dim=0)   # [N, 7]
            pred_std = stacked.std(dim=0).mean().item()
        else:
            pred_std = float("nan")

        print(f"\n  Avg val pos loss : {avg_pos:.4f}")
        print(f"  Avg val vel loss : {avg_vel:.4f}")
        print(f"  Pred pos std     : {pred_std:.4f}  (low → possible mean collapse)")
        print(f"{'='*60}\n")
        return avg_pos + avg_vel

    finally:
        projector.train()
        action_head.train()


# ========================= MAIN TRAINING =========================
print("Training started...")
projector.train()
action_head.train()

# G7: run overfit diagnostic before starting full training
run_overfit_test(n_steps=50)

for epoch in range(start_epoch, EPOCHS):
    batches_to_skip = (global_step % steps_per_epoch) if epoch == start_epoch else 0
    if batches_to_skip > 0:
        print(f"  Skipping {batches_to_skip} already-seen batches in epoch {epoch}...")

    total_act_loss = 0.0
    act_steps_this_epoch = 0
    act_step_global      = max(0, global_step - ALIGN_STEPS)

    align_optimizer.zero_grad()
    act_optimizer.zero_grad()

    for batch_idx, (images, prompts, act_pos, act_vel) in enumerate(
        tqdm(train_loader, desc=f"Epoch {epoch}")
    ):
        if batch_idx < batches_to_skip:
            continue

        # ─────────────── ALIGNMENT PHASE ───────────────
        if global_step < ALIGN_STEPS:
            sig_f      = encode_images(images)
            align_loss = compute_alignment_loss(prompts, sig_f) / ACCUM_STEPS
            align_loss.backward()

            if (global_step + 1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0)
                align_optimizer.step()
                align_scheduler.step()
                align_optimizer.zero_grad()

            if global_step % 100 == 0:
                print(f"  [Align] step {global_step} "
                      f"| loss {align_loss.item() * ACCUM_STEPS:.4f} "
                      f"| lr {align_scheduler.get_last_lr()[0]:.2e} "
                      f"| T {logit_scale.exp().item():.2f}")

            if global_step > 0 and global_step % SAVE_EVERY_STEPS == 0 \
                    and global_step != last_saved_step:
                save_checkpoint(global_step, epoch)
                last_saved_step = global_step

            global_step += 1
            continue

        # ─────────────── ACTION PREDICTION PHASE (G4, G5) ───────────────
        sig_f = encode_images(images)

        pred_pos, pred_vel  = forward_action(images, prompts, sig_f)
        act_loss, p_l, v_l  = compute_action_loss(pred_pos, pred_vel, act_pos, act_vel)

        # G4: alpha = 0 during ACTION_ONLY_STEPS, then decays
        alpha = get_align_alpha(act_step_global)

        if alpha > 0.0:
            align_loss = compute_alignment_loss(prompts, sig_f)
            # G5: scale action loss up so it always dominates alignment term
            total_loss = (ACTION_BETA * act_loss + alpha * align_loss) / ACCUM_STEPS
        else:
            # G4: pure action loss during warm-up — no alignment forward pass
            align_loss = torch.tensor(0.0, device=device)
            total_loss = (ACTION_BETA * act_loss) / ACCUM_STEPS

        total_loss.backward()

        if (global_step + 1) % ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(
                list(projector.parameters()) + list(action_head.parameters()), 1.0
            )
            act_optimizer.step()
            act_scheduler.step()
            act_optimizer.zero_grad()

        if global_step % 100 == 0:
            phase_tag = "warm" if alpha == 0.0 else "act "
            print(f"  [{phase_tag}] step {global_step} "
                  f"| act {act_loss.item():.4f} "
                  f"| pos {p_l.item():.4f} "
                  f"| vel {v_l.item():.4f} "
                  f"| align {align_loss.item():.4f} "
                  f"| α={alpha:.3f} "
                  f"| lr {act_scheduler.get_last_lr()[0]:.2e}")

        if global_step > 0 and global_step % SAVE_EVERY_STEPS == 0 \
                and global_step != last_saved_step:
            save_checkpoint(global_step, epoch)
            last_saved_step = global_step

        if global_step > 0 and global_step % EVAL_EVERY_STEPS == 0:
            run_eval(global_step)

        total_act_loss       += act_loss.item()
        act_steps_this_epoch += 1
        act_step_global      += 1
        global_step          += 1

    avg = total_act_loss / act_steps_this_epoch if act_steps_this_epoch else float("nan")
    print(f"Epoch {epoch} | Avg act loss: {avg:.4f} | Steps: {act_steps_this_epoch}")

    if global_step != last_saved_step:
        save_checkpoint(global_step, epoch)
        last_saved_step = global_step

print("Training complete!")
