# Grounded Robot Policy

Vision-language grounded robot manipulation policy using SigLIP, Qwen2, multimodal token fusion, and continuous action-space prediction.

---

## Overview

Grounded Robot Policy is an experimental Vision-Language-Action (VLA) inspired framework for robot manipulation. The project investigates how large-scale vision-language representations can be leveraged for robot control by combining visual scene understanding, robot state information, and continuous action prediction.

The system integrates a frozen SigLIP visual encoder with a frozen Qwen2 language model through a trainable multimodal projector. Visual tokens are injected directly into the language model embedding space, enabling the policy to reason over visual observations and robot state representations before generating robot actions.

Unlike traditional imitation-learning pipelines that operate purely on visual features, this work explores whether grounded multimodal representations can improve policy learning in continuous robot action spaces.

---

## Architecture

```text
Camera Observation
        │
        ▼
   SigLIP Encoder
        │
        ▼
 Trainable Projector
        │
        ▼
 Visual Token Injection
        │
        ▼
      Qwen2
        │
        ▼
 Hybrid Representation
 (Vision + Language)
        │
        ▼
    Action Head
        │
        ▼
Joint Position + Velocity Commands
```

### Components

**Vision Encoder**

* SigLIP Base Patch16-224
* Frozen during training
* Produces dense visual patch embeddings

**Multimodal Projector**

* Trainable MLP mapping visual embeddings into Qwen latent space
* Residual architecture with LayerNorm and GELU activations

**Language Backbone**

* Qwen2-7B
* Frozen weights
* Receives projected visual tokens through direct embedding injection

**Action Policy Head**

* Continuous action regression
* Predicts:

  * 7-DoF joint position targets
  * 6-DoF joint velocity targets

---

## Dataset Format

The framework operates on robot demonstration trajectories stored in HDF5 format.

```text
episode_N.hdf5

observations/
├── images/
│   └── zed_left
├── follower/
│   ├── position
│   └── velocity
└── end_pose/
    └── pose

actions/
├── position
└── velocity
```

Each training sample contains:

* RGB camera observation
* Joint positions
* Joint velocities
* End-effector pose
* Ground-truth action targets

---

## Key Features

### Visual Grounding

Visual patch embeddings are projected into the language model latent space and injected as multimodal tokens, allowing the policy to condition actions on scene content rather than solely robot state.

### Attention-Based Patch Selection

Patch importance scoring is used to identify semantically relevant visual regions before language-model processing.

### Hybrid Action Representation

Action prediction combines:

* Attention-pooled visual features
* Final hidden-state language features

This preserves both scene-level information and contextual representations.

### Continuous Action Learning

The policy predicts:

* Joint position commands
* Joint velocity commands

using Huber-loss-based regression in normalized action space.

### Multi-Stage Training

Training consists of:

1. Vision-language alignment
2. Action-only warm-up
3. Joint action and alignment optimization

to stabilize policy learning and prevent representation collapse.

---

## Training Objectives

### Vision-Language Alignment

A contrastive InfoNCE objective aligns projected visual embeddings with textual robot-state descriptions.

### Action Prediction

The primary objective is continuous action regression:

```text
Image + State → Joint Position Targets
Image + State → Joint Velocity Targets
```

using Huber loss on normalized action representations.

---

## Research Focus

This project explores several research questions:

* Can vision-language representations improve robot policy learning?
* How effectively can frozen language models support robot action prediction?
* What information survives multimodal token projection into LLM latent spaces?
* Can visual grounding emerge without explicit task-language supervision?
* How should continuous robot actions be represented within VLA-inspired systems?

---

## Experimental Techniques

Implemented techniques include:

* Multimodal token projection
* Direct visual token injection
* Attention-weighted patch pooling
* Top-k semantic patch selection
* Sinusoidal positional encodings
* Continuous action normalization
* Contrastive alignment objectives
* Noise-based anti-collapse regularization
* Automated overfit diagnostics
* Checkpoint recovery and evaluation pipelines

---

## Current Status

This repository is a research-oriented exploration of grounded robot policy learning.

Experiments indicate that meaningful visual representations can emerge through multimodal token fusion and grounding mechanisms. Ongoing work focuses on improving the translation of grounded representations into robust robot action policies through temporal modeling, task conditioning, and larger-scale demonstration datasets.

---

## Technologies

* PyTorch
* Hugging Face Transformers
* SigLIP
* Qwen2-7B
* HDF5
* NumPy
* CUDA

---

## License

MIT License
