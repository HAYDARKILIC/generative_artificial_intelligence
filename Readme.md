# 🤖 Generative AI — Lecture Notes and Hands-On Notebooks

**Haydar Kilic | Faculty of Engineering, Artificial Intelligence Engineering**

This repository contains Jupyter Notebooks that reinforce the theoretical slide content of the *Generative AI* course with Python. Each notebook derives and visualises the formulas covered in lectures from scratch, and adapts them to real data scenarios.

---

## 📚 Contents

| Lecture | Topic | Notebook |
|---------|-------|----------|
| Lecture 1 | Generative Modelling Fundamentals | [`GAI_Lecture1_Notebook.ipynb`] |
| Lecture 2 | Derivation of Generative Models (MAP · MLE · Beta-Binomial · Dirichlet) | [`GAI_Lecture2_Notebook.ipynb`] |
| Lecture 3 | Deep Generative Models (VAE · GAN · GMMN · Diffusion) | [`GAI_Lecture3_Notebook.ipynb`] |
| Lecture 4 | Transformers and Large Language Models (Attention · RoPE · Mini GPT · Scaling) | [`GAI_Lecture4_Notebook.ipynb`] |

> The table will be updated as new lectures are added.

---

## 🗂 Lecture 1 — Generative Modelling Fundamentals

### Topics Covered

**Section 1 — Core Concepts**
- Handwritten digit recognition: 28×28 pixel vector representation, train/test/validation split
- Polynomial regression and curve fitting (Vandermonde matrix, Least Squares)
- Overfitting / Underfitting and RMS error analysis
- Ridge Regularisation (L2 penalty, λ hyperparameter)

**Section 2 — Probability Theory**
- Joint, marginal and conditional probability distributions
- Bayes' theorem — medical diagnosis and *base rate fallacy*
- Gaussian (Normal) distribution: PDF, CDF, numerical verification
- Maximum Likelihood Estimation (MLE) and bias
- Bayesian updating: coin flip prior → posterior

**Section 3 — Decision Theory**
- Minimum-error decision boundaries and posterior probabilities
- Reject Option and threshold θ
- Asymmetric loss matrix (medical diagnosis scenario)
- Generative / Discriminative / Discriminant model comparison

---

## 🗂 Lecture 2 — Derivation of Generative Models

### Topics Covered

**Section 1 — Learning from Positive Examples & The Number Game**
- Concept learning = binary classification; posterior predictive distribution
- Strong sampling assumption: `p(D|h) = (1/|h|)^N`
- **Size Principle:** narrow hypothesis → high likelihood
- Prior, likelihood and posterior computation; Bayesian updating
- **MAP estimation** and N → ∞ behaviour (Dirac convergence)
- **Bayesian Model Averaging (BMA)** vs. Plug-In approach
- Mixture prior (π₀ parameter): rule-based vs. interval-based hypotheses

**Section 2 — Beta-Binomial Model**
- Bernoulli likelihood and sufficient statistics (N₁, N₀)
- Beta distribution: conjugate prior, various (a, b) parameters
- Sequential Bayesian updating: Beta(a,b) → Beta(N₁+a, N₀+b)
- MLE, MAP and posterior mean formulas; convergence as N grows
- **Zero Count Problem** and Laplace succession rule
- Posterior variance and confidence interval: σ ∝ 1/√N
- **Compound Beta-Binomial distribution:** prediction of future trials

**Section 3 — Dirichlet-Multinomial**
- Multinomial likelihood and Dirichlet prior
- Visualisation of the K=3 probability simplex (barycentric coordinates)
- Dirichlet-Multinomial update and posterior prediction
- **Add-K smoothing (β):** MLE → Laplace → uniform

**Section 4 — Mixture Model**
- Effect of the π₀ parameter on the posterior predictive distribution

**Section 5 — MLE vs MAP vs Bayes Comparison**
- Error analysis, convergence of θ estimates with N

---

## 🗂 Lecture 3 — Deep Generative Models

### Topics Covered

**Section 1 — Probabilistic Framework & MLE**
- Real data simulation with a 2D Gaussian mixture
- Log-Gaussian log-likelihood function
- MLE vs. bad model comparison

**Section 2 — KL Divergence**
- Closed-form Gaussian KL computation
- KL asymmetry: KL(p‖q) ≠ KL(q‖p)
- MLE ≡ KL minimisation relationship

**Section 3 — Latent Space & Manifold Hypothesis**
- MNIST: 784 pixels → ~10-dimensional manifold (PCA variance analysis)
- Latent space visualisation via 2D PCA projection
- Latent space arithmetic: z(7) − z(1) + z(0) ≈ z(6)

**Section 4 — ELBO Derivation**
- Closed-form KL computation and heat map
- Balance between reconstruction and KL terms

**Section 5 — Variational Autoencoder (VAE)**
- Encoder–Decoder architecture, Reparametrisation Trick
- Gradient flow diagram (why backprop works)
- Training on MNIST; 2D latent space visualisation
- β-VAE: KL regularisation effect; Posterior Collapse problem

**Section 6 — Generative Adversarial Networks (GAN)**
- Generator + Discriminator architecture (LeakyReLU, BatchNorm)
- Optimal Discriminator formula and Nash equilibrium visualisation
- MNIST training; G/D loss curves and mode-collapse discussion

**Section 7 — GMMN & MMD**
- Gaussian (RBF) kernel and MMD² computation (multi-scale)
- MMD intuition test: same / nearby / distant distributions
- Discriminator-free GMMN training (MMD loss only)

**Section 8 — Diffusion Models (DDPM)**
- Forward process: β schedule, closed-form q(x_t|x_0)
- SimpleUNet: time embedding + skip-connection noise estimator
- DDPM training (MSE loss) and reverse process sampling
- Step-by-step denoising visualisation

**Section 9 — Model Comparison & FID**
- Fréchet Inception Distance computation (PCA feature space)
- Radar chart: Quality / Diversity / Speed / Stability / Latent Control
- Generative model chronology (1985–2022)
- Comprehensive comparison table

---

## 🗂 Lecture 4 — Transformers and Large Language Models

### Topics Covered

**Section 1 — RNN vs Transformer: Vanishing Gradients**
- Simulation of |dL/dh_t| ≈ |W_hh|^(T-t) exponential decay in simple RNNs
- Vanishing / stable / exploding regimes (|W_hh| = 0.85 / 1.00 / 1.15)
- Transformer O(1) connection distance: direct access to every token pair

**Section 2 — Encoder–Decoder and the Information Bottleneck**
- Cosine similarity loss at different sequence lengths with a GRU encoder
- RNN Enc-Dec single-vector bottleneck vs. Attention context vector comparison
- Visual explanation of c_t = Σ α_{t,i} · h_i

**Section 3 — Bahdanau (Additive) Attention Mechanism**
- From-scratch BahdanauAttention: W_s, W_h, v parameterised scoring
- e_{t,i} = vᵀ tanh(W_s·s_{t-1} + W_h·h_i) → softmax → context vector
- English→German translation simulation: 4×4 attention heatmap

**Section 4 — Scaled Dot-Product Attention (Q, K, V)**
- `Attention(Q,K,V) = softmax(QK^T / √d_k) · V` step-by-step implementation
- Importance of √d_k scaling: entropy analysis (unscaled softmax collapses as d_k grows)
- Dimension analysis: (B, T, d_model) → Q/K/V → (B, T, d_k) → Z

**Section 5 — Multi-Head Attention**
- Single large W_q/W_k/W_v matrix approach; split_heads → (B, n_heads, T, d_k)
- 4-head attention maps: Position / Syntax / Semantics / Distance
- Parameter analysis: 4 × d_model² weights

**Section 6 — Positional Encoding (Sinusoidal, RoPE, ALiBi)**
- PE_{pos,2i} = sin(pos/10000^{2i/d}), PE_{pos,2i+1} = cos(…): matrix visualisation
- Wave frequencies: low dimension = high frequency; PE similarity matrix
- RoPE: relative positional encoding via 2D rotation; q^T_m k_n ∝ f(m-n)
- ALiBi: e_{ij} = q_i^Tk_j − m·|i−j| linear penalty; slope m_i = 2^{−8i/n_heads}
- Comparison table: Sinusoidal / Learned / RoPE / ALiBi

**Section 7 — Feed-Forward Network & Activation Functions**
- ReLU → GELU → Swish/SiLU → SwiGLU(x,W,V) = Swish(xW) ⊙ xV
- Gradient analysis: dead neuron problem in ReLU for x<0 region
- d_ff = 4×d_model expansion rule and FFN parameter growth

**Section 8 — Layer Normalization: LayerNorm vs RMSNorm / Pre-LN vs Post-LN**
- LN(x) = γ·(x−μ)/√(σ²+ε)+β vs. RMSNorm(x) = γ·x/RMS(x) (no β, ~10% faster)
- std/mean comparison at different input scales
- Pre-LN (modern) vs Post-LN (original): gradient distribution histogram
- BN vs LN vs RMSNorm: preference analysis in sequence models

**Section 9 — Attention Masking: Full vs Causal**
- make_full_mask (Bidirectional): BERT/RoBERTa — every token attends to every other
- make_causal_mask (lower triangular): GPT — only past visible, future −∞
- Masking → model family → task matching table (Encoder / Decoder / Enc-Dec)

**Section 10 — Full Transformer Block (From-Scratch Implementation)**
- TransformerEncoderBlock: Pre-LN + MHA + FFN + Residual
- TransformerEncoder: N layers, learned PE, final LayerNorm
- Parameter analysis for 3 model configurations (Small / BERT-mini / BERT-base)
- #params ≈ 12 × N × d²_model estimation formula

**Section 11 — Mini GPT: Character-Level Language Model**
- GPTDecoderBlock: Causal MHA + Pre-LN + FFN
- MiniGPT: tok_emb + pos_emb + 3 decoder blocks + lm_head (weight tying)
- Autoregressive generate(): top-k sampling + temperature control
- 500-step training on Turkish text: loss curve + attention map
- Generated text samples at different temperatures (0.5 / 1.0 / 1.5)

**Section 12 — Hyperparameter Analysis & Scaling Laws**
- Real LLM table: BERT-base/large, GPT-2, GPT-3, LLaMA-2 7B/70B
- Scaling law: L ∝ N^{−0.076} log-log visualisation
- d_model vs number of heads (d_k = d_model/h ≈ 64–128 rule)
- GPT vs BERT comparison table: architecture, task, context, usage
- Modern LLM block: RMSNorm + Pre-LN + SwiGLU + RoPE

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/HAYDARKILIC/generative_artificial_intelligence
cd generative_artificial_intelligence

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

---

## 📦 Requirements

```
numpy>=2.0
matplotlib>=3.7
scipy>=1.11
scikit-learn>=1.3
jupyter>=1.0
ipykernel>=6.0
torch>=2.0
torchvision>=0.15
tqdm>=4.65
```

> The `requirements.txt` file is included in the repository.
>
> ⚠️ `torch` and `torchvision` are required from Lecture 3 onwards. For GPU support, select a CUDA-compatible version at [pytorch.org](https://pytorch.org/get-started/locally/).

---

## 🏗 Project Structure

```
generative-ai/
├── README.md
├── requirements.txt
├── GAI_Lecture1_Notebook.ipynb   # Lecture 1 — Generative Modelling Fundamentals
├── GAI_Lecture2_Notebook.ipynb   # Lecture 2 — MAP · MLE · Beta-Binomial · Dirichlet
├── GAI_Lecture3_Notebook.ipynb   # Lecture 3 — VAE · GAN · GMMN · Diffusion
├── GAI_Lecture4_Notebook.ipynb   # Lecture 4 — Transformer · Attention · Mini GPT · LLM
└── (future lecture notebooks will be added here)
```

---

*Generative AI — Haydar Kılıç, Artificial Intelligence Engineering*
