# Chapter 4: Implementation and Training

## 4.1 Multi-Center Dataset Processing and Stratified Splits
The multi-center dataset (**BUSI**, **OASBUD**, **BrEaST**) was partitioned using stratified sampling:
- **Training Set (70%, $N=828$):** Used for gradient-based parameter optimization.
- **Validation Set (15%, $N=177$):** Used to monitor generalization, trigger learning rate schedulers, and select optimal checkpoints.
- **Held-Out Test Set (15%, $N=177$):** Isolated for final objective benchmarking.

### Class Imbalance & Weighted Cross-Entropy Loss
To prevent algorithmic bias toward the majority benign class, reciprocal class weights are integrated into the cross-entropy loss function:
$$W_c = \frac{N}{K \cdot N_c}$$
$$\mathcal{L}_{\text{WCE}} = - \frac{1}{N} \sum_{i=1}^N \sum_{c=1}^K W_c \cdot y_{i, c} \cdot \log \hat{p}_{i, c}$$

## 4.2 Optimization Dynamics & Hyperparameters
- **Loss Function:** Class-Weighted Cross-Entropy Loss.
- **Optimizer:** AdamW with decoupled weight decay ($\lambda = 1 \times 10^{-4}$), $\beta_1 = 0.9, \beta_2 = 0.999$, and initial learning rate $\eta_0 = 1 \times 10^{-4}$.
- **Learning Rate Schedule:** Cosine Annealing decay across 25 epochs:
  $$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})\left( 1 + \cos\left( \frac{t}{T_{\max}} \pi \right) \right)$$
- **Batch Size:** 16.
- **Regularization & Augmentation:** Dropout ($p = 0.5$) in the classification head, combined with Random Horizontal/Vertical Flips ($p=0.5$) and Random Rotations ($\pm 15^\circ$).

## 4.3 Training Convergence
Training loss converged rapidly from $0.55$ to $0.015$ over 10 epochs for EfficientNet-B0, maintaining stable validation loss without evidence of gradient divergence or severe overfitting.
