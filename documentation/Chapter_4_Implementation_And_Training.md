# Chapter 4: Implementation and Training

## 4.1 Dataset Processing and Splits
The input images are resized to $224 \times 224$ pixels to match the input specifications of our model backbones. The data is partitioned into:
- **Training Set (70%)**: Used to optimize model parameters.
- **Validation Set (15%)**: Used to monitor generalization and tune hyperparameters.
- **Test Set (15%)**: Reserved for final evaluation.

### Class Balance & Weighted Sampling
Medical datasets are typically imbalanced, with benign cases outnumbering malignant cases. To prevent model bias, we compute class weights based on reciprocal frequencies:
$$W_c = \frac{N}{C \cdot N_c}$$
where $N$ is total samples, $C$ is class count, and $N_c$ is sample count for class $c$. We load batches using a PyTorch `WeightedRandomSampler` to ensure uniform representation.

## 4.2 Optimization & Hyperparameters
- **Loss Function**: Cross-Entropy Loss.
- **Optimizer**: Adam ($\beta_1 = 0.9, \beta_2 = 0.999$) with learning rate $= 1 \times 10^{-4}$ and weight decay $= 1 \times 10^{-5}$ to prevent overfitting.
- **Batch Size**: 16.
- **Regularization**: Dropout ($p = 0.5$) in the classification head, combined with Random Rotation ($15^{\circ}$) and Horizontal Flips for data augmentation.
