# Biologically Plausible Deep Learning: PyTorch Reimplementation and Extension

**Course Information:** <br/>
CS757 Generative Deep Learning <br/>
Spring 2025 <br/>
George Mason University <br/>

A PyTorch Reimplementation and Extension of "Towards Biologically Plausible Deep Learning" by Bengio et al. <br/>

# Overview
This project reimplements the biologically plausible deep learning framework from Bengio et al. (2015), which replaces backpropagation with target propagation using local encoder-decoder pairs and iterative inference.

## Key Features:

* PyTorch reimplementation of the original model
* Experiments on MNIST and Fashion-MNIST datasets
* Modified architecture with increased capacity
* Comparison of baseline vs. modified approaches

# Paper Summary
"Towards Biologically Plausible Deep Learning" (Bengio et al., 2015) proposes an alternative to backpropagation that is more aligned with biological learning mechanisms.

## Key Concepts:

* Target Propagation: Uses local encoder-decoder pairs instead of global gradient propagation
* Denoising Autoencoders: Each layer is trained with Gaussian noise for robustness
* Iterative Inference: Refines hidden states through feedback during training and generation
* Generative Modeling: Samples from Gaussian priors and evaluates using Parzen window estimation

The framework demonstrates that iterative inference produces significantly more realistic samples than direct decoding, supporting the biological plausibility of feedback-based learning.

# Repository Structure
```
notebooks/
├── Baseline-MNIST.ipynb # Original model on MNIST
├── Baseline-FashionMNSIT.ipynb # Original model on Fashion-MNIST
├── ModifiedV2-MNIST.ipynb # Modified architecture on MNIST
└── ModifiedV2-FashionMNIST.ipynb # Modified architecture on Fashion-MNIST
```
# Key Findings
This PyTorch reimplementation successfully reproduces the original MNIST experiments and extends them to Fashion-MNIST with a deeper architectural variant.
## Main Takeaways:

* Target propagation with denoising objectives is a viable alternative to backpropagation in unsupervised learning
* Iterative inference significantly improves generation quality, especially with weak priors
* Overly expressive decoders can reduce the need for inference and harm alignment between prior and posterior distributions
* The approach generalizes well to more complex datasets like Fashion-MNIST
* Parzen window log-likelihood effectively evaluates generative model quality

## Implementation: 
Complete reimplementation in PyTorch, faithfully capturing the algorithmic foundations including layer-wise denoising autoencoder training, target propagation, iterative inference, and generative sampling.

# Requirements
```
torch
torchvision
numpy
matplotlib
tqdm
```

# Usage
Explore the notebooks to see the implementation and results:
```
Baseline-MNIST.ipynb - Original model reproduction
Baseline-FashionMNIST.ipynb - Extension to Fashion-MNIST
ModifiedV2-MNIST.ipynb - Deeper architecture experiments
ModifiedV2-FashionMNIST.ipynb - Modified architecture on Fashion-MNIST
```
Detailed experimental results, analysis, and implementation details are available in the project report.

# References
Bengio, Y., Lee, D. H., Bornschein, J., Mesnard, T., & Lin, Z. (2015). Towards biologically plausible deep learning. arXiv preprint arXiv:1502.04156.

