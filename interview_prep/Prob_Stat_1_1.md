# Study Guide: 1.1 Probability and Statistics

This document provides study materials, required readings, and exercises designed to master the Probability and Statistics concepts required for ML/AI interviews. The exercises are structured around the 4 levels of mastery: Conceptual, Mathematical, Implementation, and Systems/Research judgment.

---

## Part 1: Recommended Reading & Resources

### General Probability & Statistics
1. **Deep Learning Book (Goodfellow et al.) - Chapter 3**: Essential reading for probability and information theory (Entropy, KL Divergence, Cross-Entropy) from a machine learning perspective.
   - [Read Online](https://www.deeplearningbook.org/contents/prob.html)
2. **CS229 Probability Theory Review**: A fantastic, concise review of probability tailored for ML engineers.
   - [CS229 Notes](http://cs229.stanford.edu/section/cs229-prob.pdf)
3. **Pattern Recognition and Machine Learning (Bishop) - Chapters 1 & 2**: Great for deep dives into distributions, MLE, MAP, and the exponential family.

### Specific Topics
- **MLE vs MAP**: Understand the Bayesian view of regularization. L2 regularization is equivalent to MAP with a Gaussian prior.
- **Hypothesis Testing for ML**: Thomas Dietterich's paper "Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms" is a classic read for understanding permutation tests and confidence intervals in ML.
- **Aleatoric vs Epistemic Uncertainty**: Read Kendall & Gal (2017), "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?"

---

## Part 2: Targeted Exercises

These exercises are designed to help you confidently answer the core questions and master the material at all 4 levels of depth.

### Topic A: Cross-Entropy and Maximum Likelihood (Q1)
*Question: Why does cross-entropy correspond to maximum likelihood?*

- **Level 1: Conceptual**
  - Explain to a non-technical manager why minimizing cross-entropy loss is the same as maximizing the probability of our training data given the model.
- **Level 2: Mathematical**
  - **Exercise**: Start with the likelihood function \(L(\theta) = \prod_{i=1}^n P(y_i | x_i; \theta)\). Take the negative log-likelihood (NLL). Show that for a categorical distribution (classification), the NLL is mathematically equivalent to the Cross-Entropy loss \(\sum -y_i \log(\hat{y}_i)\).
- **Level 3: Implementation**
  - **Exercise**: Write a Python/NumPy function from scratch that computes the Cross-Entropy loss between a batch of true labels (one-hot encoded) and predicted probabilities. Then compute the gradient of this loss with respect to the predicted probabilities.
- **Level 4: Systems/Research Judgment**
  - **Exercise**: If you are training a model and the cross-entropy loss drops to near zero, what does this tell you about the model's confidence? Can a model have low cross-entropy loss but still be poorly calibrated? Why?

### Topic B: Information Theory (Q2)
*Question: What is KL divergence measuring?*

- **Level 1: Conceptual**
  - How would you explain KL Divergence intuitively using the concept of "surprise" or "extra bits"? Why is it not a true distance metric?
- **Level 2: Mathematical**
  - **Exercise**: Prove that \(KL(P || Q) \ge 0\) using Jensen's inequality. 
  - **Exercise**: Show the relationship between Cross-Entropy, Entropy, and KL Divergence: \(H(P, Q) = H(P) + KL(P || Q)\).
- **Level 3: Implementation**
  - **Exercise**: Write a PyTorch function to compute \(KL(P || Q)\) for two discrete probability distributions. Show empirically that \(KL(P || Q) \neq KL(Q || P)\).
- **Level 4: Systems/Research Judgment**
  - **Exercise**: In the context of Reinforcement Learning (like PPO) or RLHF (like DPO), KL divergence is often used as a penalty term against a reference model. Why do we penalize large KL divergence, and what happens to the agent's behavior if the KL penalty is too small or too large?

### Topic C: Uncertainty (Q3)
*Question: Difference between aleatoric and epistemic uncertainty.*

- **Level 1: Conceptual**
  - Define aleatoric and epistemic uncertainty. Give a real-world example for both in the context of an autonomous vehicle predicting pedestrian movement.
- **Level 2: Mathematical**
  - **Exercise**: How does the variance of a predictive distribution decompose into aleatoric and epistemic parts conceptually? 
- **Level 3: Implementation**
  - **Exercise**: Implement Monte Carlo Dropout (MC Dropout) in PyTorch for a simple regression task. Use the variance of the predictions across multiple forward passes to estimate epistemic uncertainty.
- **Level 4: Systems/Research Judgment**
  - **Exercise**: You are building an LLM for medical diagnosis. It occasionally hallucinates wildly with high confidence. Which type of uncertainty is failing here? How would you design a system to detect and reject high-epistemic-uncertainty queries?

### Topic D: Statistical Testing and Evaluation (Q4)
*Question: How would you test whether model A is significantly better than model B?*

- **Level 1: Conceptual**
  - Why is it not enough to say "Model A got 85% accuracy and Model B got 84%, therefore Model A is better"? What is a p-value in this context?
- **Level 2: Mathematical**
  - **Exercise**: Formulate the null hypothesis (\(H_0\)) and alternative hypothesis (\(H_A\)) for comparing Model A and Model B. Explain how a permutation test calculates the p-value.
- **Level 3: Implementation**
  - **Exercise**: Write a Python script to perform a Paired Bootstrap test. Given an array of predictions from Model A, Model B, and the Ground Truth, resample the test set with replacement 10,000 times to create a 95% confidence interval around the difference in their performance metrics (e.g., F1 score).
- **Level 4: Systems/Research Judgment**
  - **Exercise**: You work at a large tech company. Model A performs +0.5% better than Model B on a static benchmark, with statistical significance (\(p < 0.01\)). However, Model A is 3x larger. Discuss the trade-offs. What online metrics (A/B testing) would you look at before deciding to deploy Model A over Model B?

---

## Part 3: General Mastery Exercises (Core Fundamentals)

**1. Probability Distributions & Expectation**
- **Math**: Derive the expected value and variance for a Bernoulli distribution.
- **Math**: Show that the Gaussian distribution is part of the exponential family. What are its natural parameters?

**2. Bayes' Rule, MLE, and MAP**
- **Conceptual**: Explain the difference between MLE and MAP estimation. When does MAP estimation equal MLE?
- **Math**: Derive the MAP estimate for the mean of a Gaussian distribution with a Gaussian prior. Show how the prior acts as a regularization term pulling the estimate towards the prior mean.

**3. Bias-Variance Tradeoff**
- **Math**: Decompose the expected mean squared error (MSE) of a model into Bias squared, Variance, and Irreducible Error.
- **Systems**: Explain how Random Forests reduce variance and how Gradient Boosting models reduce bias. 

**4. Calibration**
- **Implementation**: Write a function to compute the Expected Calibration Error (ECE) of a classification model by bucketing its predicted probabilities.
- **Systems**: Why do modern neural networks tend to be overconfident (poorly calibrated)? What techniques (like temperature scaling) can be used to fix this post-training?
