# Hybrid Text Classification with Explainable AI

A hybrid text classification system that combines **TF–IDF lexical features** with **sentence-level semantic embeddings** to improve news classification accuracy while maintaining interpretability.  
The project also integrates **Explainable AI (XAI)** techniques using **SHAP** and **LIME**, and is deployed as a lightweight **Flask web application**.

---

## Project Overview

Traditional text classification models using TF–IDF are fast and interpretable but fail to capture semantic meaning.  
Neural embeddings capture semantics but are computationally expensive and difficult to interpret.

This project combines both approaches to leverage their complementary strengths:
- TF–IDF captures word importance and interpretability
- Sentence embeddings capture semantic similarity
- A hybrid ensemble improves accuracy and robustness

The system is evaluated on the **AG News dataset** and achieves **92.3% accuracy** with explainable predictions.

---

## How the System Works

1. **Text Preprocessing**
   - Lowercasing
   - Special character removal
   - Tokenization
   - Stopword removal
   - Stemming  
   Cleaned text is used for TF–IDF, while original text is preserved for embeddings.

2. **Feature Engineering**
   - **TF–IDF vectors** (20,000 dimensions, unigrams + bigrams)
   - **Sentence embeddings** using SentenceTransformer (384 dimensions)
   - Both feature sets are concatenated to form a hybrid representation.

3. **Classification Models**
   - Logistic Regression (primary model)
   - Random Forest (diversity model)
   - Soft-voting ensemble combining both models

4. **Explainability**
   - **SHAP** explains important TF–IDF words contributing to predictions
   - **LIME** provides local, human-readable explanations for individual predictions

5. **Deployment**
   - Flask-based web application
   - CPU-only inference
   - Optimized sparse matrix operations for memory efficiency
   - Sub-100ms prediction latency (without explainability)

---

## Example Output

<!-- Insert screenshot of web demo here -->
![Demo](files/demo.png)
<!-- Insert screenshot of SHAP explanation here -->
![SHAP](files/shap.png)
<!-- Insert screenshot of LIME explanation here -->
![LIME](files/lime.png)
---

## Dataset

- **AG News Corpus**
- 120,000 training samples
- 7,600 test samples
- 4 balanced classes:
  - World
  - Sports
  - Business
  - Sci/Tech

---

## Model Performance

| Model | Accuracy |
|------|----------|
| Logistic Regression | 92.1% |
| Random Forest | 89.7% |
| Hybrid Ensemble | **92.3%** |


---

## Tech Stack

**Machine Learning**
- scikit-learn
- SentenceTransformers
- SciPy (sparse matrices)

**Explainability**
- SHAP
- LIME

**Backend**
- Python
- Flask

---

## Key Highlights

- Hybrid TF–IDF + semantic embeddings
- Explainable predictions using SHAP and LIME
- Efficient CPU-only deployment
- Memory-optimized sparse feature fusion
- Production-ready Flask application

---


## Authors

**Vaibhav Singh**  
Department of Computer Science and Engineering  

