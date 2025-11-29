
import os
import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from scipy import sparse

from utils import TfidfWrapper, EmbeddingTransformer

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def get_ag_news(as_pandas=True, sample=None):
    ds = load_dataset("ag_news")
    if as_pandas:
        train = pd.DataFrame(ds['train'])
        test = pd.DataFrame(ds['test'])
        if sample:
            train = train.sample(sample, random_state=42)
            test = test.sample(min(len(test), sample//10), random_state=42)
        return train, test
    return ds

def make_hybrid_features(texts, tfidf_wrapper, emb_transformer):
    # tfidf -> sparse, embeddings -> dense
    X_tfidf = tfidf_wrapper.transform(texts)
    X_emb = emb_transformer.transform(texts)
    X_emb_sparse = sparse.csr_matrix(X_emb)
    X = sparse.hstack([X_tfidf, X_emb_sparse], format='csr')
    return X

def main(sample=None):
    print("Loading AG News dataset...")
    train, test = get_ag_news(sample=sample)
    X_train_text = (train['text']).tolist()
    y_train = train['label'].tolist()
    X_test_text = (test['text']).tolist()
    y_test = test['label'].tolist()

    print("Fitting TF-IDF...")
    tfidf = TfidfWrapper(max_features=20000)
    tfidf.fit(X_train_text)

    print("Preparing embeddings (SentenceTransformers)...")
    emb = EmbeddingTransformer(model_name="all-MiniLM-L6-v2")
    emb.fit(X_train_text)

    print("Building hybrid features for training set...")
    X_train = make_hybrid_features(X_train_text, tfidf, emb)
    X_test = make_hybrid_features(X_test_text, tfidf, emb)

    print("Training Logistic Regression (LR)...")
    lr = LogisticRegression(max_iter=1000, n_jobs=-1)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    print("LR accuracy:", accuracy_score(y_test, y_pred_lr))
    print(classification_report(y_test, y_pred_lr))

    print("Training Random Forest (RF)...")
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    print("RF accuracy:", accuracy_score(y_test, y_pred_rf))
    print(classification_report(y_test, y_pred_rf))

    print("Building hybrid ensemble (confidence-sum)...")
    prob_lr = lr.predict_proba(X_test)
    prob_rf = rf.predict_proba(X_test)
    prob_sum = prob_lr + prob_rf
    y_pred_hybrid = np.argmax(prob_sum, axis=1)
    print("Hybrid accuracy:", accuracy_score(y_test, y_pred_hybrid))
    print(classification_report(y_test, y_pred_hybrid))

    joblib.dump(lr, os.path.join(MODEL_DIR, "lr_model.joblib"))
    joblib.dump(rf, os.path.join(MODEL_DIR, "rf_model.joblib"))
    joblib.dump(tfidf.vec, os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
    joblib.dump({"emb_model_name": emb.model_name}, os.path.join(MODEL_DIR, "emb_meta.joblib"))

    print("Saved models to", MODEL_DIR)

if __name__ == "__main__":
    main(sample=None)
