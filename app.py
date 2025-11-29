from flask import Flask, request, render_template_string
import joblib
import numpy as np
import shap, lime.lime_text
from scipy import sparse
from utils import EmbeddingTransformer

# Load Models + Transformers
MODEL_DIR = "models"
LABELS = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}
tfidf = joblib.load(f"{MODEL_DIR}/tfidf_vectorizer.joblib")
emb_meta = joblib.load(f"{MODEL_DIR}/emb_meta.joblib")

lr = joblib.load(f"{MODEL_DIR}/lr_model.joblib")
rf = joblib.load(f"{MODEL_DIR}/rf_model.joblib")

emb = EmbeddingTransformer(model_name=emb_meta['emb_model_name'])
emb.fit([])

# SHAP KernelExplainer Initialization (safe for hybrid features)
HYBRID_DIM = lr.coef_.shape[1]         
background = np.zeros((1, HYBRID_DIM))

shap_explainer = shap.KernelExplainer(
    lambda x: lr.predict_proba(x),
    background
)

lime_explainer = lime.lime_text.LimeTextExplainer(
    class_names=["World", "Sports", "Business", "Sci/Tech"]
)

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Hybrid Text Classifier</title>
<style>
body { font-family: Arial; margin: 40px; }
textarea { width: 100%; height: 160px; font-size: 17px; padding: 10px; }
.box { margin-top: 20px; padding: 15px; border: 1px solid #aaa; border-radius: 6px; }
.word { padding: 3px 5px; margin: 3px; display: inline-block; border-radius: 4px; }
.pos { background-color: #b5f7b5; }
.neg { background-color: #f7b5b5; }
</style>
</head>

<body>
<h1>Hybrid Text Classifier</h1>

<form method="POST">
<textarea name="text" placeholder="Type or paste text...">{{ text }}</textarea>
<br><br>
<button type="submit">Classify</button>
</form>

{% if results %}
<div class="box">
<h2>Model Predictions</h2>
<p><b>Logistic Regression:</b> {{ results.lr }}</p>
<p><b>Random Forest:</b> {{ results.rf }}</p>
<p><b>Hybrid (LR + RF):</b> {{ results.hybrid }}</p>

<h2>SHAP (TF-IDF Feature Contributions)</h2>
<p>
{% for w,v in results.shap %}
<span class="word {% if v > 0 %}pos{% else %}neg{% endif %}">
{{ w }} ({{ "%.4f"|format(v) }})
</span>
{% endfor %}
</p>

<h2>LIME Explanation</h2>
<div>{{ results.lime|safe }}</div>

</div>
{% endif %}
</body>
</html>
"""

def make_features(texts):
    X_tfidf = tfidf.transform(texts)
    X_emb = emb.transform(texts)
    X_emb_sparse = sparse.csr_matrix(X_emb)
    return sparse.hstack([X_tfidf, X_emb_sparse], format='csr')

# SHAP explanation (TF-IDF only) - robust extraction
def explain_shap(text):
    
    X = make_features([text]).toarray()  
    shap_vals_all = shap_explainer.shap_values(X)

    class_idx = int(np.argmax(lr.predict_proba(X)[0]))
    shap_arr = np.asarray(shap_vals_all[class_idx])

    if shap_arr.ndim == 1:
        shap_row = shap_arr
    else:
        shap_row = shap_arr[0]

    tfidf_words = tfidf.get_feature_names_out()
    n_tfidf = len(tfidf_words)
    tfidf_shap = shap_row[:n_tfidf]

    safe_pairs = []
    for w, v in zip(tfidf_words, tfidf_shap):
        try:
            val = float(np.asarray(v).item())
        except Exception:
            val = float(np.squeeze(np.asarray(v)))
        safe_pairs.append((w, val))

    safe_pairs_sorted = sorted(safe_pairs, key=lambda x: abs(x[1]), reverse=True)
    return safe_pairs_sorted[:12]

@app.route("/", methods=["GET", "POST"])
def home():
    text = ""
    results = None

    if request.method == "POST":
        text = request.form.get("text", "")
        X = make_features([text])

        # Predictions
        pred_lr = int(lr.predict(X)[0])
        pred_rf = int(rf.predict(X)[0])

        prob_lr = lr.predict_proba(X)
        prob_rf = rf.predict_proba(X)
        pred_hybrid = int(np.argmax(prob_lr + prob_rf))

        # SHAP
        shap_words = explain_shap(text)
        probs = lr.predict_proba(X)[0]
        top_class = int(np.argmax(probs))
        
        lime_html = lime_explainer.explain_instance(
            text,
            lambda x: lr.predict_proba(make_features(x)),
            num_features=10,
            labels=[top_class],
        ).as_html()

        results = {
            "lr": LABELS[pred_lr],
            "rf": LABELS[pred_rf],
            "hybrid": LABELS[pred_hybrid],
            "shap": shap_words,
            "lime": lime_html
}

    return render_template_string(HTML, text=text, results=results)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
