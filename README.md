Create virtualenv and install:

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt


Train models (this will download AG News and sentence-transformers weights):

python train.py


Run Flask app:

python app.py
# Or run with gunicorn for production:
# gunicorn -w 4 app:app


Example predict (curl):

curl -X POST -H "Content-Type: application/json" \
  -d '{"text": "Apple releases new iPhone with improved camera and battery life."}' \
  http://127.0.0.1:5000/predict