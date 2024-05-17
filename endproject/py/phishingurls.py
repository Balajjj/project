import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


app = Flask(__name__)
CORS(app)


print("Reading CSV file...")
df = pd.read_csv('C:/Users/Kingb/OneDrive/Desktop/endproject/py/phishing_site_urls.csv')
print("CSV file read successfully!")

label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['Label'])


urls = df['URL']
tokens = [" ".join(url.split("/")) for url in urls]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(tokens)
y = df['label']



print("Training the model...")
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X, y)
print("Model trained successfully!")



@app.route('/predict', methods=['POST'])
def predict():

  
    data = request.json
    
 
    tokens = [" ".join(url.split("/")) for url in data['urls']]
    
 
    X = vectorizer.transform(tokens)
    
   
    predictions = rf_classifier.predict(X)

    labels = ['good' if pred == 0 else 'bad' for pred in predictions]
    
   
    return jsonify({'predictions': predictions.tolist()})


if __name__ == '__main__':
    app.run(debug=True)