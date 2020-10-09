# Importing essential libraries
from flask import Flask, render_template, request
import pickle

# Load the Multinomial Naive Bayes model and CountVectorizer object from disk
filename = 'restaurant-sentiment-mnb-model.pkl'
text_model = pickle.load(open(filename, 'rb'))
tfidf = pickle.load(open('tfidf-transform.pkl','rb'))

app1 = Flask(__name__)

@app1.route('/')
def home():
	return render_template('index.html')

@app1.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
    	message = request.form['message']
    	data = [message]
    	vect = tfidf.transform(data).toarray()
    	my_prediction = text_model.predict(vect)
    	return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app1.run(debug=True)
