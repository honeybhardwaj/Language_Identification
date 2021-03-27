from flask import Flask, redirect, url_for, render_template, request
import pickle
from preprocess import *

app = Flask(__name__)

#loading model into app
filename = '../unigram_model_rfc.sav'
model = pickle.load(open(filename, 'rb'))

#routing
@app.route("/", methods=["POST", "GET"])
def home():
	if request.method == "POST":
		textinput = request.form["nm"]
		return redirect(url_for("predict", inp=textinput))
	else:
		return render_template("base.html")

@app.route("/<inp>")
def predict(inp):
	
	#converting input into feature vector
	user_input= uni_vector.transform([inp])
	a=user_input.toarray()
	user_input=pd.DataFrame(a, columns=uni_feature_names)

	# fit the model on input
	language = model.predict(user_input)  
	language=language[0]
	return render_template("predict.html", val=language)
	

if __name__ == "__main__":
    app.run(debug=True)