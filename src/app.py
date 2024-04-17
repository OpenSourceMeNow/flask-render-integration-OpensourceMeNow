from flask import Flask, request, render_template
from joblib import load
import traceback

app = Flask(__name__)

try:
    model = load("../models/decision_tree_classifier_default_42.sav")
except Exception as e:
    print(f"Error loading the model: {str(e)}")
    print(traceback.format_exc())
    model = None

class_dict = {
    "0": "Iris setosa",
    "1": "Iris versicolor",
    "2": "Iris virginica"
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Obtain values from form
            val1 = float(request.form["val1"])
            val2 = float(request.form["val2"])
            val3 = float(request.form["val3"])
            val4 = float(request.form["val4"])

            data = [[val1, val2, val3, val4]]
            prediction = str(model.predict(data)[0])
            pred_class = class_dict[prediction]
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            print(traceback.format_exc())
            pred_class = "Error occurred during prediction"
    else:
        pred_class = None

    return render_template("index.html", prediction=pred_class)