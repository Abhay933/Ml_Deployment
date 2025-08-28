from flask import Flask, render_template, request
import pickle

application = Flask(_name_)
app = application

# Load your model and scaler
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# Root route → directly show prediction page
@app.route("/", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "POST":
        # Collect form data
        data = [
            float(request.form['Temperature']),
            float(request.form['RH']),
            float(request.form['Ws']),
            float(request.form['Rain']),
            float(request.form['FFMC']),
            float(request.form['DMC']),
            float(request.form['ISI']),
            float(request.form['Classes']),
            float(request.form['Region'])
        ]

        # Scale and predict
        scaled_data = standard_scaler.transform([data])
        prediction = ridge_model.predict(scaled_data)[0]

        # Show result on home.html
        return render_template("home.html", result=prediction)

    # For GET request → just show the form
    return render_template("home.html")


if_name_=="_main_":
app.run(host="0.0.0.0")

    