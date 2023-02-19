# Description: This is the main file for the Flask app. 
from flask import Flask, render_template, request
app = Flask(__name__)

# Route for the home page
@app.route("/")
def home():
    return render_template('home.html')


# Route for the prediction page
@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        stock_symbol = request.form['stock_symbol']
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        prediction_days = request.form['prediction_days']

        prediction = 0
        return render_template('prediction.html', prediction=prediction)
    else:
        return render_template('predict.html')

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
