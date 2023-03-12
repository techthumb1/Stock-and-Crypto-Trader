from flask import Flask, render_template, request, jsonify
#from flask_bootstrap import Bootstrap
#from flask_wtf import FlaskForm
#from wtforms import StringField, SubmitField
#from wtforms.validators import DataRequired
import gdown


# Create the app
app = Flask(__name__)

# Route for the home page
@app.route("/")
def home():
    return render_template('home.html')

# Route for to run colab notebook daily
@app.route("/run-colab")
def run_colab():
    gdown.download('https://colab.research.google.com/drive/1MLHV-gJ4_MbASHn26oEvWMj07QK2_8uv?usp=share_link', 'FinRL_Ensemble_StockTrading_ICAIF_2023.ipynb', quiet=False)
    return jsonify(message='colab notebook ran successfully')


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
 