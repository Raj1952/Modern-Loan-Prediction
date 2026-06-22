from flask import Flask, render_template, request, redirect, url_for
import os
from backend.model import LoanSimulatorModel

app = Flask(__name__)

data_path = os.path.join(os.path.dirname(__file__), "data", "loan_approval_dataset.csv")
simulator = LoanSimulatorModel(data_path)
simulator.train()

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        # Get data from form
        try:
            form_data = {
                "income_annum": int(request.form.get("income_annum", 0)),
                "loan_amount": int(request.form.get("loan_amount", 0)),
                "loan_term": int(request.form.get("loan_term", 0)),
                "cibil_score": int(request.form.get("cibil_score", 0)),
                "education": request.form.get("education", " Graduate"),
                "self_employed": request.form.get("self_employed", " No"),
                "no_of_dependents": int(request.form.get("no_of_dependents", 0)),
                "bank_asset_value": int(request.form.get("bank_asset_value", 0)),
                "residential_assets_value": int(request.form.get("residential_assets_value", 0)),
                "commercial_assets_value": int(request.form.get("commercial_assets_value", 0)),
                "luxury_assets_value": int(request.form.get("luxury_assets_value", 0)),
            }
            result = simulator.predict_explainable(form_data)
        except Exception as e:
            print("Error processing form:", e)
            result = {"status": "Error", "message": "Failed to evaluate application."}

    return render_template('applicant.html', result=result)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
