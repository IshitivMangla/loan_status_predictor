import os
import joblib
import pandas as pd
from django.shortcuts import render

# Absolute path to model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "loan_approval_pipeline.pkl")

# Load model
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print("Model not loaded:", e)
    model = None


def index(request):
    result = None
    error = None

    if request.method == "POST":
        try:
            data = {
                "person_age": [int(request.POST.get("person_age"))],
                "person_education": [request.POST.get("person_education")],
                "person_income": [float(request.POST.get("person_income"))],
                "person_emp_exp": [float(request.POST.get("person_emp_exp"))],
                "person_home_ownership": [request.POST.get("person_home_ownership")],
                "loan_amnt": [float(request.POST.get("loan_amnt"))],
                "loan_intent": [request.POST.get("loan_intent")],
                "cb_person_cred_hist_length": [float(request.POST.get("cb_person_cred_hist_length"))],
                "credit_score": [float(request.POST.get("credit_score"))],
                "previous_loan_defaults_on_file": [request.POST.get("previous_loan_defaults_on_file")],
            }

            new_data = pd.DataFrame(data)

            # If previous default = yes, reject directly
            if data["previous_loan_defaults_on_file"][0].lower() == "yes":
                result = "❌ Loan Rejected (Previous Default Found)"
            else:
                if model:
                    pred = model.predict(new_data)[0]
                    result = "✅ Loan Approved" if pred == 1 else "❌ Loan Rejected"
                else:
                    error = "Model not loaded properly."

        except Exception as e:
            error = f"Error in prediction: {e}"

    return render(request, "predictor/index.html", {"result": result, "error": error})
