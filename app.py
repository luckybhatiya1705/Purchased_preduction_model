import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# ------------------------------------------------------------------
# Load the trained Naive Bayes model
# ------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "naive_model.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ------------------------------------------------------------------
# HTML Template (styled inline with CSS — dark navy + teal/gold accent)
# ------------------------------------------------------------------
PAGE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Purchase Predictor | Naive Bayes</title>
<style>
  :root {
    --bg-dark: #0f172a;
    --bg-card: #1e293b;
    --accent: #14b8a6;
    --accent-hover: #0d9488;
    --gold: #f59e0b;
    --text-light: #e2e8f0;
    --text-muted: #94a3b8;
    --border-color: #334155;
    --success: #22c55e;
    --danger: #ef4444;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    background: radial-gradient(circle at top left, #1e293b, #0f172a 60%);
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 30px 15px;
    color: var(--text-light);
  }

  .wrapper {
    width: 100%;
    max-width: 480px;
  }

  .header {
    text-align: center;
    margin-bottom: 24px;
  }

  .header .badge {
    display: inline-block;
    background: rgba(20, 184, 166, 0.15);
    color: var(--accent);
    border: 1px solid var(--accent);
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 12px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 12px;
  }

  .header h1 {
    font-size: 28px;
    font-weight: 700;
    background: linear-gradient(90deg, var(--accent), var(--gold));
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
  }

  .header p {
    color: var(--text-muted);
    font-size: 14px;
    margin-top: 6px;
  }

  .card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 32px;
    box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
  }

  .form-group {
    margin-bottom: 20px;
  }

  label {
    display: block;
    font-size: 13px;
    font-weight: 600;
    color: var(--text-muted);
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  select, input[type="number"] {
    width: 100%;
    padding: 12px 14px;
    background: #0f172a;
    border: 1px solid var(--border-color);
    border-radius: 10px;
    color: var(--text-light);
    font-size: 15px;
    outline: none;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
  }

  select:focus, input[type="number"]:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px rgba(20, 184, 166, 0.2);
  }

  button {
    width: 100%;
    padding: 14px;
    margin-top: 8px;
    background: linear-gradient(90deg, var(--accent), var(--accent-hover));
    color: #ffffff;
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    border: none;
    border-radius: 10px;
    cursor: pointer;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
  }

  button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 20px -8px rgba(20, 184, 166, 0.5);
  }

  .result {
    margin-top: 24px;
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    font-size: 17px;
    font-weight: 700;
    letter-spacing: 0.3px;
    animation: fadeIn 0.4s ease;
  }

  .result.positive {
    background: rgba(34, 197, 94, 0.12);
    border: 1px solid var(--success);
    color: var(--success);
  }

  .result.negative {
    background: rgba(239, 68, 68, 0.12);
    border: 1px solid var(--danger);
    color: var(--danger);
  }

  .result .sub {
    display: block;
    margin-top: 4px;
    font-size: 12px;
    font-weight: 400;
    color: var(--text-muted);
    text-transform: none;
    letter-spacing: normal;
  }

  footer {
    text-align: center;
    margin-top: 20px;
    color: var(--text-muted);
    font-size: 12px;
  }

  footer a {
    color: var(--accent);
    text-decoration: none;
  }

  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
  }
</style>
</head>
<body>
  <div class="wrapper">
    <div class="header">
      <span class="badge">Gaussian Naive Bayes</span>
      <h1>Purchase Predictor</h1>
      <p>Estimate whether a customer will purchase based on demographics</p>
    </div>

    <div class="card">
      <form method="POST" action="/predict">
        <div class="form-group">
          <label for="gender">Gender</label>
          <select name="gender" id="gender" required>
            <option value="1" {{ 'selected' if gender == '1' else '' }}>Male</option>
            <option value="0" {{ 'selected' if gender == '0' else '' }}>Female</option>
          </select>
        </div>

        <div class="form-group">
          <label for="age">Age</label>
          <input type="number" name="age" id="age" min="1" max="120" placeholder="e.g. 32" value="{{ age or '' }}" required>
        </div>

        <div class="form-group">
          <label for="salary">Estimated Salary ($)</label>
          <input type="number" name="salary" id="salary" min="0" step="100" placeholder="e.g. 60000" value="{{ salary or '' }}" required>
        </div>

        <button type="submit">Predict</button>
      </form>

      {% if prediction is not none %}
        <div class="result {{ 'positive' if prediction == 1 else 'negative' }}">
          {% if prediction == 1 %}
            ✅ Likely to Purchase
          {% else %}
            ❌ Unlikely to Purchase
          {% endif %}
          <span class="sub">Confidence: {{ probability }}%</span>
        </div>
      {% endif %}
    </div>

    <footer>
      Powered by scikit-learn &middot; Flask &middot; Deployed on <a href="https://render.com" target="_blank">Render</a>
    </footer>
  </div>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def home():
    return render_template_string(PAGE_TEMPLATE, prediction=None, probability=None,
                                   gender=None, age=None, salary=None)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        gender = request.form.get("gender", "1")
        age = request.form.get("age")
        salary = request.form.get("salary")

        features = np.array([[float(gender), float(age), float(salary)]])

        prediction = int(model.predict(features)[0])
        proba = model.predict_proba(features)[0]
        probability = round(max(proba) * 100, 2)

        return render_template_string(
            PAGE_TEMPLATE,
            prediction=prediction,
            probability=probability,
            gender=gender,
            age=age,
            salary=salary,
        )
    except Exception as e:
        return render_template_string(
            PAGE_TEMPLATE,
            prediction=None,
            probability=None,
            gender=None,
            age=None,
            salary=None,
        ), 400


# Simple health check endpoint (useful for Render)
@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}, 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
