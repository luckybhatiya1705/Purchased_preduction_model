from flask import Flask, request, render_template_string
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load the model
# Using the exact filename provided: vaive_model.pkl
MODEL_PATH = 'naive_model.pkl'
model = None

try:
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    print(f"Error loading model: {e}")

# HTML & CSS Template embedded directly in the app
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Prediction</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Poppins', sans-serif;
        }
        
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.95);
            padding: 40px;
            border-radius: 16px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            width: 100%;
            max-width: 450px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .header h2 {
            color: #333;
            font-weight: 600;
            margin-bottom: 10px;
        }
        
        .header p {
            color: #666;
            font-size: 0.9em;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 8px;
            color: #444;
            font-weight: 600;
            font-size: 0.9em;
        }
        
        .form-control {
            width: 100%;
            padding: 12px 15px;
            border: 1px solid #ddd;
            border-radius: 8px;
            font-size: 1em;
            transition: border-color 0.3s ease;
        }
        
        .form-control:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 5px rgba(102, 126, 234, 0.5);
        }
        
        .btn {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(118, 75, 162, 0.4);
        }
        
        .result-box {
            margin-top: 25px;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            font-weight: 600;
            font-size: 1.1em;
        }
        
        .result-success {
            background-color: #e8f5e9;
            color: #2e7d32;
            border: 1px solid #c8e6c9;
        }
        
        .result-error {
            background-color: #ffebee;
            color: #c62828;
            border: 1px solid #ffcdd2;
        }
    </style>
</head>
<body>

    <div class="card">
        <div class="header">
            <h2>Prediction System</h2>
            <p>Enter the details below to generate a prediction</p>
        </div>
        
        <form method="POST">
            <div class="form-group">
                <label for="gender">Gender</label>
                <select id="gender" name="gender" class="form-control" required>
                    <option value="" disabled selected>Select Gender</option>
                    <option value="1">Male</option>
                    <option value="0">Female</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="age">Age</label>
                <input type="number" id="age" name="age" class="form-control" placeholder="e.g., 25" required step="any">
            </div>
            
            <div class="form-group">
                <label for="salary">Estimated Salary</label>
                <input type="number" id="salary" name="salary" class="form-control" placeholder="e.g., 50000" required step="any">
            </div>
            
            <button type="submit" class="btn">Predict Now</button>
        </form>

        {% if prediction_text %}
            <div class="result-box result-success">
                {{ prediction_text }}
            </div>
        {% elif error_text %}
            <div class="result-box result-error">
                {{ error_text }}
            </div>
        {% endif %}
    </div>

</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = None
    error_text = None

    if request.method == 'POST':
        if model is None:
            error_text = "Error: Model not loaded correctly on the server."
            return render_template_string(HTML_TEMPLATE, error_text=error_text)

        try:
            # Extract values from form
            gender = int(request.form['gender'])
            age = float(request.form['age'])
            salary = float(request.form['salary'])

            # Create a DataFrame to match the model's expected 'feature_names_in_'
            input_data = pd.DataFrame(
                [[gender, age, salary]], 
                columns=['Gender', 'Age', 'EstimatedSalary']
            )

            # Make prediction
            prediction = model.predict(input_data)[0]

            # Format the output (assuming binary 0/1 outcome based on standard models)
            # You can change the "Class 0" and "Class 1" labels to match your specific dataset
            result = "Positive (1)" if prediction == 1 else "Negative (0)"
            
            prediction_text = f"Prediction Result: {result}"

        except Exception as e:
            error_text = f"An error occurred during prediction: {str(e)}"

    return render_template_string(
        HTML_TEMPLATE, 
        prediction_text=prediction_text, 
        error_text=error_text
    )

if __name__ == "__main__":
    # Render binds to the PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
