from flask import Flask, request, render_template, redirect, url_for, send_file
import pandas as pd
import pickle
import os
import re

# Initialize Flask app
app = Flask(__name__)

# Load models and vectorizer
with open('models/lr_label_model.pkl', 'rb') as label_model_file:
    lr_label_model = pickle.load(label_model_file)

with open('models/lr_subclass_model.pkl', 'rb') as subclass_model_file:
    lr_subclass_model = pickle.load(subclass_model_file)

with open('models/tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)


# Function to clean Marathi text
def clean_marathi_text(text):
    # Use a regular expression to remove anything that is not a Marathi character
    marathi_only = re.sub(r'[^\u0900-\u097F\s]', '', text)  # Unicode range for Marathi characters
    marathi_only = re.sub(r'\s+', ' ', marathi_only).strip()  # Remove extra spaces
    return marathi_only


# Index route for rendering the form
@app.route('/')
def index():
    return render_template('index.html')


# Route for handling text classification
@app.route('/classify-text', methods=['POST'])
def classify_text():
    if request.method == 'POST':
        # Get the text input from the form
        input_text = request.form['text']

        # Preprocess and vectorize input text
        input_text_cleaned = clean_marathi_text(input_text)
        input_vectorized = tfidf_vectorizer.transform([input_text_cleaned])

        # Predict using both models
        label_prediction = lr_label_model.predict(input_vectorized)[0]
        subclass_prediction = lr_subclass_model.predict(input_vectorized)[0]

        # Display results
        return render_template('index.html', text=input_text,
                               predicted_label=label_prediction,
                               predicted_subclass=subclass_prediction)


# Route for handling CSV uploads
@app.route('/classify-csv', methods=['POST'])
def classify_csv():
    if request.method == 'POST':
        # Check if a file is uploaded
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            # Save the uploaded file
            file_path = os.path.join('uploads', uploaded_file.filename)
            uploaded_file.save(file_path)

            # Read the uploaded CSV file
            try:
                df = pd.read_csv(file_path)

                # Normalize the column names
                df.columns = df.columns.str.strip().str.lower()  # Convert to lower case and strip whitespace

                if 'text' in df.columns:
                    # Clean and vectorize the text data
                    df['cleaned_text'] = df['text'].apply(clean_marathi_text)
                    X_vectorized = tfidf_vectorizer.transform(df['cleaned_text'])

                    # Predict using both models
                    df['Predicted_Label'] = lr_label_model.predict(X_vectorized)
                    df['Predicted_Subclass'] = lr_subclass_model.predict(X_vectorized)

                    # Save the result to a new CSV file
                    output_file = os.path.join('uploads', 'classified_' + uploaded_file.filename)
                    df.to_csv(output_file, index=False)

                    # Provide the file for download
                    return send_file(output_file, as_attachment=True)

                else:
                    return "Error: 'Text' column not found in the uploaded CSV.", 400

            except pd.errors.ParserError as e:
                return f"Error reading the CSV file: {e}", 400

    return redirect(url_for('index'))


# Run the app
if __name__ == "__main__":
    app.run(debug=True)

# from flask import Flask, request, render_template, redirect, url_for, send_file
# import pandas as pd
# import pickle
# import os
#
# # Initialize Flask app
# app = Flask(__name__)
#
# # Load models and vectorizer
# with open('models/lr_label_model.pkl', 'rb') as label_model_file:
#     lr_label_model = pickle.load(label_model_file)
#
# with open('models/lr_subclass_model.pkl', 'rb') as subclass_model_file:
#     lr_subclass_model = pickle.load(subclass_model_file)
#
# with open('models/tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
#     tfidf_vectorizer = pickle.load(vectorizer_file)
#
# # Index route for rendering the form
# @app.route('/')
# def index():
#     return render_template('index.html')
#
# # Route for handling text classification
# @app.route('/classify-text', methods=['POST'])
# def classify_text():
#     if request.method == 'POST':
#         # Get the text input from the form
#         input_text = request.form['text']
#
#         # Preprocess and vectorize input text
#         input_text_cleaned = clean_marathi_text(input_text)
#         input_vectorized = tfidf_vectorizer.transform([input_text_cleaned])
#
#         # Predict using both models
#         label_prediction = lr_label_model.predict(input_vectorized)[0]
#         subclass_prediction = lr_subclass_model.predict(input_vectorized)[0]
#
#         # Display results
#         return render_template('index.html', text=input_text,
#                                predicted_label=label_prediction,
#                                predicted_subclass=subclass_prediction)
#
# # Route for handling CSV uploads
# @app.route('/classify-csv', methods=['POST'])
# def classify_csv():
#     if request.method == 'POST':
#         # Check if a file is uploaded
#         uploaded_file = request.files['file']
#         if uploaded_file.filename != '':
#             # Save the uploaded file
#             file_path = os.path.join('uploads', uploaded_file.filename)
#             uploaded_file.save(file_path)
#
#             # Read the uploaded CSV file
#             df = pd.read_csv(file_path)
#
#             # Clean and vectorize the text data
#             df['cleaned_text'] = df['Text'].apply(clean_marathi_text)
#             X_vectorized = tfidf_vectorizer.transform(df['cleaned_text'])
#
#             # Predict using both models
#             df['Predicted_Label'] = lr_label_model.predict(X_vectorized)
#             df['Predicted_Subclass'] = lr_subclass_model.predict(X_vectorized)
#
#             # Save the result to a new CSV file
#             output_file = os.path.join('uploads', 'classified_' + uploaded_file.filename)
#             df.to_csv(output_file, index=False)
#
#             # Provide the file for download
#             return send_file(output_file, as_attachment=True)
#
#     return redirect(url_for('index'))
#
# # Function to clean Marathi text
# def clean_marathi_text(text):
#     import re
#     # Use a regular expression to remove anything that is not a Marathi character
#     marathi_only = re.sub(r'[^\u0900-\u097F\s]', '', text)  # Unicode range for Marathi characters
#     marathi_only = re.sub(r'\s+', ' ', marathi_only).strip()  # Remove extra spaces
#     return marathi_only
#
# # Run the app
# if __name__ == "__main__":
#     app.run(debug=True)


# from flask import Flask, render_template, request, redirect, url_for, send_file
# import pandas as pd
# import pickle
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# import re
# import matplotlib.pyplot as plt
# import os
#
# # Load pre-trained models and vectorizer
# with open('lr_label_model.pkl', 'rb') as label_file:
#     lr_label = pickle.load(label_file)
#
# with open('lr_subclass_model.pkl', 'rb') as subclass_file:
#     lr_subclass = pickle.load(subclass_file)
#
# with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
#     tfidf_vectorizer = pickle.load(vectorizer_file)
#
# # Initialize Flask app
# app = Flask(__name__)
#
# # Function to clean and process text (removing non-Marathi characters)
# def clean_marathi_text(text):
#     marathi_only = re.sub(r'[^\u0900-\u097F\s]', '', text)
#     marathi_only = re.sub(r'\s+', ' ', marathi_only).strip()
#     return marathi_only
#
# # Function to predict label and subclass
# def predict_hate_speech(text):
#     cleaned_text = clean_marathi_text(text)
#     vectorized_text = tfidf_vectorizer.transform([cleaned_text])
#     label_pred = lr_label.predict(vectorized_text)[0]
#     subclass_pred = lr_subclass.predict(vectorized_text)[0] if label_pred == 1 else 0
#     return label_pred, subclass_pred
#
# # Route for the home page
# @app.route('/')
# def index():
#     return render_template('index.html')
#
# # Route to handle text input and CSV file uploads
# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'text_input' in request.form and request.form['text_input']:
#         # Single text input prediction
#         text_input = request.form['text_input']
#         label_pred, subclass_pred = predict_hate_speech(text_input)
#         return render_template('result.html', text=text_input, label=label_pred, subclass=subclass_pred)
#
#     elif 'file' in request.files and request.files['file'].filename.endswith('.csv'):
#         # CSV file input prediction
#         file = request.files['file']
#         df = pd.read_csv(file)
#
#         # Clean the text column
#         df['cleaned_text'] = df['Text'].apply(clean_marathi_text)
#
#         # Vectorize and make predictions
#         X_tfidf = tfidf_vectorizer.transform(df['cleaned_text'])
#         df['Hate_Speech_Classification'] = lr_label.predict(X_tfidf)
#         df['Subclass_Prediction'] = df['Hate_Speech_Classification'].apply(lambda x: lr_subclass.predict([X_tfidf[i]])[0] if x == 1 else 0 for i in range(len(X_tfidf)))
#
#         # Save results as a new CSV file
#         output_file = 'predictions_output.csv'
#         df.to_csv(output_file, index=False)
#
#         # Generate a pie chart for subclass distribution
#         subclass_counts = df['Subclass_Prediction'].value_counts()
#         labels = [f'Subclass {i}' for i in subclass_counts.index]
#         plt.figure(figsize=(8, 6))
#         plt.pie(subclass_counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
#         plt.title('Subclass Distribution of Hate Speech Predictions')
#         pie_chart_file = 'subclass_distribution.png'
#         plt.savefig(pie_chart_file)
#
#         # Return the downloadable CSV and show the pie chart
#         return render_template('result.html', csv_file=output_file, pie_chart=pie_chart_file)
#
#     else:
#         return redirect(url_for('index'))
#
# # Route to download the generated CSV
# @app.route('/download/<filename>')
# def download(filename):
#     return send_file(filename, as_attachment=True)
#
# # Start the Flask app
# if __name__ == '__main__':
#     app.run(debug=True)
