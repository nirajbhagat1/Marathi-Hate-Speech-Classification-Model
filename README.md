# Marathi Hate Speech Classification Model

**October 2024**

## Overview

This project involves the development and deployment of a web-based Marathi hate speech classification model. The model aims to identify and classify hate speech in Marathi text, providing insights into various types of hate speech present in user inputs.

## Tools and Technologies

- **Programming Language**: Python
- **Web Framework**: Flask
- **Machine Learning**: Logistic Regression
- **Data Handling**: Pandas
- **Data Visualization**: Matplotlib

## Features

- **Hate Speech Detection**: The model detects hate speech in Marathi text and categorizes it into specific subclasses, including:
  - Defamatory/Insulting
  - Gender Abusive
  - Racist
  - Homophobic
  - And more (10 subclasses in total)

- **Text and CSV Upload**: Users can upload individual text or CSV files for batch classification.

- **CSV Output**: The application generates downloadable CSV files with predictions that include:
  - Hate Speech Classification (Hate Speech or Not Hate Speech)
  - Subclass Prediction (specific subclass type)

- **Visualization**: Integrated pie chart visualization to display the distribution of subclasses in uploaded CSV files.










Here’s the complete installation process for your Marathi Hate Speech Classification Model, all in one text box:

```markdown
## Installation Process

To set up and run the Marathi Hate Speech Classification Model locally, follow these steps:

### Prerequisites

Before you begin, ensure you have the following installed on your system:

- **Python**: Version 3.7 or higher. You can download it from [python.org](https://www.python.org/downloads/).
- **pip**: Python package installer, usually comes with Python installations.

### Step-by-Step Installation

1. **Clone the Repository**

   Open your terminal or command prompt and clone the repository using the following command:

   ```bash
   git clone https://github.com/nirajbhagat1/Marathi-Hate-Speech-Classification-Model.git
   ```

   Navigate into the cloned repository:

   ```bash
   cd Marathi-Hate-Speech-Classification-Model
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   It’s a good practice to use a virtual environment to manage dependencies for your project. You can create one using the following commands:

   For Windows:

   ```bash
   python -m venv venv
   ```

   For macOS/Linux:

   ```bash
   python3 -m venv venv
   ```

   Activate the virtual environment:

   For Windows:

   ```bash
   venv\Scripts\activate
   ```

   For macOS/Linux:

   ```bash
   source venv/bin/activate
   ```

3. **Install Required Packages**

   Ensure your virtual environment is activated, then install the required packages using pip. If a `requirements.txt` file is available, run:

   ```bash
   pip install -r requirements.txt
   ```

   If you don’t have a `requirements.txt` file, you can manually install the necessary libraries with the following command:

   ```bash
   pip install Flask pandas matplotlib scikit-learn
   ```

   (Add any other dependencies as needed based on your project.)

4. **Run the Flask Application**

   Start the Flask server by running:

   ```bash
   python app.py
   ```

   You should see output indicating that the server is running, typically something like:

   ```
   * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
   ```

5. **Access the Application**

   Open your web browser and go to the following URL:

   ```
   http://127.0.0.1:5000
   ```

   This will open the web interface for your Marathi Hate Speech Classification Model.

6. **Using the Application**

   - Upload a text file or a CSV file containing Marathi text.
   - Click on the "Classify" button to process the input.
   - Once the classification is complete, download the resulting CSV file with predictions.

### Additional Notes

- If you encounter any issues related to package installations, ensure that your `pip` is updated:

  ```bash
  pip install --upgrade pip
  ```

- Make sure your environment variables are set up correctly if you're using any specific configurations for your Flask app.


#DEMO VIDEO
<iframe src="https://www.linkedin.com/embed/feed/update/urn:li:ugcPost:7247526508775321600?compact=1" height="399" width="710" frameborder="0" allowfullscreen="" title="Embedded post"></iframe>

https://www.linkedin.com/posts/nirajbhagat7803_machinelearning-hatespeechdetection-marathilanguage-activity-7247526587212951554-aLrT?utm_source=share&utm_medium=member_desktop
