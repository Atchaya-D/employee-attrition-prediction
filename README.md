Employee Attrition Prediction

This project is a machine learning based application that predicts whether an employee is likely to leave an organization. I built this project to understand how machine learning can be applied to HR data and how the results can be presented through a simple web application.

Project Overview

The project uses employee information such as age, job role, income, overtime, job satisfaction, work experience and other factors to predict employee attrition.

The data is first cleaned and prepared using Python. Categorical features are encoded and numerical features are scaled before training the machine learning model.

A Random Forest Classifier is used for prediction. The trained model is saved using Joblib and integrated into a Streamlit application.

What the application does

• Shows the total number of employees
• Shows the number of employees who left the organization
• Calculates the overall attrition rate
• Displays attrition based on department
• Displays attrition based on job role
• Displays attrition based on overtime
• Predicts attrition for a new employee
• Shows the probability of employee attrition

Technologies Used

• Python
• Pandas
• NumPy
• Scikit-learn
• Joblib
• Streamlit
• Jupyter Notebook

Machine Learning

The main machine learning model used in this project is Random Forest Classifier.

The preprocessing pipeline includes:

• Removing unnecessary columns
• Separating input features and target variable
• Encoding categorical features using One-Hot Encoding
• Scaling numerical features using StandardScaler
• Splitting the data into training and testing sets

The target variable is converted into:

0 – Employee stays

1 – Employee leaves

Project Structure

Employee-Attrition-Prediction/
¦
+-- app.py
+-- best_attrition_model.pkl
+-- requirements.txt
+-- README.md
+-- .gitignore
¦
+-- data/
¦   +-- WA_Fn-UseC_-HR-Employee-Attrition.csv
¦
+-- notebooks/
¦   +-- Data_preprocessing.ipynb
¦
+-- src/

How to Run the Project

1. Clone the repository

git clone https://github.com/YOUR_USERNAME/employee-attrition-prediction.git

2. Open the project folder

cd employee-attrition-prediction

3. Create a virtual environment

python -m venv venv

4. Activate the virtual environment on Windows

venv\Scripts\activate

5. Install the required libraries

pip install -r requirements.txt

6. Place the dataset inside the data folder.

The expected file name is:

WA_Fn-UseC_-HR-Employee-Attrition.csv

7. Run the Streamlit application

python -m streamlit run app.py

The application will open in the browser at:

http://localhost:8501

Streamlit: https://employee-attrition-prediction-dkbz4mjwnqt4zzhbxyagwb.streamlit.app/

Application Sections

Dashboard

The dashboard gives a quick overview of employee attrition. It displays employee count, employees who left, attrition rate and charts showing attrition across different departments, job roles and overtime categories.

Attrition Prediction

This section allows the user to enter employee details and get a prediction from the trained machine learning model. The application also displays the estimated probability of attrition.

Future Improvements

I plan to improve this project further by adding model comparison, feature importance visualizations, better dashboard interactions, explainable AI techniques and online deployment.

Author

Atchayapriya D

Data Science Student
