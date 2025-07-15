# Obesity Prediction - ML Prediction App.

# 🎯 **What You're Aiming For.**
This project demonstrates how to predict whether a person is obese or not using a high-dimensional dataset. The aim is to build a robust RandomForestClassifier model that accurately predicts obesity based on various features.

# ℹ️ **Instructions.**

1. Install the necessary packages:
   ```sh
   pip install pandas scikit-learn streamlit
   ```

2. Import your data and perform basic data exploration:
   ```python
   import pandas as pd
   data = pd.read_csv('path_to_your_dataset.csv')
   print(data.head())
   print(data.info())
   ```

3. Display general information about the dataset.
4. Create a pandas profiling report to gain insights into the dataset.
5. Handle missing and corrupted values.
6. Remove duplicates, if they exist.
7. Handle outliers, if they exist.
8. Encode categorical features:
   ```python
   from sklearn.preprocessing import LabelEncoder
   le = LabelEncoder()
   data['encoded_column'] = le.fit_transform(data['categorical_column'])
   ```

9. Based on the previous data exploration, train and test a machine learning classifier:
   ```python
   from sklearn.model_selection import train_test_split
   from sklearn.ensemble import RandomForestClassifier

   features = data.drop('target_column', axis=1)
   label = data['target_column']
   x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)
   model = RandomForestClassifier()
   model.fit(x_train, y_train)
   ```

10. Create a Streamlit application (locally) and add input fields for your features and a validation button at the end of the form.
11. Deploy your application on Streamlit Share:
    - Create a GitHub and a Streamlit Share account.
    - Create a new git repository.
    - Upload your local code to the newly created git repository.
    - Log in to your Streamlit account and deploy your application from the git repository.

🛠️ **Tools Used**
- Anaconda
- Jupyter Notebook
- Python
- VSCode
- GitHub
- Google Chrome
- Streamlit
