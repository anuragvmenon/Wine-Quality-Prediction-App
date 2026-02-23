# Wine-Quality-Prediction-App
A machine learning Streamlit app that predicts wine quality

## Overview
This project is an interactive, end-to-end machine learning web application that predicts the quality score of a wine based on its chemical composition. By adjusting sliders for 11 different physicochemical properties (like acidity, residual sugar, and alcohol content), users can instantly see the model's quality prediction.

## 🚀 Key Features
* **Interactive Web App:** A user-friendly frontend built with **Streamlit** that allows anyone to easily input data and see predictions without needing to code.
* **Robust Machine Learning:** Powered by an **XGBoost Classifier**, selected after rigorously testing and comparing eight different algorithms (including Random Forest, SVM, and Logistic Regression).
* **Advanced Data Handling:** Implements **SMOTE** (Synthetic Minority Over-sampling Technique) during the training phase to address heavy class imbalance in the original dataset, ensuring a fairer, more accurate model.
* **Full Data Pipeline:** Includes complete data preprocessing scripts (scaling with `StandardScaler` and label encoding) that seamlessly connect the raw input from the web app to the trained model.

## 🛠️ Tech Stack
* **Frontend:** Streamlit
* **Machine Learning:** XGBoost, Scikit-Learn, Imbalanced-Learn (SMOTE)
* **Data Manipulation:** Pandas, NumPy
* **Model Deployment:** Joblib (for saving/loading the model, scaler, and encoder)

## 💻 How to Run Locally

Follow these steps to set up and run the application on your own machine.

**1. Clone the repository:**
```bash
git clone [https://github.com/YourUsername/your-repo-name.git](https://github.com/YourUsername/your-repo-name.git)
cd your-repo-name
2. Install the required dependencies:

Bash
pip install pandas joblib scikit-learn xgboost imbalanced-learn streamlit
3. Train the model:
The app relies on a pre-trained model. Run the training script to generate the necessary .pkl files inside a new model directory. Make sure WineQT.csv is in the main folder.

Bash
python train_model.py
4. Launch the app:
Start the Streamlit server to interact with the web interface.

Bash
streamlit run app.py
