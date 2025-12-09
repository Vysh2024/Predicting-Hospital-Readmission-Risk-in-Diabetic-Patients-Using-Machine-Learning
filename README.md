# Predicting-Hospital-Readmission-Risk-in-Diabetic-Patients-Using-Machine-Learning
Hospital readmission within 30 days is a key quality indicator in healthcare. Diabetic patients are especially vulnerable due to the chronic nature of the disease. This project builds a machine learning model to predict the risk of readmission.
To identify diabetic patients who are likely to be readmitted within 30 days using patient
demographics, medical history, and treatment features.
# **Predicting Hospital Readmission Risk in Diabetic Patients Using Machine Learning**

## **1. Project Overview**

Hospital readmission is a major challenge in healthcare, especially for diabetic patients who often require continuous monitoring.
This project builds a machine learning model to predict whether a diabetic patient is likely to be readmitted to the hospital within 30 days after discharge.
The goal is to assist hospitals in improving patient follow-up care and reducing preventable readmissions.

---

## **2. Objectives**

* To analyze a real-world medical dataset containing patient demographics, diagnoses, and treatment details.
* To clean and preprocess healthcare records for model development.
* To build and compare different machine learning models for readmission prediction.
* To identify the most important factors contributing to readmission risk.
* To support healthcare decision-making with data-driven insights.

---

## **3. Dataset Information**

**Dataset Name:** Diabetes 130-US Hospitals for Years 1999–2008
**Source:** UCI Machine Learning Repository
**Records:** 17,000+ patient encounters
**Features:** More than 50 medical, demographic, and treatment-related variables
**Target Variable:**

* `<30`  – readmitted within 30 days
* `>30` – readmitted after 30 days
* `NO`  – not readmitted

**Dataset Access Links:**
Primary UCI Link: [https://archive.ics.uci.edu/dataset/296/diabetes+130us+hospitals+for+years+19992008](https://archive.ics.uci.edu/dataset/296/diabetes+130us+hospitals+for+years+19992008)

---

## **4. Tools and Technologies**

* Python
* Pandas and NumPy for data manipulation
* Matplotlib and Seaborn for visualization
* Scikit-learn for machine learning models
* Google Colab / Jupyter Notebook

---

## **5. Data Preprocessing**

Key preprocessing steps included:

* Handling missing values and replacing “?” entries
* Removing irrelevant identifier columns
* Encoding categorical variables
* Normalizing numerical features
* Converting the target variable into binary classes

  * 1 = readmitted within 30 days
  * 0 = not readmitted
* Creating additional meaningful features such as total_visits
* Splitting the dataset into training and testing sets

These steps ensure that the dataset is clean, consistent, and ready for machine learning.

---

## **6. Exploratory Data Analysis (EDA)**

Exploratory analysis helped understand:

* The distribution of readmitted vs. non-readmitted patients
* Trends in age, gender, and race
* Length of hospital stay
* Number of procedures and medications
* Correlations between numerical features
* Patterns related to diagnosis codes

Visualizations included histograms, bar plots, scatter plots, and heatmaps.

---

## **7. Machine Learning Models**

Three machine learning models were developed and compared:

1. Logistic Regression
2. Random Forest Classifier
3. K-Nearest Neighbors (KNN)

Each model was trained on 80% of the data and evaluated on the remaining 20%.

---

## **8. Model Performance**

| Model               | Accuracy |
| ------------------- | -------- |
| Random Forest       | 63.13%   |
| Logistic Regression | 62.10%   |
| KNN                 | 57%      |

The Random Forest model achieved the best overall performance and was selected as the final model.

---

## **9. Conclusion**

This project demonstrates that machine learning can be used to predict hospital readmission risk among diabetic patients.
Although the accuracy is moderate, the model provides useful insights into factors affecting readmission, such as number of medications, patient age, number of visits, and length of hospital stay.

The Random Forest model performed better than other models and can be used as a baseline for further improvement.

This system can help hospitals:

* Identify high-risk diabetic patients
* Improve patient follow-up strategies
* Reduce hospital readmission rates
* Optimize allocation of medical resources

---

## **10. Future Enhancements**

Several improvements can further increase model performance:

* Apply SMOTE or other sampling techniques to handle class imbalance
* Use advanced algorithms such as XGBoost or LightGBM
* Incorporate additional clinical features if available
* Build a web-based prediction interface using Flask or Streamlit
* Deploy the model in a hospital management system

---

## **11. How to Run the Project**

1. Install required libraries:

```
pip install pandas numpy matplotlib seaborn scikit-learn
```

2. Open the Jupyter notebook or Google Colab file.

3. Run the cells sequentially to preprocess the data, train models, and generate results.

