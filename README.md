# Tour Package Sales & Customer Conversion Analysis

## Project Overview

This project analyzes customer data from a travel company to understand **tour package purchase behavior** and identify factors associated with customer conversion.

The project includes:

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Customer conversion-rate analysis
- Statistical hypothesis testing
- Correlation and outlier analysis
- Feature engineering
- Classification model comparison
- Interactive Streamlit dashboard
- Business-oriented customer insights

The dataset contains customer demographic, financial, travel, and sales-pitch information. The target variable is `ProdTaken`, which indicates whether a customer purchased a tour package (`1`) or not (`0`).

The cleaned analysis dataset contains **4,657 customers and 20 features**, of whom **898 customers (19.28%) purchased a package**.

---

##  Project Objectives

1. Understand customer characteristics associated with tour-package purchases.
2. Analyze conversion rates across different customer and product segments.
3. Identify important numerical and categorical factors related to package purchase.
4. Perform statistical tests to examine relationships between variables and package purchase.
5. Create additional features for customer analysis and machine learning.
6. Build and compare classification models for purchase prediction.
7. Develop an interactive dashboard for exploring customer conversion patterns.

---

##  Dataset

### Dataset Information

The dataset contains customer information related to a travel company's tour package sales.

**Dataset file:** `tour_package.csv`

**Records after cleaning:** 4,657

**Features:** 20

**Target variable:** `ProdTaken`

### Target Variable

| Value | Meaning |
|---|---|
| `0` | Package Not Purchased |
| `1` | Package Purchased |

### Main Features

- `CustomerID`
- `Age`
- `TypeofContact`
- `CityTier`
- `DurationOfPitch`
- `Occupation`
- `Gender`
- `NumberOfPersonVisiting`
- `NumberOfFollowups`
- `ProductPitched`
- `PreferredPropertyStar`
- `MaritalStatus`
- `NumberOfTrips`
- `Passport`
- `PitchSatisfactionScore`
- `OwnCar`
- `NumberOfChildrenVisiting`
- `Designation`
- `MonthlyIncome`
- `ProdTaken`

### Dataset Link

The dataset used in this project is the Travel Package Purchase Prediction / Visit with Us dataset.

Original/public dataset source:

kaggle.com/datasets/sanamps/tourpackageprediction

---

##  Data Cleaning

The data cleaning process includes:

- Checking missing values
- Checking duplicate rows
- Detecting inconsistent categorical values
- Correcting the `Fe Male` value to `Female`
- Filling missing `Age` values using the median
- Filling missing `MonthlyIncome` using designation-wise median values
- Filling missing `DurationOfPitch` using the median
- Filling missing `NumberOfFollowups` using the median
- Filling missing `PreferredPropertyStar` using the mode
- Handling extreme `MonthlyIncome` values
- Removing remaining rows containing missing values
- Validating the cleaned dataset

The cleaned dataset is saved as:

`tour_package_cleaned.csv`

---

##  Exploratory Data Analysis

The project performs EDA on both numerical and categorical variables.

### Analysis includes:

- Target variable distribution
- Numerical feature distributions
- Boxplots for outlier detection
- Categorical feature distributions
- Numerical features vs. package purchase
- Conversion rate by categorical variables
- Correlation heatmap
- Age distribution by purchase outcome
- Monthly income by designation
- Product pitched vs. package purchase
- City tier vs. package purchase
- Passport and own-car analysis
- Pairplot of selected features

EDA visualizations are saved in the:

`eda_plots/`

folder.

---

##  Statistical Analysis

The project performs several statistical analyses, including:

- Chi-Square tests
- T-Tests
- Pearson correlation analysis
- Outlier detection using the IQR method
- Cross-tabulation analysis
- Conversion-rate analysis
- Age-band analysis
- Follow-up analysis
- Income analysis by designation

The generated statistical reports are stored in:

`analysis_report/`

---

##  Feature Engineering

Additional features were created to support deeper customer analysis and machine learning.

### Engineered Features

- `DesignationRank`
- `IncomePerPerson`
- `EngagementScore`
- `TravelPropensity`
- `FamilySize`
- `IsMarried`
- `AgeGroup`
- `IncomeTier`
- `HighValue`
- `CompanyInvited`

These features were created using customer demographic, income, travel, engagement, and contact information.

---

##  Machine Learning

The project compares multiple classification algorithms:

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost

### Techniques Used

- Train-test split
- Stratified sampling
- Feature scaling
- Label encoding
- SMOTE for class imbalance
- Cross-validation
- ROC-AUC evaluation
- Precision-Recall analysis
- Confusion matrix
- Classification report
- SHAP for model interpretation

The trained pipeline and model-related artifacts are stored in:

`artifacts/`

---

##  Interactive Streamlit Dashboard

The project includes an interactive Streamlit dashboard for exploring customer conversion behavior.

### Dashboard Sections

-  Overview
-  Demographics
-  Product & Pitch
-  Income & Designation
- Correlations
-  Statistical Tests
-  Data Explorer

### Available Filters

- Gender
- Occupation
- City Tier
- Designation
- Marital Status
- Passport
- Monthly Income

### Run the Dashboard

```bash
streamlit run app.py
