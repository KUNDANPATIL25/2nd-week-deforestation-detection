# Explanation of the Code and Concepts: Classification of Fire Types in India Using MODIS Data (2021–2023)

## Project Objective

To build a machine learning classification model that predicts the **type of fire** (e.g., MODIS, VIIRS) using MODIS fire detection data for India from 2021 to 2023. This involves:

* Cleaning and preprocessing the dataset
* Exploring spatial, temporal, and statistical patterns
* Engineering features
* Visualizing and understanding data distributions
* Addressing class imbalance
* Preparing the dataset for machine learning

---

## 1. Data Source & Domain Context

* **MODIS**: A NASA satellite sensor that detects thermal anomalies (fires) with 1km resolution.
* **FIRMS**: Fire Information for Resource Management System that provides MODIS datasets.

---

## 2. Libraries and Tools Used

* **Pandas, NumPy**: Data manipulation and array processing.
* **Matplotlib, Seaborn**: Data visualization.
* **Scikit-learn**: ML models, feature selection, data preprocessing.
* **XGBoost**: Boosted tree classifier.
* **Folium**: Interactive geographical maps.
* **Statsmodels, Scipy**: Statistical visualization.
* **Imbalanced-learn (SMOTE)**: Handling class imbalance.

---

## 3. Data Loading and Merging

```python
pd.read_csv()
pd.concat([...])
```

* Loaded fire incident CSVs for 2021, 2022, 2023.
* Combined using `concat()` to form one DataFrame.

---

## 4. Data Exploration & Cleaning

```python
df.info(), df.describe(), df.isnull().sum(), df.duplicated().sum()
```

* Checked for missing values, data types, duplicates.
* Summarized statistics (mean, std, min, max).

---

## 5. Exploratory Data Analysis (EDA)

### A. Categorical Plots

* `countplot()` for fire types, day/night, satellite, version.
* Shows distribution of categorical features.

### B. Numerical Plots

* `histplot()`, `boxplot()`, `scatterplot()`, `pairplot()`
* Shows distribution, relationships, and outliers.

### C. Heatmap of Correlation

* Analyzed linear correlation among features like brightness, frp, confidence, etc.

---

## 6. Temporal Feature Engineering

```python
df['acq_date'] = pd.to_datetime(...)
```

* Extracted `year`, `month`, `day_of_week`, `hour`, etc.
* Visualized patterns by month/day of week using `countplot()`.

---

## 7. Outlier Detection and Removal

* Used **Interquartile Range (IQR)** method:

```python
Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1
```

* Removed values beyond `1.5 * IQR`.
* Visualized before/after with `boxplot()`.

---

## 8. Encoding Categorical Features

```python
pd.get_dummies(..., drop_first=True)
```

* Applied **One-Hot Encoding** to: `daynight`, `satellite`, `instrument`
* Converted string categories into numeric format for model compatibility.

---

## 9. Geospatial Mapping

```python
folium.Map(), folium.CircleMarker()
```

* Visualized fire incidents on map of India.
* Sampled 10,000 rows for efficient rendering.
* Showed popup with FRP and date per fire.

---

## 10. Feature Scaling

```python
StandardScaler().fit_transform()
```

* Scaled numerical features to standard normal distribution (mean=0, std=1):

  * `brightness`, `scan`, `track`, `confidence`, `bright_t31`, `frp`

---

## 11. Feature Selection for Modeling

* Selected key features for modeling:

```python
features = ['brightness', 'scan', 'track', 'confidence', 'bright_t31', 'frp']
target = 'type'
```

* X = features, y = target

---

## 12. Handling Class Imbalance

### SMOTE (Synthetic Minority Oversampling Technique)

```python
from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = smote.fit_resample(X, y)
```

* Generates synthetic samples of minority class
* Balances the `type` class distribution

---

## 13. Terminologies Summary

* **MODIS**: NASA's Moderate Resolution Imaging Spectroradiometer
* **FRP**: Fire Radiative Power - indicates intensity
* **Brightness**: Infrared brightness from MODIS
* **Confidence**: Likelihood that a fire detection is real
* **SMOTE**: A technique to handle class imbalance by generating synthetic examples
* **One-Hot Encoding**: Convert categorical features into binary numeric columns
* **StandardScaler**: Normalizes features by removing mean and scaling to unit variance
* **Correlation**: Measure of linear relationship between two variables
* **Outliers**: Data points significantly different from the rest, treated with IQR
* **IQR**: Interquartile Range used to detect and remove outliers

---

## Outcome:

A clean, balanced, and visualized dataset ready for classification modeling. The preprocessing pipeline ensures the model learns meaningful patterns from MODIS satellite fire data to predict fire types with better accuracy and generalization.

---

## Next Steps (Beyond Code Shared):

* Train multiple ML models (e.g., Logistic Regression, Random Forest, XGBoost)
* Evaluate using metrics (Accuracy, Precision, Recall, F1-score, Confusion Matrix)
* Use hyperparameter tuning and cross-validation
* Deploy the trained model as a fire monitoring tool.

---

This notebook is an excellent demonstration of geospatial data science, remote sensing analytics, and machine learning preprocessing pipeline in action for a real-world problem like fire classification in India.
 git remote add origin https://github.com/KUNDANPATIL25/2nd-week-deforestation-detection.git
