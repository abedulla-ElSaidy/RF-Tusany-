#  Drought Prediction and Spatial Visualization — *Bruna River Generalization Example*  
**Author:** Elsaidy, Abedulla (2025)  
**Last Updated:** November 2025  

---

##  Overview
This code demonstrates a **machine learning–based drought prediction pipeline** applied to the *Bruna River Basin*.  
The workflow includes:
1. **Data preprocessing and feature generation**
2. **Feature set evaluation using Random Forest models**
3. **Model training and spatial prediction**
4. **Geospatial visualization of drought probability**

The implementation combines **hydrological**, **climatic**, **geographical**, and **seasonal** predictors, applying **Random Forests** and **SMOTE** to address class imbalance and improve generalization.

---


## Methodology

### 1. **Feature Engineering**
The dataset integrates multiple feature categories:

| Feature Type | Example Variables |
|---------------|------------------|
| **SPEI indices** | SPEI01, SPEI03, SPEI06, SPEI09, SPEI12 |
| **SPI indices**  | SPI01, SPI03, SPI06, SPI09, SPI12 |
| **Hydrological** | gr, if, rf, ws |
| **Weather** | ae, pr, td, tp |
| **Geographical** | Lat, Lon, Quota [m] |
| **Categorical** | Provincia, Codice |
| **Seasonal ** | sin_month, cos_month |

Categorical variables are ** encoded**, and **Month** is transformed into cyclic features (`sin_month`, `cos_month`).

---

### 2. **Model Training & Evaluation**
Each feature combination is evaluated using a **Random Forest Classifier** wrapped in an **imbalanced pipeline**:


**Performance metrics** include:
- Accuracy
- F1-Score
- ROC-AUC  
on both training and test datasets.

---

### 3. **Feature Set Comparison**
The script evaluates **13 feature sets**, ranging from full combinations to specialized subsets (e.g., “SPEI only”, “Weather + Geo”).  

```
outputs/drought_feature_set_results.csv
```

---

### 4. **Prediction for New Coordinates**
Once trained, the model can predict drought probability for new pixel locations (e.g., gridded points):

- Input: `ready.csv`  
- Output: `ready_predicted.csv`  
  (contains predicted drought labels and probabilities)

---

### 5. **Spatial Visualization**

- Input: `Mean Annual Drought Probability.csv`
- Shapefile: `shape file.shp`
- Output Map:  
  ```
  Mean_Annual_Drought_Probability.jpg
  ```

Each subplot represents a **yearly spatial distribution** of drought probabilities, normalized across all years.

---

##  Dependencies

### Python Environment
Python ≥ 3.9 recommended.


---

##  Citation
If you use or modify this work, please cite:

> Elsaidy, A. (2025). *Generalization of Machine Learning for Drought Prediction in the Bruna River Basin*, 2025.

---

##  License
This code is distributed under the **MIT License**.  
You are free to use, modify, and distribute it with proper attribution and citation.
