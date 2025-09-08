# House Prices Prediction

## 📌 Project Overview  
This project focuses on building **machine learning models** to predict house prices based on various features such as number of rooms, population, income levels, and location.  
The workflow includes **data exploration, preprocessing, visualization, model training, hyperparameter tuning, and evaluation**.  

The project was implemented in a **Jupyter Notebook** using Python libraries such as `pandas`, `numpy`, `matplotlib`, `seaborn`, and `scikit-learn`, along with **XGBoost**.

---

## 📂 Dataset  
The dataset used in this project is:  

- **File:** `housing.csv`  
- **Description:** Contains demographic, geographic, and housing-related features for California districts.  
- **Target Variable:** `median_house_value`  

### Key Features:
- `longitude`, `latitude` → geographic location  
- `housing_median_age` → median age of houses  
- `total_rooms`, `total_bedrooms`, `population`, `households` → demographic and housing info  
- `median_income` → median income of households  
- `ocean_proximity` → categorical feature describing location relative to the ocean  

---

## 🔎 Project Workflow  

### 1. Importing Libraries  
Essential Python libraries (`numpy`, `pandas`, `matplotlib`, `seaborn`, `sklearn`, `xgboost`) were imported.  

### 2. Data Loading & Initial Exploration  
- Loaded the dataset with `pandas.read_csv()`.  
- Displayed the first rows to understand structure.  
- Used `.info()`, `.describe()`, and `.shape` to check data size and summary.  

### 3. Data Cleaning & Preprocessing  
- Checked and handled **missing values**.  
- Removed duplicates if any.  
- Encoded categorical features (`ocean_proximity`).  
- Applied **feature scaling** (e.g., StandardScaler) for numerical features.  

### 4. Exploratory Data Analysis (EDA)  
- Distribution plots for features.  
- Correlation heatmap to identify relationships.  
- Scatter plots of key features against target (`median_house_value`).  

### 5. Splitting Data  
- Train-test split using `train_test_split()` (default 80/20 split).  

### 6. Model Training  
Implemented and compared multiple models:  
- **Linear Regression (LR)**  
- **Random Forest (RF)**  
- **XGBoost (XGB)**  

### 7. Hyperparameter Tuning  
- Applied **GridSearchCV** on the **Random Forest model** to optimize hyperparameters (e.g., number of trees, depth).  

### 8. Predictions & Evaluation  
- Made predictions on the test set.  
- Evaluated with metrics:  
  - Mean Squared Error (MSE)  
  - Root Mean Squared Error (RMSE)  
  - R² Score  

---

## 📊 Results  
- **Linear Regression** provided a baseline performance.  
- **Random Forest** outperformed LR with better accuracy and lower errors.  
- **XGBoost** achieved competitive results, capturing complex relationships.  
- **GridSearchCV** improved the Random Forest performance by tuning hyperparameters.  
- Final comparison of models was based on MSE, RMSE, and R².  

---


   git clone https://github.com/yourusername/house-prices-prediction.git
   cd house-prices-prediction
