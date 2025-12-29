# An Interactive Data Cleaning and Incremental Learning Framework

This repository implements an **interactive, human-in-the-loop data preprocessing and incremental learning framework** for structured datasets.  
The system supports **progressive user prompting**, **column-wise model training**, **real-time dataset updates**, and **automatic model retraining**.

The project is designed for **machine learning research, academic papers, and hackathon demos**.

---

## 🔍 Project Overview

Traditional machine learning pipelines rely on static datasets and one-time training.  
This project introduces an **adaptive workflow** where:

- Data is cleaned and preprocessed automatically  
- Each dataset column is modeled independently  
- Users can input data in **any order**  
- The system **suggests likely values** for remaining fields  
- Users can accept or override suggestions  
- Cleaned data can be appended to the dataset  
- Models are **retrained automatically** after updates  

This enables **continuous dataset enhancement** and **incremental learning**.

---

## ✨ Key Features

- Automated data cleaning and preprocessing  
- Column-wise supervised learning (one model per field)  
- Progressive, adaptive user prompting  
- Human-in-the-loop validation  
- Categorical decoding using stored encoders  
- Incremental dataset updates  
- Automatic retraining of all models  
- Model-agnostic and reusable pipeline  

---

## 📄 File Descriptions

### `preprocessing.py`
- Cleans the raw dataset  
- Handles missing values, duplicates, encoding, and scaling  
- Prepares data for model training  

---

### `train_csv_initial_model.py`
- Trains the **initial baseline model** from the CSV dataset  
- Used to establish first predictive capability  

---

### `train_models_per_column.py`
- Trains **one model per dataset column**  
- Saves models as:  
model_<column_name>.pkl

rust
Copy code
- Saves encoders for categorical columns as:  
encoder_<column_name>.pkl

yaml
Copy code
- Enables valid **column-wise conditional prediction**

---

### `interactive_update_and_retrain.py`
- Interactive CLI application  
- User can:
- Choose any field to start with  
- Receive intelligent suggestions  
- Accept or override inputs  
- Clean data automatically  
- Decide whether to add data to the dataset  

- If confirmed:
- Dataset is updated  
- All column-wise models are retrained automatically  

This is the **core innovation of the project**.

---

### `predict.py`
- Loads trained models  
- Makes predictions on new, unseen data  
- Demonstrates inference after training  

---

### `freelancer_earnings_vs_skillstack_dataset.csv`
- Structured dataset used for training and retraining  
- Ignored from version control if needed via `.gitignore`  

---

### `freelancer_earnings_model.pkl`
- Serialized trained model  
- Generated after training  
- Can be re-created anytime  

---

## ▶️ How to Run

### 1. Install dependencies
```bash
pip install pandas scikit-learn joblib
2. Preprocess the dataset
bash
Copy code
python preprocessing.py
3. Train column-wise models
bash
Copy code
python train_models_per_column.py
4. Run interactive prompting and retraining
bash
Copy code
python interactive_update_and_retrain.py
Run scripts in a terminal, not in an IDE output window.