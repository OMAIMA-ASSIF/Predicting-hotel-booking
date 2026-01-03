# 🏨 Hotel Booking Cancellation Predictor

This project is a **Machine Learning** web application designed to predict whether a hotel reservation will be canceled or fulfilled. By analyzing various factors such as lead time, price, and special requests, the model helps hotel managers anticipate cancellations and optimize their occupancy rates.

## 🚀 Overview

The project involves a complete pipeline:

1. **Data Analysis & Modeling:** Exploratory Data Analysis (EDA) and model training using a Logistic Regression approach with polynomial features.
2. **Deployment:** A web interface built with **Flask** that allows users to input booking details and receive an instant prediction.

## 🛠️ Tech Stack

* **Language:** Python 3.x
* **Machine Learning:** Scikit-Learn, Pandas, NumPy
* **Web Framework:** Flask
* **Frontend:** HTML5, CSS3 (Bootstrap)
* **Data Source:** `HotelReservations.csv`

## 📂 Project Structure

* `model.ipynb`: Jupyter Notebook containing data preprocessing, feature engineering, and model training.
* `deploy.py`: The Flask application script to serve the model.
* `model.sav`: The trained serialized model (exported using Pickle).
* `templates/`: Folder containing your HTML interface.
* `HotelReservations.csv`: The dataset containing booking details.

## 📊 Model Details

The prediction engine uses a **Logistic Regression** model. Key steps included:

* **Feature Engineering:** Implementation of `PolynomialFeatures` to capture non-linear relationships.
* **Scaling:** Data normalization using `StandardScaler` (Z-score) to ensure all features contribute equally.
* **Metrics:** The model was evaluated based on Accuracy and Confusion Matrix results to minimize false negatives (unpredicted cancellations).

## 🔧 Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/hotel-cancellation-predictor.git
cd hotel-cancellation-predictor

```


2. **Install dependencies:**
```bash
pip install flask pandas numpy scikit-learn

```


3. **Run the application:**
```bash
python deploy.py

```


4. **Access the app:**
Open your browser and navigate to `http://127.0.0.1:5000`
