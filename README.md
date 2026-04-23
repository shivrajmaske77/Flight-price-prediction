# End-to-End Data Science: Flight Ticket Price Prediction (Regression)

Welcome to the **Flight Ticket Price Prediction** project! You'll walk through the complete data science lifecycle — from raw data to a deployed web application that predicts Indian domestic flight prices in real time.

---

##  Learning Objectives

1. Perform **Exploratory Data Analysis (EDA)** on flight booking data.
2. **Clean and engineer features** (time-of-day encoding, stop mapping, label encoding).
3. Build and evaluate a **Random Forest Regressor** using Scikit-Learn.
4. Deploy the trained model using a **full-stack Django** web application with a sleek, glassmorphic UI.

---

##  Project Structure

```
flight_price_prediction/
├── data/
│   └── flight_price.csv          ← Raw dataset (airline, route, class, price…)
├── notebooks/
│   └── flight_price_prediction.ipynb  ← Full EDA + model training notebook
├── models/
│   └── flight_price_rf_model.joblib   ← Saved model (generated after training)
├── webapp/
│   ├── manage.py
│   ├── flightproject/            ← Django project settings & URLs
│   └── predictor/                ← Django app (views, templates, static)
├── train_model.py                ← Quick-train script (skip the notebook)
├── requirements.txt
└── .gitignore
```

---

##  Dataset Features

| Feature           | Description                                     |
|-------------------|-------------------------------------------------|
| `airline`         | Airline carrier name                            |
| `source_city`     | Departure city                                  |
| `destination_city`| Arrival city                                    |
| `departure_time`  | Time-of-day category (Morning, Evening, etc.)   |
| `arrival_time`    | Time-of-day category of arrival                 |
| `stops`           | Number of stops (0, 1, 2 or more)               |
| `travel_class`    | Economy or Business                             |
| `duration`        | Flight duration in hours                        |
| `days_left`       | Days between booking date and departure         |
| `price` *(target)*| Ticket price in INR (₹)                        |

---

##  Phase 1: The Data Science Lifecycle

### 1. Setup the Environment

It is best practice to use a virtual environment:

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

>  Your terminal prompt will be prefixed with `(venv)` when the environment is active.

---

### 2. Run the Jupyter Notebook

Launch the notebook to walk through EDA and training step by step:

```bash
jupyter notebook notebooks/flight_price_prediction.ipynb
```

The notebook covers:
- **EDA** — price distributions, airline comparisons, correlation heatmaps
- **Feature Engineering** — encoding time-of-day, stops, categorical variables
- **Model Training** — `RandomForestRegressor` with 200 trees
- **Evaluation** — MAE, RMSE, R², residual plots, feature importance
- **Export** — saves `models/flight_price_rf_model.joblib`

>  **Shortcut:** Skip the notebook and train instantly with:
> ```bash
> python train_model.py
> ```

---

##  Phase 2: Deploying with Django

Once your model is saved, launch the web application.

### 1. Run the Django Server

```bash
cd webapp
python manage.py runserver
```

### 2. View the App

Open your browser and go to: **http://127.0.0.1:8000/**

Fill in the flight details:
- **Airline, Source & Destination Cities**
- **Departure / Arrival time-of-day**
- **Number of stops, Travel class**
- **Flight duration (hours)**
- **Days before departure** (slide to adjust)

The app will **instantly predict the ticket price (₹)** using your trained model.

---

##  Model Details

| Property       | Value                          |
|----------------|--------------------------------|
| Algorithm      | Random Forest Regressor        |
| n_estimators   | 200 trees                      |
| max_depth      | 20                             |
| Features used  | 9 (encoded + numeric)          |
| Target         | Price in INR (₹)               |

---

## Web App Features

-  **Glassmorphic UI** with animated sky background & flying plane
-  Fully **responsive** layout
-  **Async prediction** via Fetch API (no page reload)
-  Interactive **range slider** for days before departure
-  Clear error messaging if the model isn't trained yet

---

##  Troubleshooting

**"Model not found" error in the browser?**
Run `python train_model.py` from the project root first, then restart the server.

**ModuleNotFoundError for Django?**
Make sure your virtual environment is activated before running `manage.py`.

**Different prediction for the same inputs?**
Random Forest uses a fixed `random_state=42`, so results are deterministic once trained.
