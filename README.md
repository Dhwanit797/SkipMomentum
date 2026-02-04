🎧 SkipMomentum — Session-Based Skip Probability Modeling

## 📌 Overview

**SkipMomentum** is a session-based machine learning project that predicts the probability of a Spotify user skipping a track during a listening session.
The project focuses on modeling **dynamic user behavior** by capturing session momentum and fatigue patterns — showing how previous listening actions influence future skips.

It demonstrates practical skills in **data preprocessing, feature engineering, machine learning, and visualization**.

---

## 🚀 Key Features

* 📊 Session-based skip prediction
* 📈 Dynamic modeling of user behavior within listening sessions
* 🧠 Momentum & fatigue feature engineering
* 📉 Visual analysis of skip probability trends
* 🤖 Logistic Regression–based predictive modeling

### Engineered Behavioral Features

* **`prev_track_skipped`**
  Indicates whether the previous track was skipped.

* **`prev_ms_played`**
  Duration (milliseconds) the previous track was played.

* **`past_consecutive_skips`**
  Number of consecutive skipped tracks before the current one.

* **`past_skip_ratio`**
  Proportion of skipped tracks so far in the session.

---

## 🛠️ Tech Stack

* **Language:** Python 3
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-learn

  * Logistic Regression
  * StandardScaler
* **Visualization:** Matplotlib

---

## ▶️ How to Run

### 1️⃣ Setup

Place your Spotify listening dataset:

```
spotify_history.csv
```

in the same directory as the script.

### 2️⃣ Execute the Script

```bash
python skip_momentum.py
```

### 3️⃣ Expected Output

* ROC AUC score of the predictive model
* Feature importance bar plot
* Skip probability curve for the longest session
* Sample session predictions with probabilities

---

## 📂 Project Structure

```
SkipMomentum/
│
├── skip_momentum.py        # Main modeling & visualization script
├── spotify_history.csv     # Spotify listening history dataset
└── README.md               # Project documentation
```

---

## 📊 Visualizations Included

* Feature influence on skip probability (Bar Plot)
* Session progression vs skip probability (Line Plot)

---

## 👨‍💻 Author

**Dhwanit**
Government Polytechnic, Ahmedabad

---

## 🔮 Future Enhancements

* 👤 Multi-user personalization
* 🎼 Integration of track metadata (genre, popularity, artist features)
* 🌐 Web dashboard for real-time skip prediction
* 🤖 Advanced models comparison (Random Forest, XGBoost, Gradient Boosting)

---

## 📄 License

This project is intended for educational and portfolio purposes.

---
