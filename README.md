# 🧠 Diabetes Progression Prediction API

This project builds and deploys a machine learning model to predict diabetes progression based on medical features. It uses the Diabetes dataset from **scikit-learn**, serves predictions through a **FastAPI** web application, and is containerized with **Docker** for easy deployment.

---

## 📦 Features

- Train a regression model on the Diabetes dataset
- Expose a RESTful prediction API using FastAPI
- Run the app with Uvicorn server
- Containerized with Docker

---

## 🔧 Technologies Used

- Python 3.8+
- scikit-learn
- FastAPI
- Uvicorn
- Docker

---

## 📁 Project Structure

```
diabetes-predictor/
│
├── app
    ├── template
        ├──index.html
    ├── main.py
├── models
    ├── diabetes_model.pkl
├── train_model.py
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Create the virtual environment

```
python -m venv diabetes-app
source diabetes-app/bin/activate
```

### 2. Clone the Repository

```bash
git clone https://github.com/truongdong212005/diabetes-predictor.git
cd diabetes-predictor
```

### 3. Train the Model

```bash
python train_model.py
```

This script:
- Loads the dataset
- Trains a regression model (using RandomForestRegressor)
- Saves it as `diabetes_model.pkl`

### 4. Run the API Server (Locally)

#### Install dependencies:

```bash
pip install -r requirements.txt
```

#### Start FastAPI with Uvicorn:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Now visit: `http://127.0.0.1:8000/docs` for interactive Swagger UI.

---

## 🧪 Example API Usage

### `POST /predict`

**Input JSON:**

```json
{
  "age": 0.038,
  "sex": 0.050,
  "bmi": 0.061,
  "bp": 0.021,
  "s1": -0.044,
  "s2": -0.034,
  "s3": -0.043,
  "s4": -0.002,
  "s5": 0.019,
  "s6": -0.017
}
```

**Response:**

```json
{
  "predicted_progression_score": 152.4,
  "interpretation": "Moderate risk of diabetes progression"
}
```

---

## 🐳 Run with Docker

### Build the Docker image

```bash
docker build -t diabetes-api .
```

### Run the container

```bash
docker run -d -p 8000:8000 diabetes-api
```

Visit the API at: `http://localhost:8000/docs`

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- Dataset from [scikit-learn Diabetes Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)
- API built using [FastAPI](https://fastapi.tiangolo.com)
