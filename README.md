# Feedforward Neural Network Demo with FastAPI + Streamlit

## 🚀 Project Purpose
This project demonstrates a simple **feedforward neural network** using **PyTorch**, with:

- A **backend** that trains, tests, and serves the model via **FastAPI**.
- A **frontend** UI built with **Streamlit** to enter features and get predictions.
- Clean separation of **frontend** and **backend**, each with its own virtual environment.
- Model checkpointing in `backend/checkpoints/model.pt`.

The project is suitable for learning ML deployment best practices and showcasing ML skills in a portfolio.

---

## 📂 Project Structure

```
my_project/
│
├── backend/
│   ├── checkpoints/        # Saved model
│   │   └── model.pt
│   ├── model.py            # Model architecture
│   ├── train.py            # Training script
│   ├── test.py             # Testing script
│   ├── api.py              # FastAPI backend
│   └── requirements.txt
│
└── frontend/
    ├── app.py              # Streamlit frontend
    └── requirements.txt
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd my_project
```

---

### 2. Setup backend virtual environment
```bash
cd backend
python -m venv venv                # create backend venv
# Activate venv
source venv/bin/activate           # Linux/macOS
# venv\Scripts\activate           # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

---

### 3. Setup frontend virtual environment
```bash
cd ../frontend
python -m venv venv                # create frontend venv
# Activate venv
source venv/bin/activate           # Linux/macOS
# venv\Scripts\activate           # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🏋️ Training the model

Activate **backend venv**:

```bash
cd backend
source venv/bin/activate            # Linux/macOS
# venv\Scripts\activate             # Windows

python train.py --epochs 20 --hidden-dim 64 --batch-size 32
```

This will save the model to `backend/checkpoints/model.pt`.

---

## 🤚 Testing the model

Activate **backend venv** and run:

```bash
python test.py --batch-size 32 --model-path checkpoints/model.pt
```

You should see **Test Accuracy** printed in the terminal.

---

## ⚡ Running the backend (FastAPI)

Activate **backend venv**:

```bash
uvicorn api:app --reload
```

The API will run at:
```
http://127.0.0.1:8000
```

Swagger docs available at:
```
http://127.0.0.1:8000/docs
```

---

## 🌐 Running the frontend (Streamlit)

Activate **frontend venv**:

```bash
cd frontend
streamlit run app.py
```

Streamlit app will open in the browser. Enter **4 features** and click **Predict** to get the model's output.

---

## 📌 Notes

- The project uses **dummy random data** for training and testing. Replace with a real dataset for meaningful predictions.
- The **backend and frontend** are completely separated, each with its own venv.
- Model checkpoints are stored in `backend/checkpoints/`. This is tracked in git.
- `.gitignore` excludes venvs, caches, logs, and OS/IDE junk.

---

## 📝 Requirements

### Backend
```
torch
fastapi
uvicorn
pydantic
```

### Frontend
```
streamlit
requests
```

---

## ✅ Best Practices Illustrated

1. Separate **frontend and backend** in different folders with isolated virtual environments.
2. Use a **checkpoints folder** for model storage.
3. Frontend communicates with backend via **HTTP API**.
4. Demonstrates **PyTorch training/testing loops**, **FastAPI serving**, and **Streamlit UI**.