from __future__ import annotations

import os
import json
import csv
import io
import base64
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
import joblib

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RECIPES_PATH = os.path.join(DATA_DIR, "recipes.json")
FEEDBACK_PATH = os.path.join(DATA_DIR, "feedback.csv")
MODEL_PATH = os.path.join(DATA_DIR, "model.joblib")

from fastapi.responses import RedirectResponse

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

app = FastAPI(title="Romantic Recipes ML", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # depois restringe
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Schemas
# ----------------------------
class QuizAnswers(BaseModel):
    mood: str              # ex: "cansado", "romantico", "animado"
    sweet_or_savory: str   # "doce" | "salgado"
    time_level: str        # "muito_rapido" | "rapido" | "sem_pressa"
    hunger: str            # "baixa" | "media" | "alta"
    temperature: str       # "frio" | "quente" | "tanto_faz"

class RecommendRequest(BaseModel):
    answers: QuizAnswers
    epsilon: float = 0.20  # exploração (20% aleatório)

class FeedbackRequest(BaseModel):
    recipe_id: str
    rating: int  # 1..5
    answers: QuizAnswers

# ----------------------------
# Helpers
# ----------------------------
def load_recipes() -> List[Dict[str, Any]]:
    if not os.path.exists(RECIPES_PATH):
        raise RuntimeError(f"recipes.json não encontrado em {RECIPES_PATH}")
    with open(RECIPES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_feedback_file() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(FEEDBACK_PATH):
        with open(FEEDBACK_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp",
                "recipe_id",
                "rating",
                "mood",
                "sweet_or_savory",
                "time_level",
                "hunger",
                "temperature",
            ])

def append_feedback(row: Dict[str, Any]) -> None:
    ensure_feedback_file()
    with open(FEEDBACK_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            row["timestamp"],
            row["recipe_id"],
            row["rating"],
            row["mood"],
            row["sweet_or_savory"],
            row["time_level"],
            row["hunger"],
            row["temperature"],
        ])

def load_feedback_df() -> pd.DataFrame:
    ensure_feedback_file()
    df = pd.read_csv(FEEDBACK_PATH)
    return df

def build_model(df: pd.DataFrame) -> Pipeline:
    # Features: respostas + recipe_id
    feature_cols = ["recipe_id", "mood", "sweet_or_savory", "time_level", "hunger", "temperature"]
    X = df[feature_cols].copy()
    y = df["rating"].astype(float)

    cat_cols = feature_cols  # tudo categórico aqui
    pre = ColumnTransformer(
        transformers=[
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols)
        ],
        remainder="drop"
    )

    model = Ridge(alpha=1.0, random_state=42)

    pipe = Pipeline(steps=[
        ("pre", pre),
        ("model", model)
    ])
    pipe.fit(X, y)
    return pipe

def save_model(pipe: Pipeline) -> None:
    joblib.dump(pipe, MODEL_PATH)

def load_model() -> Optional[Pipeline]:
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

def featurize_candidate(recipe_id: str, a: QuizAnswers) -> Dict[str, Any]:
    return dict(
        recipe_id=recipe_id,
        mood=a.mood,
        sweet_or_savory=a.sweet_or_savory,
        time_level=a.time_level,
        hunger=a.hunger,
        temperature=a.temperature,
    )

def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# ----------------------------
# Endpoints
# ----------------------------
@app.get("/api/recipes")
def api_recipes() -> Dict[str, Any]:
    recipes = load_recipes()
    return {"n": len(recipes), "recipes": recipes}

@app.post("/api/recommend")
def api_recommend(req: RecommendRequest) -> Dict[str, Any]:
    recipes = load_recipes()
    if len(recipes) == 0:
        raise HTTPException(status_code=400, detail="Nenhuma receita cadastrada.")

    pipe = load_model()
    epsilon = float(req.epsilon)

    # exploração: com prob epsilon escolhe aleatória
    if pipe is None:
        # sem modelo ainda: aleatório
        chosen = np.random.choice(recipes)
        return {"strategy": "cold_start_random", "recipe": chosen, "predicted_rating": None}

    if np.random.rand() < epsilon:
        chosen = np.random.choice(recipes)
        return {"strategy": "epsilon_random", "recipe": chosen, "predicted_rating": None}

    # exploração 0: prediz nota para cada receita e pega a maior
    candidates = [featurize_candidate(r["id"], req.answers) for r in recipes]
    Xcand = pd.DataFrame(candidates)
    preds = pipe.predict(Xcand)  # rating esperado

    best_idx = int(np.argmax(preds))
    best_recipe = recipes[best_idx]
    best_pred = float(preds[best_idx])

    return {"strategy": "model_best", "recipe": best_recipe, "predicted_rating": best_pred}

@app.post("/api/feedback")
def api_feedback(req: FeedbackRequest) -> Dict[str, Any]:
    if req.rating < 1 or req.rating > 5:
        raise HTTPException(status_code=400, detail="rating deve ser entre 1 e 5.")

    recipes = load_recipes()
    if req.recipe_id not in {r["id"] for r in recipes}:
        raise HTTPException(status_code=400, detail="recipe_id inválido.")

    row = dict(
        timestamp=datetime.utcnow().isoformat(),
        recipe_id=req.recipe_id,
        rating=req.rating,
        mood=req.answers.mood,
        sweet_or_savory=req.answers.sweet_or_savory,
        time_level=req.answers.time_level,
        hunger=req.answers.hunger,
        temperature=req.answers.temperature,
    )
    append_feedback(row)

    # re-treina e salva (rápido, dataset pequeno)
    df = load_feedback_df()
    if len(df) >= 5:  # a partir de 5 feedbacks já faz sentido treinar
        pipe = build_model(df)
        save_model(pipe)

    return {"ok": True, "n_feedback": int(len(df))}

@app.get("/api/analytics")
def api_analytics() -> Dict[str, Any]:
    df = load_feedback_df()
    if df.empty:
        return {"n_feedback": 0, "plot_base64": None}

    # gráfico: média de rating por receita
    g = df.groupby("recipe_id")["rating"].mean().sort_values(ascending=False)

    fig = plt.figure()
    plt.bar(g.index.astype(str), g.values)
    plt.title("Média de rating por receita")
    plt.xlabel("recipe_id")
    plt.ylabel("rating médio")
    plt.xticks(rotation=45, ha="right")

    return {"n_feedback": int(len(df)), "plot_base64": fig_to_base64(fig)}