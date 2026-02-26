const API_BASE = "http://127.0.0.1:8002";

const mood = document.getElementById("mood");
const sweetOrSavory = document.getElementById("sweetOrSavory");
const timeLevel = document.getElementById("timeLevel");
const hunger = document.getElementById("hunger");
const temperature = document.getElementById("temperature");

const btnRecommend = document.getElementById("btnRecommend");
const btnSendFeedback = document.getElementById("btnSendFeedback");
const btnAnalytics = document.getElementById("btnAnalytics");

const epsilon = document.getElementById("epsilon");
const epsilonVal = document.getElementById("epsilonVal");

const errorBox = document.getElementById("error");

const recipeCard = document.getElementById("recipeCard");
const recipeTitle = document.getElementById("recipeTitle");
const recipeMeta = document.getElementById("recipeMeta");
const recipeIngredients = document.getElementById("recipeIngredients");
const recipeSteps = document.getElementById("recipeSteps");

const rating = document.getElementById("rating");
const strategyHint = document.getElementById("strategyHint");

const analyticsBox = document.getElementById("analyticsBox");

let lastRecipe = null;
let lastAnswers = null;

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.classList.remove("hidden");
}

function clearError() {
  errorBox.textContent = "";
  errorBox.classList.add("hidden");
}

function getAnswers() {
  return {
    mood: mood.value,
    sweet_or_savory: sweetOrSavory.value,
    time_level: timeLevel.value,
    hunger: hunger.value,
    temperature: temperature.value,
  };
}

function renderRecipe(recipe, metaText) {
  recipeCard.classList.remove("hidden");
  recipeTitle.textContent = recipe.title;
  recipeMeta.textContent = metaText || `${recipe.time_min} min`;

  recipeIngredients.innerHTML = "";
  recipe.ingredients.forEach((ing) => {
    const li = document.createElement("li");
    li.textContent = ing;
    recipeIngredients.appendChild(li);
  });

  recipeSteps.innerHTML = "";
  recipe.steps.forEach((st) => {
    const li = document.createElement("li");
    li.textContent = st;
    recipeSteps.appendChild(li);
  });
}

epsilon.addEventListener("input", () => {
  epsilonVal.textContent = Number(epsilon.value).toFixed(2);
});

btnRecommend.addEventListener("click", async () => {
  clearError();
  analyticsBox.innerHTML = "";

  const answers = getAnswers();
  lastAnswers = answers;
  lastRecipe = null;

  const payload = { answers, epsilon: Number(epsilon.value) };

  try {
    const res = await fetch(`${API_BASE}/api/recommend`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await res.json();
    if (!res.ok) {
      showError(data.detail || "Erreur lors de la recommandation.");
      return;
    }

    lastRecipe = data.recipe;

    let hint = "";
    if (data.strategy === "cold_start_random") {
      hint = "Je n’ai pas encore assez de données — je choisis au hasard 💞";
    } else if (data.strategy === "epsilon_random") {
      hint = "Mode surprise activé (exploration) ✨";
    } else if (data.strategy === "model_best") {
      const pr = (typeof data.predicted_rating === "number")
        ? data.predicted_rating.toFixed(2)
        : "—";
      hint = `Choix personnalisé (note prévue ≈ ${pr}) 💗`;
    } else {
      hint = "—";
    }

    strategyHint.textContent = hint;
    renderRecipe(
      data.recipe,
      `${data.recipe.time_min} min • ${data.recipe.tags.join(", ")}`
    );
  } catch (e) {
    showError("Impossible de contacter le backend. Vérifie qu’il tourne sur http://127.0.0.1:8002");
  }
});

btnSendFeedback.addEventListener("click", async () => {
  clearError();

  if (!lastRecipe || !lastAnswers) {
    showError("Demande d’abord une recette ✨");
    return;
  }

  const payload = {
    recipe_id: lastRecipe.id,
    rating: Number(rating.value),
    answers: lastAnswers,
  };

  try {
    const res = await fetch(`${API_BASE}/api/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await res.json();
    if (!res.ok) {
      showError(data.detail || "Erreur lors de l’envoi de l’avis.");
      return;
    }

    strategyHint.textContent =
      `Avis enregistré ! Total d’évaluations : ${data.n_feedback} 💖 (le modèle se met à jour automatiquement)`;
  } catch (e) {
    showError("Impossible de contacter le backend (/api/feedback).");
  }
});

btnAnalytics.addEventListener("click", async () => {
  clearError();
  analyticsBox.innerHTML = "Chargement du graphique...";

  try {
    const res = await fetch(`${API_BASE}/api/analytics`);
    const data = await res.json();

    if (!res.ok) {
      showError(data.detail || "Erreur lors du chargement des statistiques.");
      analyticsBox.innerHTML = "";
      return;
    }

    if (!data.plot_base64) {
      analyticsBox.innerHTML = "Pas encore de données 😔 Donne quelques notes d’abord !";
      return;
    }

    const img = document.createElement("img");
    img.src = `data:image/png;base64,${data.plot_base64}`;
    analyticsBox.innerHTML = "";
    analyticsBox.appendChild(img);
  } catch (e) {
    showError("Impossible de contacter le backend (/api/analytics).");
    analyticsBox.innerHTML = "";
  }
});