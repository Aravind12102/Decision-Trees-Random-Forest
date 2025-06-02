# Decision-Trees-Random-Forest

This project implements tree-based machine learning models to predict the presence of **heart disease** using the **Heart Disease UCI dataset**. The focus is on **Decision Trees** and **Random Forests** for classification, model interpretability, and evaluation using cross-validation.

---

## Dataset Overview

- **Source**: UCI Heart Disease dataset (`heart.csv`)
- **Samples**: 1,025
- **Features**: 13 (e.g., age, sex, chest pain type, cholesterol, etc.)
- **Target**: `target` — `1` (disease), `0` (no disease)

---

## Tools & Libraries

- `pandas`, `matplotlib`, `seaborn`
- `scikit-learn`
- `Graphviz` (optional for advanced tree visualization)

---

## Objectives

1. Train a **Decision Tree Classifier** and visualize it.
2. Prune/control tree depth to prevent overfitting.
3. Train a **Random Forest Classifier** and compare performance.
4. Interpret **feature importances** from Random Forest.
5. Use **cross-validation** to evaluate model performance.

---

## Project Workflow

### Step 1: Load and Preprocess Data
- Drop the target column to separate features (`X`) and labels (`y`)
- Train/test split using `train_test_split`

### Step 2: Decision Tree Classifier
- Train using `DecisionTreeClassifier`
- Visualize first 3 levels with `plot_tree`
- Prune using `max_depth=4` to reduce overfitting

### Step 3: Random Forest Classifier
- Train using `RandomForestClassifier`
- Compare performance vs. Decision Tree

### Step 4: Feature Importances
- Plot top contributing features using `.feature_importances_`

### Step 5: Cross-Validation
- 5-fold cross-validation for both classifiers using `cross_val_score`

---

## Results

| Model              | Test Accuracy | CV Accuracy | Notes                        |
|-------------------|---------------|-------------|------------------------------|
| Decision Tree      | ~80–85%       | ~78–82%     | May overfit without pruning |
| Pruned Tree (depth=4) | Lower variance | ~80%     | Better generalization        |
| Random Forest       | ~85–90%       | ~85–88%     | Most stable and accurate     |

---

## 📁 File Structure

