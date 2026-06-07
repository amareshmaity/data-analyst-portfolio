#!/usr/bin/env python
# coding: utf-8

# ## Models Training

# In[1]:


# import library
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import FunctionTransformer, PowerTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# In[2]:


## Load feature engineered dataset into a dataframe
df = pd.read_csv("../data/processed/feature_engineered_flipcart_laptop_data.csv")

## Top five observations
df.head()


# In[3]:


df.info()


# In[4]:


corr = df.select_dtypes(include=np.number).corr()

plt.figure(figsize=(10,6))
sns.heatmap(corr, annot=True, cmap="coolwarm")

plt.title("Feature Correlation Matrix")
plt.show()


# ### Create Features and Target Variable

# In[5]:


# Features
X = df.drop(columns=['Discount price'])
# Target
y = df["Discount price"]

X.head()


# In[6]:


X.shape


# In[7]:


y.head()


# In[8]:


## Apply log transformation on y as it too much right skewed
def log_transformer(series):
    transformed_series = np.log1p(series)
    return transformed_series

def reversed_log_transformer(log_series):
    reversed_log_series = np.expm1(log_series)
    return reversed_log_series

log_y = log_transformer(y)
log_y


# ### Splitting into training and testing set

# In[9]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train_log, y_test_log = train_test_split(X, log_y, test_size=0.2, random_state=42)

print("Shape of X_train", X_train.shape)
print("Shape of X_test", X_test.shape)
print("Shape of y_train", y_train_log.shape)
print("Shape of y_test", y_test_log.shape)


# In[10]:


X_train.head()


# In[11]:


## Initialize the PowerTransformer
yeo_trf = PowerTransformer(method='yeo-johnson')

## Instantiate of log transformer
log_trf = FunctionTransformer(np.log1p)

## Pipeline for sequential execution
numeric_pipeline = Pipeline([
    ("log_transform", log_trf),
    ("scaler", StandardScaler())
])

yeo_pipeline = Pipeline([
    ("yeo_transform", yeo_trf),
    ("scaler", StandardScaler())
])


# In[12]:


## Create Column Transformer
transformer = ColumnTransformer(
    transformers=[
        ("numeric_pipeline", numeric_pipeline, ["Rating", "Reviews", "ram_gb", "storage_gb", "discount_pct(%)"]),
        ("yeo_pipeline", yeo_pipeline, ["Stars"]),

        # 1. TARGET ENCODING FOR BRAND (Keeps it as 1 column)
        ("target_enc", TargetEncoder(smooth="auto", cv=5, target_type='continuous'), ["brand"]),

        ("OneHotEncoding", OneHotEncoder(
            drop='first',
            min_frequency=6,          # Categories appearing < 6 times become 'Other'
            handle_unknown='infrequent_if_exist', 
            sparse_output=False
            ), ["storage_type", "processor_brand"]
        ),

        ("OrdinalEncoding",OrdinalEncoder(categories=[["Entry", "Mid", "High", "Ultra", "Other"]]),["processor_tier"])
    ],

    remainder='passthrough'
)


# In[13]:


## Fit and Transform the Training Set
X_train_trf = transformer.fit_transform(X_train, y_train_log)
X_train_trf


# In[14]:


X_train_trf.shape


# In[15]:


## Transform only Testing set
X_test_trf = transformer.transform(X_test)
X_test_trf


# ### Regularized Model Training
# 
# This section replaces the earlier high-variance model blocks with a smaller set of regularized candidates.
# The goal is to improve generalization by:
# 
# - using a single target strategy for every model
# - keeping tree depth shallow
# - increasing minimum leaf sizes
# - using cross-validation and explicitly tracking the train/test gap
# 

# In[ ]:


from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, cross_val_score


y_train = y.loc[X_train.index].copy()
y_test = y.loc[X_test.index].copy()

cv = KFold(n_splits=5, shuffle=True, random_state=42)


def build_pipeline(regressor):
    return Pipeline([
        ("preprocessor", transformer),
        (
            "model",
            TransformedTargetRegressor(
                regressor=regressor,
                func=np.log1p,
                inverse_func=np.expm1,
            ),
        ),
    ])


def evaluate_pipeline(name, pipeline):
    pipeline.fit(X_train, y_train)

    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="r2",
        n_jobs=1,
    )

    metrics = {
        "Model": name,
        "Train R2": r2_score(y_train, y_train_pred),
        "Test R2": r2_score(y_test, y_test_pred),
        "CV Mean R2": cv_scores.mean(),
        "CV Std R2": cv_scores.std(),
        "Train-Test Gap": r2_score(y_train, y_train_pred) - r2_score(y_test, y_test_pred),
        "Train MAE": mean_absolute_error(y_train, y_train_pred),
        "Test MAE": mean_absolute_error(y_test, y_test_pred),
        "Train RMSE": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "Test RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
    }

    return metrics, y_train_pred, y_test_pred


# #### Tune Stable Candidate Models
# 
# Only regularized candidates are compared below. Every search uses `n_jobs=1` so the notebook runs cleanly in restricted environments.

# In[ ]:


elastic_net_pipeline = build_pipeline(
    ElasticNet(max_iter=20000, random_state=42)
)

elastic_net_grid = {
    "model__regressor__alpha": [0.01, 0.05, 0.1, 0.5, 1.0],
    "model__regressor__l1_ratio": [0.2, 0.4, 0.6, 0.8],
}

elastic_net_search = GridSearchCV(
    estimator=elastic_net_pipeline,
    param_grid=elastic_net_grid,
    cv=cv,
    scoring="r2",
    n_jobs=1,
    return_train_score=True,
)

elastic_net_search.fit(X_train, y_train)

print("ElasticNet best params:", elastic_net_search.best_params_)
print("ElasticNet CV R2:", round(elastic_net_search.best_score_, 4))


# In[ ]:


rf_pipeline = build_pipeline(
    RandomForestRegressor(
        random_state=42,
        bootstrap=True,
    )
)

rf_param_grid = {
    "model__regressor__n_estimators": [200, 300],
    "model__regressor__max_depth": [3, 4, 5],
    "model__regressor__min_samples_split": [10, 20, 30],
    "model__regressor__min_samples_leaf": [8, 12, 16],
    "model__regressor__max_features": [0.4, 0.6, "sqrt"],
}

rf_search = RandomizedSearchCV(
    estimator=rf_pipeline,
    param_distributions=rf_param_grid,
    n_iter=10,
    cv=cv,
    scoring="r2",
    n_jobs=1,
    random_state=42,
    return_train_score=True,
)

rf_search.fit(X_train, y_train)

print("Random Forest best params:", rf_search.best_params_)
print("Random Forest CV R2:", round(rf_search.best_score_, 4))


# In[ ]:


gb_pipeline = build_pipeline(
    GradientBoostingRegressor(random_state=42)
)

gb_param_grid = {
    "model__regressor__n_estimators": [100, 150, 200],
    "model__regressor__learning_rate": [0.03, 0.05],
    "model__regressor__max_depth": [2, 3],
    "model__regressor__min_samples_split": [10, 20],
    "model__regressor__min_samples_leaf": [6, 10],
    "model__regressor__subsample": [0.7, 0.8],
    "model__regressor__max_features": [0.5, 0.7],
}

gb_search = RandomizedSearchCV(
    estimator=gb_pipeline,
    param_distributions=gb_param_grid,
    n_iter=10,
    cv=cv,
    scoring="r2",
    n_jobs=1,
    random_state=42,
    return_train_score=True,
)

gb_search.fit(X_train, y_train)

print("Gradient Boosting best params:", gb_search.best_params_)
print("Gradient Boosting CV R2:", round(gb_search.best_score_, 4))


# #### Compare Models and Select the Most Stable One

# In[ ]:


candidate_models = {
    "ElasticNet": elastic_net_search.best_estimator_,
    "Random Forest": rf_search.best_estimator_,
    "Gradient Boosting": gb_search.best_estimator_,
}

evaluation_rows = []
predictions = {}

for model_name, model_pipeline in candidate_models.items():
    model_metrics, train_pred, test_pred = evaluate_pipeline(model_name, model_pipeline)
    evaluation_rows.append(model_metrics)
    predictions[model_name] = {
        "train": train_pred,
        "test": test_pred,
    }

results_df = pd.DataFrame(evaluation_rows).sort_values(
    by=["Test R2", "Train-Test Gap"],
    ascending=[False, True],
)

results_df.round(4)


# In[ ]:


stable_results = results_df[
    (results_df["Train-Test Gap"] <= 0.08)
    & (results_df["CV Mean R2"] >= results_df["Test R2"] - 0.05)
].sort_values(
    by=["Test R2", "Train-Test Gap"],
    ascending=[False, True],
)

if stable_results.empty:
    stable_results = results_df.sort_values(
        by=["Train-Test Gap", "CV Std R2", "Test R2"],
        ascending=[True, True, False],
    ).head(1)

selected_model_name = stable_results.iloc[0]["Model"]
final_pipeline = candidate_models[selected_model_name]

print("Selected model:", selected_model_name)
print(stable_results.round(4))


# ## Final Model
# 
# The final model is selected only after balancing predictive power with stability on unseen data.

# In[ ]:


final_pipeline.fit(X_train, y_train)

y_train_pred = final_pipeline.predict(X_train)
y_test_pred = final_pipeline.predict(X_test)

final_metrics_df = pd.DataFrame(
    {
        "Metric": ["R2 Score", "Mean Absolute Error", "Root Mean Squared Error"],
        "Train Set": [
            r2_score(y_train, y_train_pred),
            mean_absolute_error(y_train, y_train_pred),
            np.sqrt(mean_squared_error(y_train, y_train_pred)),
        ],
        "Test Set": [
            r2_score(y_test, y_test_pred),
            mean_absolute_error(y_test, y_test_pred),
            np.sqrt(mean_squared_error(y_test, y_test_pred)),
        ],
    }
)

print("Final model:", selected_model_name)
final_metrics_df.round(4)


# #### Feature Importance / Coefficient Strength

# In[ ]:


numeric_features = ["Rating", "Reviews", "ram_gb", "storage_gb", "discount_pct(%)"]
yeo_features = ["Stars"]
target_features = ["brand"]
ordinal_features = ["processor_tier"]
passthrough_features = ["processor_generation", "is_gaming", "premium_brand"]

fitted_preprocessor = final_pipeline.named_steps["preprocessor"]
ohe_features = fitted_preprocessor.named_transformers_["OneHotEncoding"].get_feature_names_out(
    ["storage_type", "processor_brand"]
)

feature_names = (
    numeric_features
    + yeo_features
    + target_features
    + list(ohe_features)
    + ordinal_features
    + passthrough_features
)

final_regressor = final_pipeline.named_steps["model"].regressor_

if hasattr(final_regressor, "feature_importances_"):
    importance_values = final_regressor.feature_importances_
    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importance_values}
    ).sort_values(by="Importance", ascending=False)
else:
    importance_values = np.abs(final_regressor.coef_)
    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importance_values}
    ).sort_values(by="Importance", ascending=False)

importance_df.head(15)


# In[ ]:


plt.figure(figsize=(12, 8))
plt.scatter(y_test, y_test_pred, alpha=0.7)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red",
    linestyle="--",
)
plt.xlabel("Actual value")
plt.ylabel("Predicted value")
plt.title(f"Actual vs Predicted: {selected_model_name}")
plt.show()


# In[ ]:


residuals = y_test - y_test_pred

plt.figure(figsize=(8, 5))
sns.histplot(residuals, kde=True)
plt.title("Residual Distribution")
plt.show()


# In[ ]:


plt.figure(figsize=(8, 5))
plt.scatter(y_test_pred, residuals, alpha=0.6)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Residual Analysis")
plt.show()

