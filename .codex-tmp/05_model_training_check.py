#!/usr/bin/env python
# coding: utf-8

# ## Models Training

# In[77]:


# import library
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import FunctionTransformer, PowerTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# In[78]:


## Load feature engineered dataset into a dataframe
df = pd.read_csv("../data/processed/feature_engineered_flipcart_laptop_data.csv")

## Top five observations
df.head()


# In[79]:


df.info()


# In[80]:


corr = df.select_dtypes(include=np.number).corr()

plt.figure(figsize=(10,6))
sns.heatmap(corr, annot=True, cmap="coolwarm")

plt.title("Feature Correlation Matrix")
plt.show()


# ### Create Features and Target Variable

# In[81]:


# Features
X = df.drop(columns=['Discount price'])
# Target
y = df["Discount price"]

X.head()


# In[82]:


X.shape


# In[83]:


y.head()


# In[84]:


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

# In[85]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train_log, y_test_log = train_test_split(X, log_y, test_size=0.2, random_state=42)

print("Shape of X_train", X_train.shape)
print("Shape of X_test", X_test.shape)
print("Shape of y_train", y_train_log.shape)
print("Shape of y_test", y_test_log.shape)


# In[86]:


X_train.head()


# In[87]:


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


# In[88]:


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


# In[89]:


## Fit and Transform the Training Set
X_train_trf = transformer.fit_transform(X_train, y_train_log)
X_train_trf


# In[90]:


X_train_trf.shape


# In[91]:


## Transform only Testing set
X_test_trf = transformer.transform(X_test)
X_test_trf


# ### Train different models
# 
# - #### Linear Regression 

# In[92]:


from sklearn.linear_model import LinearRegression

## Create instantiate
lr = LinearRegression()

## Fit the dataset into model
lr.fit(X_train_trf, y_train_log)


# In[93]:


## Prediction On Training Set
lr_train_pred = reversed_log_transformer(lr.predict(X_train_trf))

## Prediction on Testing Set
lr_pred = reversed_log_transformer(lr.predict(X_test_trf))


# In[94]:


## Actual value of y
y_train = reversed_log_transformer(y_train_log)
y_test = reversed_log_transformer(y_test_log)


# In[95]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

## Mean Square Error
lr_mse = mean_squared_error(y_true=y_train, y_pred=lr_train_pred)
## Mean Absolute Error
lr_mae = mean_absolute_error(y_true=y_train, y_pred=lr_train_pred)
## R2 score
lr_r2_score = r2_score(y_true=y_train, y_pred=lr_train_pred)

print("MSE: ", lr_mse)
print("MAE: ", lr_mae)
print("R2-Score: ", lr_r2_score)


# In[96]:


## Mean Square Error
lr_test_mse = mean_squared_error(y_true=y_test, y_pred=lr_pred)
## Mean Absolute Error
lr_test_mae = mean_absolute_error(y_true=y_test, y_pred=lr_pred)
## R2 score
lr_test_r2_score = r2_score(y_true=y_test, y_pred=lr_pred)

print("MSE: ", lr_test_mse)
print("MAE: ", lr_test_mae)
print("R2-Score: ", lr_test_r2_score)


# - #### Lasso Regression

# In[97]:


from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

lasso_pipeline = Pipeline([
    ('preprocessor', transformer),
    ('model', Lasso(max_iter=10000, random_state=42)) # Increased iterations for convergence
])

# Define the alpha grid (smaller alpha = less regularization, larger = more)
param_grid = {
    'model__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
}

grid_search = GridSearchCV(lasso_pipeline, param_grid, cv=5, scoring='r2', return_train_score=True)
grid_search.fit(X_train, y_train_log)

# Print best parameters and scores
print(f"Best Alpha: {grid_search.best_params_}")
print(f"Train R2 Score: {grid_search.cv_results_['mean_train_score'][grid_search.best_index_]:.4f}")
print(f"Validation R2 Score: {grid_search.best_score_:.4f}")


# In[98]:


# Check which features Lasso selected (kept non-zero)
best_model = grid_search.best_estimator_.named_steps['model']
print(f"Total features: {len(best_model.coef_)}")
print(f"Features kept by Lasso: {sum(best_model.coef_ != 0)}")


# In[99]:


best_model.coef_


# In[100]:


# import pandas as pd
# import numpy as np

# ## Extract the fitted preprocessor
# fitted_preprocessor = grid_search.best_estimator_.named_steps['preprocessor']

# ## Reconstruct names safely by block sequence
# numeric_cols = ["Rating", "Reviews", "ram_gb", "storage_gb", "discount_pct(%)"]
# yeo_cols = ["Stars"]
# brand_cols = ["brand"]

# # Extract dynamic column names from the One-Hot Encoder
# ohe_step = fitted_preprocessor.named_transformers_['OneHotEncoding']
# ohe_cols = list(ohe_step.get_feature_names_out(["storage_type", "processor_brand"]))

# ordinal_cols = ["processor_tier"]

# # Extract remainder columns manually from ColumnTransformer configuration
# # These are the columns not mention in any other transformer block
# all_transformed = set(numeric_cols + yeo_cols + brand_cols + ["storage_type", "processor_brand"] + ordinal_cols)
# remainder_cols = [col for col in X_train.columns if col not in all_transformed]

# # Combine everything in the exact order of ColumnTransformer definition
# feature_names = numeric_cols + yeo_cols + brand_cols + ohe_cols + ordinal_cols + remainder_cols

# ## Pull coefficients from the model
# best_lasso_model = grid_search.best_estimator_.named_steps['model']
# coefficients = best_lasso_model.coef_

# ## Check array length to ensure it perfectly matches up
# if len(feature_names) == len(coefficients):
#     feature_importance_df = pd.DataFrame({
#         'Feature': feature_names,
#         'Coefficient': coefficients,
#         'Absolute_Coefficient': np.abs(coefficients)
#     }).sort_values(by='Absolute_Coefficient', ascending=False)

#     print("\n--- Features Kept vs. Dropped by Lasso Regularization ---\n")
#     print(feature_importance_df)
# else:
#     print(f"Error: Array lengths mismatch. Expected {len(coefficients)} names, but manually constructed {len(feature_names)}.")


# In[ ]:


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

## Retrieve the optimal model pipeline from grid search
best_lasso_pipeline = grid_search.best_estimator_

## Generate predictions for both datasets
y_train_pred = reversed_log_transformer(best_lasso_pipeline.predict(X_train))
y_test_pred = reversed_log_transformer(best_lasso_pipeline.predict(X_test))


# In[102]:


## Calculate Performance Metrics
metrics_summary = {
    "Metric": ["R² Score", "Mean Absolute Error (MAE)", "Root Mean Squared Error (RMSE)"],
    "Train Set": [
        r2_score(y_train, y_train_pred),
        mean_absolute_error(y_train, y_train_pred),
        np.sqrt(mean_squared_error(y_train, y_train_pred))
    ],
    "Test Set": [
        r2_score(y_test, y_test_pred),
        mean_absolute_error(y_test, y_test_pred),
        np.sqrt(mean_squared_error(y_test, y_test_pred))
    ]
}

## Format and display the results cleanly
import pandas as pd
metrics_df = pd.DataFrame(metrics_summary)
print("\n--- Final Lasso Model Evaluation Summary ---\n")
print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# - #### Decision Tree

# In[103]:


from sklearn.tree import DecisionTreeRegressor

## Create instantiate
dt_reg = DecisionTreeRegressor(random_state=42)
dt_reg


# In[104]:


## Train the model
dt_reg.fit(X_train_trf, y_train_log)


# In[105]:


## Prediction on the training dataset
dt_train_pred = reversed_log_transformer(dt_reg.predict(X_train_trf))

## Prediction on the testing dataset
dt_test_pred = reversed_log_transformer(dt_reg.predict(X_test_trf))


# In[106]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

## Mean Square Error
dt_mse = mean_squared_error(y_true=y_train, y_pred=dt_train_pred)
dt_mae = mean_absolute_error(y_true=y_train, y_pred=dt_train_pred)
dt_r2_score = r2_score(y_true=y_train, y_pred=dt_train_pred)
print("MSE: ", dt_mse)
print("MAE: ", dt_mae)
print("R2-Score: ", dt_r2_score)


# In[107]:


dt_test_mse = mean_squared_error(y_true=y_test, y_pred=dt_test_pred)
dt_test_mae = mean_absolute_error(y_true=y_test, y_pred=dt_test_pred)
dt_test_r2_score = r2_score(y_true=y_test, y_pred=dt_test_pred)
print("MSE: ", dt_test_mse)
print("MAE: ", dt_test_mae)
print("R2-Score: ", dt_test_r2_score)


# - #### Decision Tree (With Hyper-Parameter Tuning)

# In[108]:


from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    "max_depth": [3, 5, 7, 10, 15],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", "log2", None]
}
# param_grid = {
#     "max_depth": [3, 5, 7, 10, 15],
#     "min_samples_split": [2, 5, 10, 15],
#     "min_samples_leaf": [5, 10, 15],
#     "max_features": ["sqrt", "log2", 0.3, 0.5]
# }


# In[109]:


random_search = RandomizedSearchCV(
    estimator=dt_reg,
    param_distributions=param_grid,
    n_iter=20,
    scoring="r2",
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)


# In[110]:


random_search.fit(X_train_trf, y_train_log)


# In[111]:


best_param = random_search.best_params_
best_param


# In[112]:


## Decision Tree with best parameters
dt_model = DecisionTreeRegressor(**best_param, random_state=42)
dt_model


# In[113]:


## Train the model
dt_model.fit(X_train_trf, y_train_log)


# In[114]:


## Prediction on the training dataset
dt_model_train_pred = reversed_log_transformer(dt_model.predict(X_train_trf))


# In[115]:


## Prediction on the training dataset
dt_model_test_pred = reversed_log_transformer(dt_model.predict(X_test_trf))


# In[116]:


## Mean Square Error
dt_model_mse = mean_squared_error(y_true=y_train, y_pred=dt_model_train_pred)
dt_model_mae = mean_absolute_error(y_true=y_train, y_pred=dt_model_train_pred)
dt_model_r2_score = r2_score(y_true=y_train, y_pred=dt_model_train_pred)
print("MSE: ", dt_model_mse)
print("MAE: ", dt_model_mae)
print("R2-Score: ", dt_model_r2_score)


# In[117]:


## Mean Square Error
dt_model_test_mse = mean_squared_error(y_true=y_test, y_pred=dt_model_test_pred)
dt_model_test_mae = mean_absolute_error(y_true=y_test, y_pred=dt_model_test_pred)
dt_model_test_r2_score = r2_score(y_true=y_test, y_pred=dt_model_test_pred)
print("MSE: ", dt_model_test_mse)
print("MAE: ", dt_model_test_mae)
print("R2-Score: ", dt_model_test_r2_score)


# - #### Random Forest

# In[118]:


from sklearn.ensemble import RandomForestRegressor

## create instantiate
rf = RandomForestRegressor(random_state=42)

rf.fit(X_train_trf, y_train_log)


# In[119]:


y_pred = reversed_log_transformer(rf.predict(X_test_trf))
y_train_pred = reversed_log_transformer(rf.predict(X_train_trf))


# In[120]:


rf_train_r2_score = r2_score(y_true=y_test, y_pred=y_pred)
rf_train_r2_score


# In[121]:


rf_test_r2_score = r2_score(y_true=y_train, y_pred=y_train_pred)
rf_test_r2_score


# In[122]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Build the Random Forest pipeline
rf_pipeline = Pipeline([
    ('preprocessor', transformer),
    ('model', RandomForestRegressor(random_state=42))
])

# Define a highly regularized grid
rf_param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [3, 4, 5],               # Kept shallow to prevent memorization
    'model__min_samples_leaf': [5, 10, 15],       # Requires more samples per leaf node
    'model__max_features': [0.3, 0.5, 'sqrt']     # Only looks at a fraction of features per split
}

rf_grid = GridSearchCV(rf_pipeline, rf_param_grid, cv=5, scoring='r2', n_jobs=-1, return_train_score=True)
rf_grid.fit(X_train, y_train_log)

print(f"Best RF Params: {rf_grid.best_params_}")
print(f"RF Train R2: {rf_grid.cv_results_['mean_train_score'][rf_grid.best_index_]:.4f}")
print(f"RF Val R2: {rf_grid.best_score_:.4f}")


# In[123]:


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd

## Retrieve the optimal, fully-trained Random Forest pipeline
best_rf_pipeline = rf_grid.best_estimator_

## Generate predictions using the optimized model
y_train_pred_rf = reversed_log_transformer(best_rf_pipeline.predict(X_train))
y_test_pred_rf = reversed_log_transformer(best_rf_pipeline.predict(X_test))

## Calculate Performance Metrics
rf_metrics_summary = {
    "Metric": ["R² Score", "Mean Absolute Error (MAE)", "Root Mean Squared Error (RMSE)"],
    "Train Set": [
        r2_score(y_train, y_train_pred_rf),
        mean_absolute_error(y_train, y_train_pred_rf),
        np.sqrt(mean_squared_error(y_train, y_train_pred_rf))
    ],
    "Test Set": [
        r2_score(y_test, y_test_pred_rf),
        mean_absolute_error(y_test, y_test_pred_rf),
        np.sqrt(mean_squared_error(y_test, y_test_pred_rf))
    ]
}

## Format and display the results cleanly
rf_metrics_df = pd.DataFrame(rf_metrics_summary)
print("\n--- Final Random Forest Model Evaluation Summary ---")
print(rf_metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# In[95]:


from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np

## Wrap Random Forest configuration so the target variable gets log-transformed
logged_rf_model = TransformedTargetRegressor(
    regressor=RandomForestRegressor(
        n_estimators=150,
        max_depth=5,
        min_samples_leaf=4,
        max_features=0.4, 
        random_state=42
    ),
    func=np.log1p,                # Applies log(x + 1) to 'Discount price' before fitting
    inverse_func=np.expm1         # Automatically converts predictions back to actual currency
)

## Build final pipeline
final_pipeline = Pipeline([
    ('preprocessor', transformer),
    ('model', logged_rf_model)
])

## Fitting on training data
final_pipeline.fit(X_train, y_train)


# In[96]:


## Generate predictions the model
y_train_pred = final_pipeline.predict(X_train)
y_test_pred = final_pipeline.predict(X_test)


# In[97]:


## Calculate Performance Metrics
rf_metrics_summary = {
    "Metric": ["R² Score", "Mean Absolute Error (MAE)", "Root Mean Squared Error (RMSE)"],
    "Train Set": [
        r2_score(y_train, y_train_pred),
        mean_absolute_error(y_train, y_train_pred),
        np.sqrt(mean_squared_error(y_train, y_train_pred))
    ],
    "Test Set": [
        r2_score(y_test, y_test_pred),
        mean_absolute_error(y_test, y_test_pred),
        np.sqrt(mean_squared_error(y_test, y_test_pred))
    ]
}

## Format and display the results cleanly
rf_metrics_df = pd.DataFrame(rf_metrics_summary)
print("\n--- Final Random Forest Model Evaluation Summary ---\n")
print(rf_metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# - #### Random Forest (WIth Hyper-parameter Tuning)

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV


## Create instantiate
rf_reg = RandomForestRegressor(random_state=42)


# In[48]:


# param_grid = {
#     "n_estimators": [100, 200],
#     "max_depth": [5, 10, 15],
#     "min_samples_split": [2, 5, 10],
#     "min_samples_leaf": [1, 2, 4],
#     "max_features": ["sqrt", "log2"]
# }


param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 8, 10, 12],
    "min_samples_split": [5, 10, 20],
    "min_samples_leaf": [2, 4, 8],
    "max_features": ["sqrt", "log2"]
}


# In[49]:


random_search = RandomizedSearchCV(
    estimator=rf_reg,
    param_distributions=param_grid,
    n_iter=20,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    verbose=2,
    random_state=42
)

random_search.fit(X_train_trf, y_train)


# In[50]:


random_search.best_estimator_


# In[51]:


random_search.best_score_


# In[52]:


best_rf_param = random_search.best_params_
best_rf_param


# In[53]:


rf_reg_model = RandomForestRegressor(**best_rf_param)
rf_reg_model.fit(X_train_trf, y_train)


# In[54]:


y_pred = rf_reg_model.predict(X_test_trf)
y_train_pred = rf_reg_model.predict(X_train_trf)


# In[55]:


rf_reg_model_train_r2_score = r2_score(y_true=y_train, y_pred=y_train_pred)
rf_reg_model_train_r2_score


# In[56]:


rf_reg_model_train_r2_score = r2_score(y_true=y_test, y_pred=y_pred)
rf_reg_model_train_r2_score


# - #### Ada Boost Regressor

# In[57]:


from sklearn.ensemble import AdaBoostRegressor

## Create instantiate
ada_reg = AdaBoostRegressor(estimator=dt_model, random_state=42, n_estimators=150, learning_rate=0.02)
ada_reg.fit(X_train_trf, y_train)


# In[58]:


y_train_pred = ada_reg.predict(X_train_trf)
y_test_pred = ada_reg.predict(X_test_trf)


# In[59]:


r2_score(y_train, y_train_pred)


# In[60]:


r2_score(y_test, y_test_pred)


# - #### Gradient Boost Regressor

# In[61]:


from sklearn.ensemble import GradientBoostingRegressor

## Create instantiate
gb_reg = GradientBoostingRegressor(random_state=42)
gb_reg.fit(X_train_trf, y_train)


# In[62]:


y_train_pred = gb_reg.predict(X_train_trf)
y_test_pred = gb_reg.predict(X_test_trf)


# In[63]:


r2_score(y_train, y_train_pred)


# In[64]:


r2_score(y_test, y_test_pred)


# - #### Gradient Boost Regressor(with Hyper-Parameter)

# In[120]:


## Create instantiate
gb_reg = GradientBoostingRegressor(max_depth=3, min_samples_split=10, min_samples_leaf=5, max_features=0.5, n_estimators=200, learning_rate=0.03, random_state=42)
gb_reg.fit(X_train_trf, y_train)

y_train_pred = gb_reg.predict(X_train_trf)
y_test_pred = gb_reg.predict(X_test_trf)


# In[121]:


r2_score(y_train, y_train_pred)


# In[122]:


r2_score(y_test, y_test_pred)


# - #### XGBoost

# In[68]:


from xgboost import XGBRegressor

xgb_model = XGBRegressor(
    n_estimators=200,
    learning_rate = 0.02,
    random_state=42
)


# In[69]:


xgb_model.fit(X_train_trf,y_train)


# In[70]:


y_train_pred = xgb_model.predict(X_train_trf)
y_test_pred = xgb_model.predict(X_test_trf)


# In[71]:


r2_score(y_train, y_train_pred)


# In[72]:


r2_score(y_test, y_test_pred)


# - #### XGB Regressor (with hyper-parameters)

# In[ ]:


# from xgboost import XGBRegressor
# from sklearn.model_selection import RandomizedSearchCV
# from scipy.stats import randint as sp_randint
# from scipy.stats import uniform

# xgb = XGBRegressor(random_state=42, objective='reg:squarederror')

# param_dist = {
#     'n_estimators': [100, 200, 300],
#     'learning_rate': [0.01, 0.05, 0.1, 0.2],
#     'max_depth': sp_randint(3, 10),
#     'subsample': uniform(0.6, 0.4),
#     'colsample_bytree': uniform(0.6, 0.4),
#     'reg_lambda': [1, 2, 5, 10]
# }

# random_search = RandomizedSearchCV(
#     estimator=xgb, 
#     param_distributions=param_dist,
#     n_iter=20, 
#     cv=5,      
#     scoring='r2', # Optimizes for R-squared
#     random_state=42,
#     n_jobs=-1  
# )

# random_search.fit(X_train_trf, y_train)

# best_prm = random_search.best_params_

# xgb_tune = XGBRegressor(**best_prm)
# xgb_tune.fit(X_train_trf, y_train)

# xgb_tune_train_pred = xgb_tune.predict(X_train_trf)
# xgb_tune_test_pred = xgb_tune.predict(X_test_trf)

# print("Training R2 Score: ", r2_score(y_train, xgb_tune_train_pred))
# print("Testing R2 Score: ", r2_score(y_test, xgb_tune_test_pred))


# In[73]:


xgb_model = XGBRegressor(max_depth=5, n_estimators=150, learning_rate=0.02, random_state=42)


# In[74]:


xgb_model.fit(X_train_trf,y_train)


# In[75]:


y_train_pred = xgb_model.predict(X_train_trf)
y_test_pred = xgb_model.predict(X_test_trf)


# In[76]:


r2_score(y_train, y_train_pred)


# In[77]:


r2_score(y_test, y_test_pred)


# ### Stacking

# In[130]:


from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# 1. Include Linear Regression inside the base estimators
base_estimators = [
    ('lr_base', LinearRegression()), # Forces the stack to learn the dominant linear trends
    ('rf', RandomForestRegressor(max_depth=4, min_samples_leaf=15, random_state=42)),
    ('xgb', XGBRegressor(max_depth=3, learning_rate=0.03, random_state=42))
]

# 2. Use a highly regularized Ridge regression as the final judge
stacking_model = StackingRegressor(
    estimators=base_estimators,
    final_estimator=Ridge(alpha=50.0) # High alpha prevents overfitting on the final prediction blend
)

# 3. Train the model
stacking_model.fit(X_train_trf, y_train_log)


# 4. Check the new scores
train_score = stacking_model.score(X_train_trf, y_train_log)
test_score = stacking_model.score(X_test_trf, y_test_log)

print(f"Stacking Train R²: {train_score:.2f}")
print(f"Stacking Test R²: {test_score:.2f}")


# In[131]:


## Evaluation
y_train_stacking_pred = reversed_log_transformer(stacking_model.predict(X_train_trf))
y_test_stacking_pred = reversed_log_transformer(stacking_model.predict(X_test_trf))

train_score = r2_score(y_true=y_train, y_pred=y_train_stacking_pred)
test_score = r2_score(y_true=y_test, y_pred=y_test_stacking_pred)

print(f"Stacking Train R²: {train_score:.2f}")
print(f"Stacking Test R²: {test_score:.2f}")


# ## Final Model
# 
# - #### Gradient Boosting

# In[125]:


from sklearn.ensemble import GradientBoostingRegressor

## Create instantiate
gb_reg = GradientBoostingRegressor(random_state=42)
gb_reg.fit(X_train_trf, y_train_log)

y_train_pred = reversed_log_transformer(gb_reg.predict(X_train_trf))
y_test_pred = reversed_log_transformer(gb_reg.predict(X_test_trf))

print("Training R2 Score: ",r2_score(y_train, y_train_pred))
print("Testing R2 Score: ",r2_score(y_test, y_test_pred))


# - ##### Feature Importance

# In[129]:


gb_reg.feature_importances_


# In[131]:


## Numeric pipelin
numeric_features = ["Rating", "Reviews", "ram_gb", "storage_gb", "discount_pct(%)"]

## Yeo-Johnson Pipeline
yeo_features = ["Stars"]

## Target Encoder
target_features = ["brand"]

## OneHot Features
ohe_features = transformer.named_transformers_[
    "OneHotEncoding"
].get_feature_names_out(
    ["storage_type", "processor_brand"]
)
## Ordinal Features
ordinal_features = ["processor_tier"]

## Passthrough Features
passthrough_features = ["processor_generation", "is_gaming", "premium_brand"]


feature_names = (
    numeric_features
    + yeo_features
    + target_features
    + list(ohe_features)
    + ordinal_features
    + passthrough_features
)

print(len(feature_names))

print(len(gb_reg.feature_importances_))


# In[132]:


feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": gb_reg.feature_importances_
})

feature_importance.sort_values(by="Importance",ascending=False,inplace=True)

feature_importance.head(15)


# In[135]:


plt.figure(figsize=(12,8))

plt.scatter(y_test, y_test_pred)
plt.xlabel("Actual value")
plt.ylabel("Predicte value")
plt.title("Actual vs Predicted");


# In[137]:


residuals = y_test - y_test_pred
residuals


# In[138]:


plt.figure(figsize=(8,5))

sns.histplot(residuals,kde=True)
plt.title("Residual Distribution")
plt.show()


# In[140]:


plt.figure(figsize=(8,5))

plt.scatter(y_test_pred,residuals,alpha=0.6)

plt.axhline(y=0,color="red",linestyle="--")

plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Residual Analysis")

plt.show()


# In[ ]:





# In[ ]:




