import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

# Feature engineering function
def feature_engineering(df):
    # Total Bathrooms (above and basement combined, full + half)
    df['TotalBath'] = (df['FullBath'] + 0.5 * df['HalfBath'] +
                       df.get('BsmtFullBath', 0) + 0.5 * df.get('BsmtHalfBath', 0))

    # Total Porch Area
    df['TotalPorchSF'] = (df['OpenPorchSF'] + df['EnclosedPorch'] +
                          df['3SsnPorch'] + df['ScreenPorch'])

    # Basement to living area ratio
    df['BasementRatio'] = df['TotalBsmtSF'] / (df['GrLivArea'] + 1)

    # Age of the house at sale time
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']

    # Years since remodel
    df['RemodelAge'] = df['YrSold'] - df['YearRemodAdd']

    # Garage age (fill missing GarageYrBlt with YearBuilt)
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])
    df['GarageAge'] = df['YrSold'] - df['GarageYrBlt']

    # Total square footage (living + basement)
    df['TotalSF'] = df['GrLivArea'] + df['TotalBsmtSF']

    # Total rooms above grade + approx basement rooms
    df['TotalRooms'] = df['TotRmsAbvGrd'] + df.get('BsmtFinSF1', 0) / 1000  # scaled basement finished SF

    # Fireplace indicator (1 if house has fireplace, else 0)
    df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)

    # Deck + porch combined area
    df['DeckPorchSF'] = df['WoodDeckSF'] + df['TotalPorchSF']

    # Garage area per car (avoid div by zero)
    df['GarageAreaPerCar'] = df['GarageArea'] / (df['GarageCars'] + 1)

    # Lot depth (lot area / frontage)
    df['LotDepth'] = df['LotArea'] / (df['LotFrontage'] + 1)

    # Miscellaneous value flag
    df['HasMisc'] = (df['MiscVal'] > 0).astype(int)

    # Months since sold (to capture seasonal trends)
    df['MonthsSinceSold'] = (2025 - df['YrSold']) * 12 + (7 - df['MoSold'])  # Adjust as needed

    return df


# Load data
house_data = pd.read_csv("train.csv", na_values=["NA"])
house_data_test = pd.read_csv("test.csv", na_values=['NA'])
test_ids = house_data_test["Id"]

# Drop Id columns
house_data.drop(columns="Id", inplace=True)
house_data_test.drop(columns="Id", inplace=True)

# Drop columns with >50% missing data
null_percentage = (house_data.isnull().sum() / len(house_data)) * 100
cols_to_drop = null_percentage[null_percentage > 50].index
house_data.drop(columns=cols_to_drop, inplace=True)
house_data_test.drop(columns=cols_to_drop, inplace=True)

# Fill missing values
def fill_nulls(df):
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].mean())
    return df

house_data = fill_nulls(house_data)
house_data_test = fill_nulls(house_data_test)

# Apply feature engineering
house_data = feature_engineering(house_data)
house_data_test = feature_engineering(house_data_test)

# Log-transform the target
house_data["SalePrice"] = np.log1p(house_data["SalePrice"])

# Prepare features and target lists
NUMERICAL_COLS = house_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
TARGET = house_data["SalePrice"]
NUMERICAL_COLS.remove("SalePrice")
CATEGORICAL_COLS = house_data.select_dtypes(include=['object']).columns.tolist()
FEATURES = house_data[NUMERICAL_COLS + CATEGORICAL_COLS]
TEST_SET = house_data_test[FEATURES.columns]

# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmsle_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(FEATURES)):
    print(f"Fold {fold + 1}")
    X_train, X_val = FEATURES.iloc[train_idx], FEATURES.iloc[val_idx]
    y_train, y_val = TARGET.iloc[train_idx], TARGET.iloc[val_idx]

    train_pool = Pool(X_train, y_train, cat_features=CATEGORICAL_COLS)
    val_pool = Pool(X_val, y_val, cat_features=CATEGORICAL_COLS)

    model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.007,
        depth=8,
        l2_leaf_reg=6,
        random_seed=42,
        early_stopping_rounds=50,
        eval_metric='RMSE',
        verbose=100
    )
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    val_pred_log = model.predict(X_val)
    val_pred = np.expm1(val_pred_log)
    y_val_orig = np.expm1(y_val)

    rmsle_score = np.sqrt(mean_squared_error(np.log1p(y_val_orig), np.log1p(val_pred)))
    print(f"Fold {fold + 1} RMSLE: {rmsle_score:.5f}")
    rmsle_scores.append(rmsle_score)

print(f"\nAverage RMSLE across folds: {np.mean(rmsle_scores):.5f}")

# Train final model on full data
final_pool = Pool(FEATURES, TARGET, cat_features=CATEGORICAL_COLS)
final_model = CatBoostRegressor(
    iterations=3000,
    learning_rate=0.007,
    depth=8,
    l2_leaf_reg=6,
    random_seed=42,
    early_stopping_rounds=50,
    eval_metric='RMSE',
    verbose=100
)
final_model.fit(final_pool)

# Predict on test set
test_pred_log = final_model.predict(TEST_SET)
test_pred = np.expm1(test_pred_log)

# Save submission
submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": test_pred
})
submission.to_csv("Submission3.csv", index=False)
print("Submission saved!")
