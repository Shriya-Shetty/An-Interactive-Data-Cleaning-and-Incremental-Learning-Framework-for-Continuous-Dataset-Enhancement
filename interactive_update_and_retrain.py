import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# ---------------- CONFIG ----------------
DATASET_PATH = r"freelancer_earnings - freelancer_earnings_vs_skillstack_dataset.csv"
DROP_COLS = ["freelancer_id"]
RANDOM_STATE = 42

FIELDS = [
    "category",
    "primary_skills",
    "years_experience",
    "experience_level",
    "region",
    "country",
    "education",
    "hourly_rate_usd",
    "primary_platform",
    "annual_income_usd"
]

NUMERIC_FIELDS = [
    "years_experience",
    "hourly_rate_usd",
    "annual_income_usd"
]

# ---------------- LOAD DATA ----------------
df = pd.read_csv(DATASET_PATH)

df["annual_income_usd"] = (
    df["annual_income_usd"]
    .replace('[\$,]', '', regex=True)
    .astype(float)
)

df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

print("\n=== Interactive Progressive Data Completion ===")
print("Press ENTER to accept suggestions, or type to override.\n")

filled_data = {}
remaining_fields = FIELDS.copy()

# ---------------- PROMPTING LOOP ----------------
while remaining_fields:
    print("\nRemaining fields:")
    for i, f in enumerate(remaining_fields):
        print(f"{i+1}. {f}")

    idx = int(input("\nChoose a field to fill (number): ")) - 1
    target = remaining_fields[idx]

    model = joblib.load(f"model_{target}.pkl")

    # Prepare input
    row = {col: filled_data.get(col, None) for col in FIELDS}
    df_input = pd.DataFrame([row])

    try:
        raw_pred = model.predict(df_input)[0]
    except:
        raw_pred = None

    # Decode categorical predictions
    encoder_path = f"encoder_{target}.pkl"
    if raw_pred is not None and os.path.exists(encoder_path):
        le = joblib.load(encoder_path)
        suggestion = le.inverse_transform([int(raw_pred)])[0]
    else:
        suggestion = raw_pred

    # User input
    if suggestion is not None:
        if target in NUMERIC_FIELDS:
            user_val = input(
                f"Suggested {target}: {round(float(suggestion), 2)} "
                f"(Enter to accept / type value): "
            )
            value = float(user_val) if user_val else float(suggestion)
        else:
            user_val = input(
                f"Suggested {target}: {suggestion} "
                f"(Enter to accept / type value): "
            )
            value = user_val if user_val else suggestion
    else:
        user_val = input(f"Enter value for {target}: ")
        value = float(user_val) if target in NUMERIC_FIELDS else user_val

    filled_data[target] = value
    remaining_fields.remove(target)

# ---------------- CLEAN FINAL ENTRY ----------------
clean_entry = {}

for k, v in filled_data.items():
    if k in NUMERIC_FIELDS:
        clean_entry[k] = float(v)
    else:
        clean_entry[k] = str(v).strip()

print("\n=== Final Cleaned Entry ===")
for k, v in clean_entry.items():
    print(f"{k}: {v}")

# ---------------- ASK TO SAVE ----------------
save = input("\nAdd this entry to dataset and retrain models? (yes/no): ").lower()

if save != "yes":
    print("Entry discarded. No changes made.")
    exit()

# ---------------- SAVE TO DATASET ----------------
df = pd.concat([df, pd.DataFrame([clean_entry])], ignore_index=True)
df.to_csv(DATASET_PATH, index=False)

print("Entry added to dataset.")

# ---------------- RETRAIN ALL MODELS ----------------
print("\nRetraining all column-wise models...")

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

for target in df.columns:
    print(f"Retraining model for: {target}")

    X = df.drop(columns=[target])
    y = df[target]

    if target in numeric_cols:
        model = RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    else:
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        le = LabelEncoder()
        y = le.fit_transform(y)
        joblib.dump(le, f"encoder_{target}.pkl")

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),

        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ])

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, f"model_{target}.pkl")

print("\n✅ Dataset updated and all models retrained successfully.")
