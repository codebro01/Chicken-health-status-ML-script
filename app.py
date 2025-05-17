import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import random
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix


# Step 1: Simulate data
def generate_chicken_data(n=1000):
    data = []
    for _ in range(n):
        temp = round(random.uniform(39.0, 44.0), 1)
        heart = random.randint(200, 400)
        activity = random.randint(1, 10)
        appetite = random.randint(1, 10)
        feather = random.choice(["smooth", "ruffled"])
        comb = random.choice(["bright red", "pale", "bluish"])
        resp = random.randint(10, 40)
        age = random.randint(1, 100)

        # Generate health status using simple rule logic
        score = sum([

            40.0 <= temp <= 42.5,
            250 <= heart <= 350,
            activity >= 7,
            appetite >= 7,
            feather == "smooth",
            comb == "bright red",
            15 <= resp <= 35,
            4 <= age <= 80
        ])

        if score >= 6:
            status = "Healthy"
        elif score >= 3 and score < 6:
            status = "Unwell"
        else:
            status = "Critical"

        data.append([temp, heart, activity, appetite, feather, comb, resp, age, status])

    columns = [
        "temperature", "heartRate", "activityLevel", "appetiteLevel",
        "featherCondition", "combColor", "respiratoryRate", "ageInWeeks", "healthStatus"
    ]
    return pd.DataFrame(data, columns=columns)

# Step 2: Create dataset
df = generate_chicken_data(1000)

# Step 3: Encode categorical data
le_feather = LabelEncoder()
le_comb = LabelEncoder()
le_status = LabelEncoder()

df["featherCondition"] = le_feather.fit_transform(df["featherCondition"])
df["combColor"] = le_comb.fit_transform(df["combColor"])
df["healthStatus"] = le_status.fit_transform(df["healthStatus"])

# Step 4: Train-test split
X = df.drop("healthStatus", axis=1)
y = df["healthStatus"]

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)


# Step 5: Train model

model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)


# df["healthStatus"].value_counts().plot(kind="bar", title="Class Distribution")
# plt.show()
# Step 6: Evaluate

y_pred = model.predict(X_test);

print(confusion_matrix(y_test, y_pred))


print(classification_report(y_test, y_pred, target_names=le_status.classes_))
print(df["healthStatus"].value_counts())


# Step 7: Save model & encoders
# joblib.dump(model, "chicken_health_model.pkl")
# joblib.dump(le_feather, "feather_encoder.pkl")
# joblib.dump(le_comb, "comb_encoder.pkl")
# joblib.dump(le_status, "status_encoder.pkl")

model_bundle = {
    "model": model,
    "feather_encoder": le_feather,
    "comb_encoder": le_comb,
    "status_encoder": le_status
}

# Dump the entire bundle to one file
joblib.dump(model_bundle, "chicken_health_bundle.joblib")

