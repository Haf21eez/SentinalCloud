from flask import Flask, render_template
import pandas as pd
import random

app = Flask(__name__)


# Attack label mapping
attack_map = {
    0: ("Benign", "green"),
    1: ("DDoS", "red"),
    2: ("Botnet", "orange"),
    3: ("Infiltration", "yellow")
}

severity_map = {
    "Benign": "Low",
    "DDoS": "Critical",
    "Botnet": "High",
    "Infiltration": "Medium"
}

# -------------------------------------------------
# COMMON DATA FUNCTION (used by ALL pages)
# -------------------------------------------------

def load_data():
    df = pd.concat([
        pd.read_csv("data/Benign-Monday.csv").sample(15),
        pd.read_csv("data/DDoS-Friday.csv").sample(15),
        pd.read_csv("data/Botnet-Friday.csv").sample(15),
        pd.read_csv("data/Infiltration-Thursday.csv").sample(15)
    ])
    return df

df = load_data()
records = df.to_dict(orient="records")

def generate_records():
    df = pd.concat([
        pd.read_csv("data/Benign-Monday.csv").sample(15),
        pd.read_csv("data/DDoS-Friday.csv").sample(15),
        pd.read_csv("data/Botnet-Friday.csv").sample(15),
        pd.read_csv("data/Infiltration-Thursday.csv").sample(15)
    ], ignore_index=True)

    df["user"] = [f"user_{i}" for i in range(len(df))]
    df["ip"] = [f"192.168.1.{random.randint(1,255)}" for _ in range(len(df))]
    df["api_calls"] = [random.randint(10, 500) for _ in range(len(df))]
    df["data_mb"] = [round(random.uniform(1, 150), 2) for _ in range(len(df))]
    df["login_failures"] = [random.randint(0, 10) for _ in range(len(df))]

    df["attack_code"] = [random.choice([0,1,2,3]) for _ in range(len(df))]
    df["attack_type"] = df["attack_code"].apply(lambda x: attack_map[x][0])
    df["color"] = df["attack_code"].apply(lambda x: attack_map[x][1])
    df["severity"] = df["attack_type"].map(severity_map)

    records = df.to_dict(orient="records")

    summary = {
        "total": int(len(df)),
        "benign": int((df["attack_type"] == "Benign").sum()),
        "ddos": int((df["attack_type"] == "DDoS").sum()),
        "botnet": int((df["attack_type"] == "Botnet").sum()),
        "infiltration": int((df["attack_type"] == "Infiltration").sum())
    }

    return records, summary

# -------------------------------------------------
# ROUTES
# -------------------------------------------------
@app.route("/")
def dashboard():
    records, summary = generate_records()
    return render_template("dashboard.html", records=records, summary=summary)

@app.route("/active-attacks")
def active_attacks():
    records, _ = generate_records()
    active = [r for r in records if r["attack_type"] != "Benign"]
    return render_template("active_attacks.html", records=active)

@app.route("/severity")
def severity():
    records, _ = generate_records()
    return render_template("severity.html", records=records)

@app.route("/settings")
def settings():
    return render_template("settings.html")

# -------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
