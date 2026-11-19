
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

# ==================================================
# 1. EINSTELLUNGEN
# ==================================================

SEQ_LEN = 80
BATCH_SIZE = 128
EPOCHS = 100

np.random.seed(42)
tf.random.set_seed(42)

# ==================================================
# 2. DATEN LADEN
# ==================================================

train_raw = pd.read_csv("train.csv")
test_raw = pd.read_csv("test.csv")

print("Train:", train_raw.shape)
print("Test:", test_raw.shape)

# ==================================================
# 3. FEATURE ENGINEERING
# ==================================================

def prepare_data(df):

    df = df.sort_values(
        ["breath_id", "time_step"]
    ).copy()

    group = df.groupby("breath_id", sort=False)

    # Kumulierte Luftmenge
    df["u_in_cumsum"] = group["u_in"].cumsum()

    # Veränderungen des Luftflusses
    df["u_in_diff1"] = (
        group["u_in"].diff().fillna(0)
    )

    df["u_in_diff2"] = (
        df.groupby("breath_id", sort=False)
        ["u_in_diff1"].diff().fillna(0)
    )

    # Vergangene Messwerte
    df["u_in_lag1"] = (
        group["u_in"].shift(1).fillna(0)
    )

    df["u_in_lag2"] = (
        group["u_in"].shift(2).fillna(0)
    )

    # Logarithmischer Luftfluss
    df["log_u_in"] = np.log1p(
        df["u_in"].clip(lower=0)
    )

    # Maximale Luftmenge pro Atemzug
    df["u_in_max"] = (
        df.groupby("breath_id", sort=False)
        ["u_in"].transform("max")
    )

    # Zeitliche Integration des Luftflusses
    dt = (
        df.groupby("breath_id", sort=False)
        ["time_step"].diff().fillna(0)
    )

    df["area"] = (
        df["u_in"] * dt
    ).groupby(
        df["breath_id"], sort=False
    ).cumsum()

    return df


train = prepare_data(train_raw)
test = prepare_data(test_raw)

# ==================================================
# 4. FEATURES DEFINIEREN
# ==================================================

features = [
    "R",
    "C",
    "time_step",
    "u_in",
    "u_out",
    "u_in_cumsum",
    "u_in_diff1",
    "u_in_diff2",
    "u_in_lag1",
    "u_in_lag2",
    "log_u_in",
    "u_in_max",
    "area"
]

# Features auf float32 reduzieren
for col in features:
    train[col] = train[col].astype(np.float32)
    test[col] = test[col].astype(np.float32)

# ==================================================
# 5. SEQUENZEN ERSTELLEN
# ==================================================

def create_sequences(df, feature_columns):

    counts = df.groupby(
        "breath_id", sort=False
    ).size()

    if not (counts == SEQ_LEN).all():
        raise ValueError(
            "Nicht jeder Atemzug hat genau 80 Messpunkte!"
        )

    X = df[feature_columns].to_numpy(
        dtype=np.float32
    )

    X = X.reshape(
        len(counts),
        SEQ_LEN,
        len(feature_columns)
    )

    return X


X = create_sequences(train, features)

X_test = create_sequences(test, features)

y = train["pressure"].to_numpy(
    dtype=np.float32
).reshape(-1, SEQ_LEN, 1)

print("X:", X.shape)
print("y:", y.shape)
print("X_test:", X_test.shape)

# ==================================================
# 6. TRAIN / VALIDATION SPLIT
# ==================================================

# Ganze Atemzüge aufteilen, nicht einzelne Zeilen!

indices = np.random.permutation(len(X))

n_val = int(len(X) * 0.1)

val_idx = indices[:n_val]
train_idx = indices[n_val:]

X_train = X[train_idx]
X_val = X[val_idx]

y_train = y[train_idx]
y_val = y[val_idx]

print("Train breaths:", len(X_train))
print("Validation breaths:", len(X_val))

# ==================================================
# 7. NORMALISIERUNG
# ==================================================

normalizer = layers.Normalization(axis=-1)

normalizer.adapt(
    X_train.reshape(-1, len(features))
)

# ==================================================
# 8. MASK FÜR KAGGLE-MAE
# ==================================================

# Nur Inspirationsphase: u_out == 0

u_out_index = features.index("u_out")

train_weights = (
    X_train[:, :, u_out_index] == 0
).astype(np.float32)

val_weights = (
    X_val[:, :, u_out_index] == 0
).astype(np.float32)

# ==================================================
# 9. CNN + BIDIRECTIONAL GRU
# ==================================================

inputs = keras.Input(
    shape=(SEQ_LEN, len(features))
)

x = normalizer(inputs)

# CNN erkennt lokale Muster
x = layers.Conv1D(
    64,
    kernel_size=5,
    padding="same",
    activation="swish"
)(x)

x = layers.Conv1D(
    64,
    kernel_size=3,
    padding="same",
    activation="swish"
)(x)

# GRU erkennt zeitliche Zusammenhänge
x = layers.Bidirectional(
    layers.GRU(
        32,
        return_sequences=True
    )
)(x)

x = layers.Dense(
    64,
    activation="swish"
)(x)

x = layers.Dropout(0.15)(x)

outputs = layers.Dense(1)(x)

model = keras.Model(
    inputs=inputs,
    outputs=outputs
)

# ==================================================
# 10. MODELL KOMPILIEREN
# ==================================================

model.compile(

    optimizer=keras.optimizers.Adam(
        learning_rate=0.0005
    ),

    loss="mae",

    weighted_metrics=[
        keras.metrics.MeanAbsoluteError(
            name="masked_mae"
        )
    ]

)

model.summary()

# ==================================================
# 11. CALLBACKS
# ==================================================

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_masked_mae",
    patience=8,
    min_delta=0.0001,
    mode="min",
    restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_masked_mae",
    factor=0.5,
    patience=3,
    min_lr=0.00001,
    mode="min"
)

checkpoint = keras.callbacks.ModelCheckpoint(
    "best_pressure.keras",
    monitor="val_masked_mae",
    save_best_only=True,
    mode="min"
)

# ==================================================
# 12. TRAINING
# ==================================================

history = model.fit(

    X_train,
    y_train,

    sample_weight=train_weights,

    validation_data=(
        X_val,
        y_val,
        val_weights
    ),

    batch_size=BATCH_SIZE,

    epochs=EPOCHS,

    callbacks=[
        early_stopping,
        reduce_lr,
        checkpoint
    ],

    shuffle=True,

    verbose=1

)

# ==================================================
# 13. VALIDATION SCORE
# ==================================================

val_predictions = model.predict(
    X_val,
    batch_size=BATCH_SIZE
)

mask = (
    X_val[:, :, u_out_index] == 0
)

errors = np.abs(
    y_val[:, :, 0] -
    val_predictions[:, :, 0]
)

kaggle_mae = errors[mask].mean()

print("--------------------------------")
print("VALIDATION RESULTS")
print("--------------------------------")

print("Kaggle Validation MAE:", kaggle_mae)

# ==================================================
# 14. TRAINING DIAGRAMM
# ==================================================

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["masked_mae"],
    label="Training MAE"
)

plt.plot(
    history.history["val_masked_mae"],
    label="Validation MAE"
)

plt.title("Kaggle Masked MAE")

plt.xlabel("Epoch")
plt.ylabel("MAE")

plt.legend()
plt.grid()

plt.show()

# ==================================================
# 15. TEST VORHERSAGEN
# ==================================================

test_predictions = model.predict(
    X_test,
    batch_size=BATCH_SIZE
).reshape(-1)

# Vorhersagen anhand der ursprünglichen ID zuordnen
prediction_map = pd.Series(
    test_predictions,
    index=test["id"].to_numpy()
)

submission = pd.DataFrame({
    "id": test_raw["id"].to_numpy()
})

submission["pressure"] = (
    submission["id"].map(prediction_map)
)

if submission["pressure"].isna().any():
    raise ValueError(
        "Einige Testvorhersagen fehlen!"
    )

submission.to_csv(
    "submission.csv",
    index=False
)

print("--------------------------------")
print("SUBMISSION ERSTELLT")
print("--------------------------------")

print(submission.head())

model.save("final_pressure_model.keras")
