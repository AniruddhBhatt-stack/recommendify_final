import os
import pickle
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

song_CSV = 'song.csv'

def train_model():
    if not os.path.exists(song_CSV):
        print(f"Error: {song_CSV} not found.")
        return None, None

    df = pd.read_csv(song_CSV)
    print(f"Model trained on song.csv with {len(df)} samples")

    # Preprocess genres
    df['genres_list'] = df['Genres'].fillna('').apply(
        lambda x: [g.strip() for g in x.split(',')] if x else []
    )

    # Encode genres
    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(df['genres_list'])

    # Mock labels: just to train a binary model
    y = [1 if i < len(df) / 2 else 0 for i in range(len(df))]

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy on test data: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    with open('model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'mlb': mlb}, f)
    print("Saved model and encoder to model.pkl")

    return model, mlb
