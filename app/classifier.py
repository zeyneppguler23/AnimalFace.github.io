from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

class AnimalFaceClassifier:
    def __init__(self):
        self.model = RandomForestClassifier()
        self.encoder = LabelEncoder()

    def preprocess(self, data):
        features = [x[:-1] for x in data]
        labels = [x[-1] for x in data]

        # Encode features and labels
        for i in range(len(features[0])):
            col = [f[i] for f in features]
            features = [[self.encoder.fit_transform(col)[j] for col in zip(*features)] for j in range(len(features))]
        
        encoded_labels = self.encoder.fit_transform(labels)
        return features, encoded_labels

    def train(self, data):
        features, labels = self.preprocess(data)
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        print("Model trained successfully.")
        return self.model

    def predict(self, features):
        encoded_features = self.encoder.transform(features)
        prediction = self.model.predict([encoded_features])
        return self.encoder.inverse_transform(prediction)

    def save_model(self, path):
        joblib.dump({"model": self.model, "encoder": self.encoder}, path)

    def load_model(self, path):
        data = joblib.load(path)
        self.model = data["model"]
        self.encoder = data["encoder"]

