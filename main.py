from flask import Flask, request, jsonify, render_template
from app.feature_extraction import extract_landmarks
from app.classifier import AnimalFaceClassifier
import os

app = Flask(__name__)
classifier = AnimalFaceClassifier()
classifier.load_model("model/animal_face_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        file_path = os.path.join("data/sample_images", file.filename)
        file.save(file_path)

        landmarks = extract_landmarks(file_path)
        if not landmarks:
            return jsonify({"error": "No face detected."}), 400

        # Predict animal face type
        prediction = classifier.predict(landmarks)
        return jsonify({"animal_face_type": prediction})

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

