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
        file = request.files.get("image")
        if not file:
            return jsonify({"error": "No file uploaded."}), 400
        
        file_path = os.path.join("data/sample_images", file.filename)
        file.save(file_path)

        # Extract features
        landmarks = extract_landmarks(file_path)
        if not landmarks:
            return jsonify({"error": "No face detected."}), 400

        # Predict
        prediction = classifier.predict(landmarks)
        return jsonify({"animal_face_type": prediction})

    # Render the HTML template for GET request
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
