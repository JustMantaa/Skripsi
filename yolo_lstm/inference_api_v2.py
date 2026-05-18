import time
from flask import Flask, jsonify, request
from flask_cors import CORS

try:
    # when package-imported
    from .model_service import predict_from_image_dataurl
except Exception:
    # when run as a script from the same folder
    from model_service import predict_from_image_dataurl


BASE_HOST = "127.0.0.1"
BASE_PORT = 5000

app = Flask(__name__)
CORS(app)


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    image = payload.get("image")

    if not image:
        return jsonify({"error": "missing image"}), 400

    response = predict_from_image_dataurl(image)
    return jsonify(response)


if __name__ == "__main__":
    print(f"API READY at http://{BASE_HOST}:{BASE_PORT}")
    app.run(host=BASE_HOST, port=BASE_PORT, debug=False)
