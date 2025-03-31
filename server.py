from flask import Flask, request, jsonify, render_template
import os
import model


app = Flask(__name__)

# Paths
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
# Folder where encrypted files are stored
ENCRYPTED_FOLDER = "D:/My_Projects/SBH/encrypted_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
decode_p=[]
def clear_folder(folder_path):
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)  # Remove file
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

# Clear the folders on app start
clear_folder(UPLOAD_FOLDER)
clear_folder(ENCRYPTED_FOLDER)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Process File
        data, out_c, context = model.load()
        encrypted_features, labels = model.encrypt_data(data, context, out_c)
        n = file.filename
        name = os.path.splitext(n)[0]
        model.save_encrypted_data(encrypted_features, labels, ENCRYPTED_FOLDER, f"{name}.json")

        return jsonify({"message": "File uploaded successfully", "filename": file.filename}), 200


@app.route('/fetch-encrypted-files', methods=['GET'])
def fetch_encrypted_files():
    """Fetch list of encrypted files in the folder"""
    files = [f for f in os.listdir(ENCRYPTED_FOLDER) if f.endswith(".json")]  # Change extension if needed
    return jsonify({"files": files})


@app.route('/fetch-encrypted-file', methods=['GET'])
def fetch_encrypted_file():
    """Fetch encrypted file content"""
    filename = request.args.get('filename')
    file_path = os.path.join(ENCRYPTED_FOLDER, filename)

    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    return jsonify({"filename": filename, "content": content})

@app.route('/train-model', methods=['POST'])
def train_model():
    # Simulated training process (replace with actual training code)
    data, out_c, context = model.load()
    encrypted_features, labels = model.encrypt_data1(data, context, out_c)
    encrypted_weights, encrypted_bias, accuracy = model.load_model(data, out_c, context)
    e_p = model.encrypted_predict(encrypted_features, encrypted_weights, encrypted_bias)
    decode_p.clear()
    decode_p.append(model.decode(e_p))
    print(e_p)
    print(round(accuracy))
    accuracy = int(accuracy)

    encode_p = list(e_p)

    # print(encode_p)
    return jsonify({
        "accuracy": accuracy,

        "Encrypted_Label": [str(vec) for vec in encode_p],
    })
@app.route('/decode', methods=['POST'])
def decode():
    dp= [str(num) for sublist in decode_p for num in sublist]
    print(decode_p)
    return jsonify({
        "predicted_labels": dp,
    })
@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)