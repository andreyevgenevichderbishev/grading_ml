import os
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
import predictor

UPLOAD_FOLDER = os.path.join("static", "uploads")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "avers" not in request.files or "revers" not in request.files:
            return "Ошибка: загрузите оба изображения."
        file_avers = request.files["avers"]
        file_revers = request.files["revers"]
        if file_avers.filename == "" or file_revers.filename == "":
            return "Ошибка: один из файлов не выбран."
        if allowed_file(file_avers.filename) and allowed_file(file_revers.filename):
            filename_avers = secure_filename(file_avers.filename)
            filename_revers = secure_filename(file_revers.filename)
            path_avers = os.path.join(app.config["UPLOAD_FOLDER"], filename_avers)
            path_revers = os.path.join(app.config["UPLOAD_FOLDER"], filename_revers)
            file_avers.save(path_avers)
            file_revers.save(path_revers)
            
            # Используем функцию предсказания
            result = predictor.predict_coin_condition(path_avers, path_revers)
            
            # Добавляем URL для отображения загруженных изображений
            result["image_avers"] = url_for("static", filename="uploads/" + filename_avers)
            result["image_revers"] = url_for("static", filename="uploads/" + filename_revers)
            
            return render_template("result.html", result=result)
        else:
            return "Ошибка: неподдерживаемый формат файла."
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

