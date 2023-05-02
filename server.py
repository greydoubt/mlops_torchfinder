import os
import zipfile

from flask import Flask, request, render_template, send_file
from werkzeug.utils import secure_filename

from deep_learning.infer import infer_path_on_image
from utils.viz import draw_path_on_image, export_image_with_path

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'zip'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_files():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        if filename.endswith('.zip'):
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(app.config['UPLOAD_FOLDER'])
            os.remove(filepath)
        return render_template('upload_success.html', filename=filename)
    return render_template('upload_failure.html')


@app.route('/results/<filename>')
def results(filename):
    original_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    results_image_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
    if not os.path.isfile(results_image_path):
        infer_path_on_image(original_image_path, results_image_path)
        draw_path_on_image(original_image_path, results_image_path, results_image_path)
        export_image_with_path(original_image_path, results_image_path, results_image_path)
    return render_template('results.html', filename=filename)


@app.route('/download/<filename>')
def download(filename):
    results_image_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
    return send_file(results_image_path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
