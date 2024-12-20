from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

# Configuración inicial de Flask
app = Flask(__name__)

# Carpeta para guardar imágenes temporalmente
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Parámetros Configurables
HSV_LOWER_BLUE = np.array([90, 100, 50])  # Rango inferior del azul (ajustable)
HSV_UPPER_BLUE = np.array([130, 255, 255])  # Rango superior del azul
MIN_LINE_LENGTH = 100  # Longitud mínima de líneas detectadas
MAX_LINE_GAP = 50  # Máxima separación entre segmentos
BLUE_AREA_THRESHOLD = 500  # Área mínima azul detectada para continuar


# Función principal para detectar líneas azules
def detect_blue_line(image_path):
    try:
        # Cargar la imagen
        image = cv2.imread(image_path)
        if image is None:
            return {"message": "No se pudo cargar la imagen"}

        # Convertir a espacio de color HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Crear la máscara para detectar el color azul
        mask = cv2.inRange(hsv, HSV_LOWER_BLUE, HSV_UPPER_BLUE)

        # Guardar la máscara intermedia para depuración
        mask_path = os.path.join(app.config["UPLOAD_FOLDER"], "debug_mask.jpg")
        cv2.imwrite(mask_path, mask)

        # Verificar si hay suficiente área azul
        blue_area = cv2.countNonZero(mask)
        if blue_area < BLUE_AREA_THRESHOLD:
            return {
                "message": "No se detectó suficiente área azul",
                "mask_image": mask_path,
                "output_image": None
            }

        # Reducir ruido usando operaciones morfológicas
        kernel = np.ones((5, 5), np.uint8)
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)

        # Detectar bordes usando Canny
        edges = cv2.Canny(mask_cleaned, 50, 150)

        # Detectar líneas usando Transformada de Hough
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=100,
                                minLineLength=MIN_LINE_LENGTH, maxLineGap=MAX_LINE_GAP)

        # Si no se detectan líneas
        if lines is None:
            return {
                "message": "No se detectaron líneas azules rectas en la imagen",
                "mask_image": mask_path,
                "output_image": None
            }

        # Dibujar las líneas detectadas
        output_image = image.copy()
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Verde para las líneas detectadas

        # Guardar la imagen resultante
        output_path = os.path.join(app.config["UPLOAD_FOLDER"], "output_blue_lines.jpg")
        cv2.imwrite(output_path, output_image)

        return {
            "message": "Líneas azules rectas detectadas",
            "mask_image": mask_path,
            "output_image": output_path
        }

    except Exception as e:
        return {"message": "Error al procesar la imagen", "error": str(e)}


# Endpoint de la API para subir imágenes y detectar líneas
@app.route("/detect_blue_line", methods=["POST"])
def upload_image():
    try:
        # Validar si hay archivo en la solicitud
        if "image" not in request.files:
            return jsonify({"message": "No se encontró el archivo de imagen"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"message": "El nombre del archivo está vacío"}), 400

        # Guardar la imagen
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Procesar la imagen
        result = detect_blue_line(file_path)
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"message": "Error en el servidor", "error": str(e)}), 500


# Ejecutar la aplicación
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
