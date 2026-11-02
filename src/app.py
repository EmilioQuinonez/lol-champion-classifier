"""Interfaz web para el clasificador de campeones.

Uso:
    python src/app.py              # abre http://localhost:8000
    python src/app.py --port 8080

Solo usa la biblioteca estándar (http.server) más TensorFlow/Keras:
la página sube la imagen y el servidor responde con el top 5 del modelo.

El modelo se entrenó con splash arts horizontales. Las imágenes con una proporción
parecida reciben exactamente el preprocesamiento del entrenamiento; las demás
(verticales o cuadradas) se colocan completas con bordes negros para no deformarlas.
"""
import argparse
import io
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "lol_champion_classifier.keras"
METADATA_PATH = ROOT / "models" / "metadata.json"
SPLASH_ASPECT = 1215 / 717   # proporción de las splash arts del dataset
ASPECT_TOLERANCE = 0.15

WEB = Path(__file__).resolve().parent / "web"
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
TOP = 5
MAX_BYTES = 20 * 1024 * 1024


def load_image(file, meta):
    """Replica el preprocesamiento del pipeline: data_setup reduce a `resize_height`
    con LANCZOS y el cargador de Keras estira la imagen a `input_size`."""
    import tensorflow as tf

    with Image.open(file) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        if meta.get("grayscale"):
            im = im.convert("L").convert("RGB")   # el modelo final se entrenó en escala de grises
        w, h = im.size
        height = meta["resize_height"]
        im = im.resize((round(w * height / h), height), Image.LANCZOS)
        img = np.asarray(im, dtype=np.float32)

    size = tuple(meta["input_size"])
    if abs(w / h / SPLASH_ASPECT - 1) <= ASPECT_TOLERANCE:
        img = tf.image.resize(img, size)            # igual que en el entrenamiento
    else:
        img = tf.image.resize_with_pad(img, *size)  # imagen completa, sin deformar
    return np.asarray(img)


class Handler(BaseHTTPRequestHandler):
    model = None
    meta = None

    def send(self, code, body, content_type):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, code, data):
        self.send(code, json.dumps(data).encode(), "application/json")

    def do_GET(self):
        if self.path in STATIC:
            name, content_type = STATIC[self.path]
            self.send(200, (WEB / name).read_bytes(), content_type)
        elif self.path == "/info":
            h, w = self.meta["input_size"]
            self.send_json(200, {"version": self.meta["version"],
                                 "classes": len(self.meta["class_names"]),
                                 "input": f"{w}x{h}"})
        else:
            self.send(404, b"No encontrado", "text/plain")

    def do_POST(self):
        if self.path != "/predict":
            return self.send(404, b"No encontrado", "text/plain")
        length = int(self.headers.get("Content-Length", 0))
        if not 0 < length <= MAX_BYTES:
            return self.send_json(400, {"error": "Imagen vacía o mayor a 20 MB"})
        try:
            img = load_image(io.BytesIO(self.rfile.read(length)), self.meta)
        except Exception:
            return self.send_json(400, {"error": "No se pudo leer la imagen"})

        probs = self.model.predict(img[None], verbose=0)[0]
        order = np.argsort(probs)[::-1][:TOP]
        names = self.meta["class_names"]
        self.send_json(200, {"top": [{"champion": names[i], "prob": float(probs[i])}
                                     for i in order]})

    def log_message(self, fmt, *args):
        print(f"{self.address_string()} - {fmt % args}")


def main():
    parser = argparse.ArgumentParser(description="Interfaz web del clasificador")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="no abrir el navegador")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit(f"No se encontró {MODEL_PATH}. Ejecuta primero notebooks/training.ipynb.")

    import keras

    Handler.model = keras.models.load_model(MODEL_PATH)
    Handler.meta = json.loads(METADATA_PATH.read_text())

    url = f"http://localhost:{args.port}"
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Modelo {Handler.meta['version']} listo en {url} (Ctrl+C para salir)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor detenido")


if __name__ == "__main__":
    main()
