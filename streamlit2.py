import base64
import logging
import uuid
from io import BytesIO
from pathlib import Path
from cachetools import TTLCache
import chess
import cv2
import numpy as np
import onnxruntime as ort
import pymupdf
from flask import Flask, jsonify, render_template, request
from PIL import Image

piece_class = ["k", "q", "r", "b", "n", "p", "K", "Q", "R", "B", "N", "P", "empty"]

ONNX_MODEL_PATH = Path(__file__).with_name("model.onnx")


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def create_inference_session(model_path: Path) -> ort.InferenceSession:
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model file not found: {model_path}")
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def preprocess_square(square: Image.Image) -> np.ndarray:
    array = np.asarray(square, dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))


def infer_square_classes(squares: list[Image.Image], session: ort.InferenceSession) -> list[int]:
    batch = np.stack([preprocess_square(square) for square in squares], axis=0).astype(np.float32)
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: batch})[0]
    return np.argmax(logits, axis=1).astype(int).tolist()


def build_fen(predicted_classes: list[int]) -> str:
    rows = []
    for y in range(8):
        row = ""
        empty_count = 0
        for x in range(8):
            predicted_class = predicted_classes[x + y * 8]
            if predicted_class == 12:
                empty_count += 1
            else:
                if empty_count > 0:
                    row += str(empty_count)
                    empty_count = 0
                row += piece_class[predicted_class]
        if empty_count > 0:
            row += str(empty_count)
        rows.append(row)
    return "/".join(rows)


def order_points(pts):
    pts = np.array(pts, dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    ordered = np.zeros((4, 2), dtype="float32")

    ordered[0] = pts[np.argmin(s)]  # top-left
    ordered[2] = pts[np.argmax(s)]  # bottom-right
    ordered[1] = pts[np.argmin(diff)]  # top-right
    ordered[3] = pts[np.argmax(diff)]  # bottom-left

    return ordered


def board_crop(image):
    image = np.array(image)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.equalizeHist(blur)

    thre4 = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )

    contours, heir = cv2.findContours(
        thre4,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    cropped = []

    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        rect = cv2.approxPolyDP(cnt, epsilon, True)

        area = cv2.contourArea(rect)

        if len(rect) == 4 and area > 500 :
            src = order_points(rect.reshape(4, 2))

            dst = np.float32([
                [0, 0],
                [799, 0],
                [799, 799],
                [0, 799],
            ])

            M = cv2.getPerspectiveTransform(src, dst)

            warped = cv2.warpPerspective(image, M, (800, 800))

            cropped.append(warped)

    return cropped


def square_crop(cropped):
    squares = []
    cropped = Image.fromarray(cropped)

    sw = 100
    sh = 100
    for y in range(8):
        for x in range(8):
            square = cropped.crop((x * sw, y * sh, (x + 1) * sw, (y + 1) * sh))
            squares.append(square)
    return squares


def piece_at(piece_placement, square):
    board = {}

    ranks = piece_placement.split('/')

    for rank_index, rank in enumerate(ranks):
        file_index = 0

        for ch in rank:
            if ch.isdigit():
                file_index += int(ch)
            else:
                file = chr(ord('a') + file_index)
                board[f"{file}{8 - rank_index}"] = ch
                file_index += 1

    return board.get(square)


def extract_fens_from_page(page: pymupdf.Page, session: ort.InferenceSession) -> tuple[list[str], list[str]]:
    pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
    img_data = BytesIO(pix.tobytes())
    image = Image.open(img_data).convert("RGB")
    cropped = board_crop(image)
    image.close()
    del image
    fent = []
    ext_field = []

    if len(cropped) == 0:
        logger.info("No board found in page")
        return fent, ext_field

    for board_image in cropped:
        squares = square_crop(board_image)
        predicted_classes = infer_square_classes(squares, session)
        fen = build_fen(predicted_classes)

        board = chess.Board(fen + " w - - 0 1")
        bord = chess.Board(fen + " b - - 0 1")

        if board.is_valid() or bord.is_valid():
            fent.append(fen)
            c = ""
            if piece_at(fen, "e1") == "K":
                if piece_at(fen, "h1") == "R":
                    c += "K"
                if piece_at(fen, "a1") == "R":
                    c += "Q"
            if piece_at(fen, "e8") == "k":
                if piece_at(fen, "h8") == "r":
                    c += "k"
                if piece_at(fen, "a8") == "r":
                    c += "q"
            if c == "":
                c = "-"
            c += "_-_0_1"
            ext_field.append(c)

    return fent, ext_field


app = Flask(__name__)

try:
    INFERENCE_SESSION = create_inference_session(ONNX_MODEL_PATH)
except Exception as exc:
    logger.exception("Failed to initialize ONNX Runtime session")
    INFERENCE_SESSION = None


@app.route("/")
def home():
    return render_template("new1.html")

pdf_store=TTLCache(maxsize=100,ttl=1800)


@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    if INFERENCE_SESSION is None:
        return jsonify(error="Model initialization failed"), 500

    data = request.json
    pdf_base64 = data["pdf"]

    pdf_bytes = base64.b64decode(pdf_base64)
    pdf_id = str(uuid.uuid4())
    pdf_store[pdf_id] = pdf_bytes

    fent = []
    ext_field = []

    docim = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = docim.load_page(0)
        fent, ext_field = extract_fens_from_page(page, INFERENCE_SESSION)
    finally:
        docim.close()

    if len(fent) == 0:
        return jsonify(fen=[], pdf_id=pdf_id, ext_field=ext_field)
    return jsonify(fen=fent, pdf_id=pdf_id, ext_field=ext_field)


@app.route("/page_count", methods=["POST"])
def page_count():
    if INFERENCE_SESSION is None:
        return jsonify(error="Model initialization failed"), 500

    data = request.json
    requested_page = data["page_count"]
    pdf_id = data["pdf_id"]
    pdf_bytes = pdf_store.get(pdf_id)

    if pdf_bytes is None:
        return jsonify(error="Unknown pdf_id"), 400

    fent = []
    ext_field = []

    docim = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = docim.load_page(requested_page - 1)
        fent, ext_field = extract_fens_from_page(page, INFERENCE_SESSION)
    finally:
        docim.close()

    if len(fent) == 0:
        return jsonify(fen=[], pdf_id=pdf_id, ext_field=ext_field)
    return jsonify(fen=fent, pdf_id=pdf_id, ext_field=ext_field)


if __name__ == "__main__":
    app.run(debug=False)
