from insightface.app import FaceAnalysis

PROVIDERS = {
    "cpu":    ["CPUExecutionProvider"],
    "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
    "cuda":   ["CUDAExecutionProvider", "CPUExecutionProvider"],
}


def make_app(device: str, det_size):
    app = FaceAnalysis(name="buffalo_l", providers=PROVIDERS[device])
    app.prepare(ctx_id=0, det_size=tuple(det_size))
    return app


def detect_faces(app, bgr, min_det_score: float):
    faces = app.get(bgr)
    return [f for f in faces if f.det_score >= min_det_score]


def prominent_face(faces):
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
