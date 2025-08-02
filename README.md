# Connect-1.0
Real‑Time Sign Language Translation Model Documentation

**1. Introduction**
The Connect system is a real‑time sign language translation application designed to break communication barriers for the deaf community in Kenya’s public service offices. It captures hand gestures via webcam, processes frames through an image‑processing pipeline, classifies signs into text, and vocalizes the result using text‑to‑speech.

**2. Objectives**

* Provide instantaneous translation of Kenyan Sign Language (KSL) into spoken English or Swahili.
* Achieve latency below 200 ms per frame for seamless conversations.
* Enable easy deployment on standard desktops or edge devices using only CLI.

**3. Functional Requirements**

1. **Webcam Input**: Continuously capture video frames at ≥ 15 FPS.
2. **Preprocessing**: Normalize, crop, and augment frames to highlight hand regions.
3. **Inference**: Run a trained deep‑learning classifier (e.g., CNN or light‑weight Transformer) on each frame.
4. **Postprocessing**: Smooth predictions over a sliding window (pattern period ≈ 75 frames) to reduce noise.
5. **Text‑to‑Speech**: Convert final recognized phrase into synthesized speech.
6. **Web Interface**: Display live video, recognized text, and play audio.
7. **Port Forwarding**: Expose the web interface via an HTTP tunnel for remote access.

**4. Non‑Functional Requirements**

* **Performance**: End‑to‑end processing ≤ 200 ms/frame.
* **Reliability**: ≥ 95% recognition accuracy under varying lighting.
* **Scalability**: Support up to 30 concurrent users via a shared FastAPI server.
* **Maintainability**: Modular codebase with clear separation of concerns (data, model, API, UI).

**5. System Architecture**

```
[Webcam] → [Capture Module] → [Preprocessor (OpenCV, MediaPipe)] → [Inference Engine (TensorFlow/PyTorch)] → [Postprocessor] → [API Server (FastAPI + WebSocket)] → [Front‑end (HTML/JS)] → [Text‑to‑Speech]
```

* **Capture Module** (`sign.py`) handles frame grabs and initial buffering.
* **Preprocessor** uses MediaPipe Hands to detect and crop hand landmarks.
* **Inference Engine** loads a Keras/PyTorch model fine‑tuned on KSL dataset with CIFAR‑style augmentations.
* **Postprocessor** applies temporal smoothing (alpha=0.0005 proximity threshold, beta=0.0005 contraction threshold).
* **API Server** (FastAPI) streams predictions via WebSocket endpoints and serves the web UI.
* **Front‑end** displays real‑time video overlay and recognized text; plays speech via browser TTS or gTTS API.

**6. Data Flow**

1. Frame captured at time *t*.
2. Hand region isolated; resized to 224×224.
3. Model outputs softmax probabilities over N classes.
4. Sliding window of last P=75 frames aggregates votes.
5. If the dominant class probability change exceeds thresholds, update recognized sign.
6. Emit text message over WebSocket.

**7. Component Details**

* **`sign.py`**: Main CLI script. Parses arguments (`--tp`, `--pattern-period`, etc.), instantiates modules, and runs the loop.
* **`capture.py`**: Wraps OpenCV’s `VideoCapture` for threaded frame reads.
* **`preprocess.py`**: Contains functions for color normalization, background subtraction, and MediaPipe landmark extraction.
* **`model.py`**: Loads the trained model checkpoint, handles batching and device placement (CPU/GPU).
* **`postprocess.py`**: Implements smoothing logic using configurable `Alpha` and `Beta` parameters.
* **`api_server.py`**: FastAPI app with endpoints:

  * `GET /` → Serves `index.html`.
  * `WebSocket /ws/predictions` → Streams JSON payloads `{timestamp, label, confidence}`.
* **`frontend/`**: Static assets (HTML, CSS, JS). Uses WebSocket client to render predictions and play audio.

**8. Deployment**

1. Install dependencies:

   ```bash
   pip install -r requirements.txt  # OpenCV, mediapipe, fastapi, uvicorn, tensorflow
   ```
2. Launch server:

   ```bash
   uvicorn api_server:app --host 0.0.0.0 --port 8000
   ```
3. (Optional) Expose via SSH tunnel or Cloudflare Tunnel for remote access.
4. Open `http://<server-ip>:8000` or forwarded URL in browser.

**9. Usage Example**

```bash
# Run on local machine
python sign.py --pattern-period 75 --alpha 0.0005 --beta 0.0005
# Access UI at http://localhost:8000
```

* Speak via sign gestures; recognized text and audio feedback appear within \~150 ms.

**10. Future Work**

* **Phrase‑Level Translation**: Extend from single‑sign classification to multi‑word sentence recognition using sequence models.
* **Cross‑Language**: Add support for Swahili and other Kenyan languages.
* **Mobile App**: Package as Android/iOS app leveraging TensorFlow Lite for on‑device inference.
* **User‑Customization**: Allow end‑users to fine‑tune the model on personal variation of signs.

**11. References**

* MediaPipe Hands (Google)
* FastAPI documentation
* TensorFlow/Keras model zoo
