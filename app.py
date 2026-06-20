from flask import Flask, jsonify, send_from_directory, Response
import serial, serial.tools.list_ports, threading, json, time, sys, platform
from datetime import datetime
import cv2

# Set console output to utf-8 to avoid Windows console UnicodeEncodeError
try:
    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

app = Flask(__name__, static_folder='static')

state = {
    "temp":      None,
    "humidity":  None,
    "mq2":       None,
    "mq2_raw":   None,
    "ai_label":  "scanning",
    "ai_conf":   0,
    "alerts":    [],
    "images_today": 0,
    "updated":   None,
    "pico_connected": False,
    "camera_connected": False
}

camera_frame = None
camera_lock = threading.Lock()

def camera_reader():
    global camera_frame
    print("🎥 Starting camera reader thread...")
    
    # Initialize YOLO model
    yolo_model = None
    try:
        from ultralytics import YOLO
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Try finding the model in 'models/' folder, then fall back to root folder
        model_path = os.path.join(base_dir, "models", "best_openvino_model")
        if not os.path.exists(model_path):
            model_path = os.path.join(base_dir, "best_openvino_model")
            
        print(f"🤖 Loading YOLOv8 Chili Disease Detection OpenVINO model ({model_path})...")
        yolo_model = YOLO(model_path, task="detect")
        print("✅ YOLO OpenVINO model loaded successfully!")
    except Exception as model_err:
        print(f"⚠️ Failed to load YOLO model: {model_err}")
    
    cap = None
    
    while True:
        if cap is None or not cap.isOpened():
            state["camera_connected"] = False
            # Attempt to connect to camera
            # Try indexes 0, 1, 2
            for index in [0, 1, 2]:
                print(f"🎥 Attempting to open camera index {index}...")
                cap = cv2.VideoCapture(index)
                if cap.isOpened():
                    state["camera_connected"] = True
                    print(f"✅ Camera successfully opened at index {index}")
                    break
                else:
                    cap.release()
                    cap = None
            
            if cap is None:
                print("⚠️ No USB camera found, retrying in 5 seconds...")
                time.sleep(5)
                continue

        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to read frame from camera, releasing and retrying...")
            cap.release()
            cap = None
            state["camera_connected"] = False
            time.sleep(2)
            continue

        # Run YOLO inference
        if yolo_model is not None:
            try:
                results = yolo_model(frame, imgsz=320, verbose=False)
                if results and len(results) > 0:
                    result = results[0]
                    # Draw bounding boxes and labels on frame
                    frame = result.plot()
                    
                    if len(result.boxes) > 0:
                        # Find object with highest confidence
                        best_box = None
                        best_conf = -1.0
                        for box in result.boxes:
                            conf = float(box.conf[0])
                            if conf > best_conf:
                                best_conf = conf
                                best_box = box
                        
                        if best_box is not None:
                            class_id = int(best_box.cls[0])
                            label = result.names[class_id]
                            state["ai_label"] = label
                            state["ai_conf"] = int(best_conf * 100)
                            
                            # If disease detected and conf > 50%, add alert
                            # e.g., anything that isn't 'healthy'
                            if "healthy" not in label.lower() and best_conf > 0.5:
                                alert_exists = False
                                for alert in state["alerts"]:
                                    if alert["title"] == "Crop disease detected" and alert["detail"].startswith(label.capitalize()):
                                        alert_exists = True
                                        break
                                if not alert_exists:
                                    add_alert("alert", "Crop disease detected", 
                                              f"{label.capitalize()} spotted by AI camera ({int(best_conf * 100)}% confidence)")
                                    state["images_today"] += 1
                        else:
                            state["ai_label"] = "healthy"
                            state["ai_conf"] = 100
                    else:
                        state["ai_label"] = "healthy"
                        state["ai_conf"] = 100
            except Exception as inference_err:
                print(f"⚠️ YOLO inference error: {inference_err}")
                state["ai_label"] = "inference error"

        ret_jpeg, jpeg = cv2.imencode('.jpg', frame)
        if ret_jpeg:
            with camera_lock:
                camera_frame = jpeg.tobytes()
        
        time.sleep(0.05)

def gen_frames():
    global camera_frame
    while True:
        if not state.get("camera_connected"):
            break
        with camera_lock:
            frame = camera_frame
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        time.sleep(0.05)

def add_alert(level, title, detail):
    state["alerts"].insert(0, {
        "level":  level,
        "title":  title,
        "detail": detail,
        "time":   datetime.now().strftime("%H:%M")
    })
    state["alerts"] = state["alerts"][:20]

def check_thresholds(d):
    mq2  = d.get("mq2")
    temp = d.get("temp")
    hum  = d.get("humidity")

    if mq2 is not None and mq2 > 400:
        add_alert("alert", "Gas/Smoke Detected",
                  f"MQ2: {mq2} ppm — high risk of smoke/fire")

    if temp and hum and temp > 35 and hum > 85:
        add_alert("warning", "Heat stress risk",
                  f"DHT11: {temp}°C, {hum}% RH")

def find_pico_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        desc = p.description.lower()
        if any(kw in desc for kw in ["pico", "usb serial", "ch340", "cp210", "acm"]):
            return p.device
    if platform.system() == "Windows":
        for p in ports:
            if "bluetooth" not in p.description.lower():
                return p.device
        return "COM3"
    else:
        return "/dev/ttyACM0"

def pico_reader():
    while True:
        port = find_pico_port()
        try:
            with serial.Serial(port, 115200, timeout=3) as ser:
                print(f"✅ Pico connected on {port}")
                state["pico_connected"] = True
                while True:
                    line = ser.readline().decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        # Map Pi Pico keys to state/UI expectations
                        if "gas_raw" in d:
                            d["mq2_raw"] = d["gas_raw"]
                        if "gas_voltage" in d:
                            # Map gas_voltage (0 to 3.3V) to estimated ppm (0 to ~1000)
                            d["mq2"] = int(d["gas_voltage"] * 300)
                        
                        state.update(d)
                        state["updated"] = datetime.now().strftime("%H:%M:%S")
                        check_thresholds(state)
                    except json.JSONDecodeError:
                        print("Bad line:", line)
        except Exception as e:
            state["pico_connected"] = False
            print(f"⚠️  Serial error: {e} — retrying in 3s")
            time.sleep(3)

@app.route("/api/data")
def api_data():
    return jsonify(state)

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/video_feed")
def video_feed():
    if not state.get("camera_connected"):
        return "Camera not connected", 503
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/snapshot")
def api_snapshot():
    global camera_frame
    with camera_lock:
        frame = camera_frame
    if frame is None:
        return jsonify({"error": "No frame captured yet or camera disconnected"}), 503
    return Response(frame, mimetype="image/jpeg")

@app.errorhandler(404)
def handle_404(e):
    return send_from_directory("static", "index.html")

if __name__ == "__main__":
    import os
    threading.Thread(target=pico_reader, daemon=True).start()
    threading.Thread(target=camera_reader, daemon=True).start()
    port = int(os.environ.get("PORT", 5000))
    print(f"🌿 Dashboard → http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)