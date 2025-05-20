import cv2
from ultralytics import YOLO
from datetime import datetime
from pymongo import MongoClient
import time
import cloudinary
import cloudinary.uploader

cloudinary.config(
  cloud_name="dsuidrqfg",
  api_key="635296799295462",
  api_secret="LA7D_cXKlpQ4ARXbfyS5rWAx6I4"
)

# ✅ Connect to MongoDB (update with your real connection string)
client = MongoClient("mongodb+srv://pythoncoding0:DOfy1SA2zYVzgTHi@cluster0.ywvtrbj.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["cat_detector"]
collection = db["detections"]

# ✅ Load your trained model
model = YOLO("datasetcombined/content/runs/detect/cat_detector_merged/weights/best.pt")

# ✅ RTSP stream from CP Plus
rtsp_url = "rtsp://aryan:admin@123@192.168.0.13/cam/realmonitor?channel=3&subtype=0"


# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

# Define video writer (set up only after getting frame size)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None  # Will initialize later when first frame is grabbed

# Cooldown time to avoid multiple logs
last_logged_time = 0
cooldown_seconds = 10  # seconds

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Initialize video writer once frame size is known
        if out is None:
            height, width = frame.shape[:2]
            out = cv2.VideoWriter("output.avi", fourcc, 20.0, (width, height))

        results = model(frame, conf=0.30)
        annotated_frame = results[0].plot()

        # Write the annotated frame to the output video
        out.write(annotated_frame)

        # Check for cat detection
        for box in results[0].boxes:
            cls = int(box.cls[0].item())
            if results[0].names[cls] == "cat":
                    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    filename = f"cat_{current_time}.jpg"
                    cv2.imwrite(filename, frame)

                    # Upload to Cloudinary
                    upload_result = cloudinary.uploader.upload(filename, folder="cat_detections")
                    image_url = upload_result['secure_url']

                    # Log to MongoDB with image_url
                    db.collection.insert_one({
                        "timestamp": datetime.now(),
                        "confidence": float(box.conf.item()),
                        "location": "Camera 1",
                        "imageUrl": image_url
                    })
                    last_logged_time = time.time()

        # 🚫 REMOVE: Don't display the window
        # cv2.imshow("Cat Detector", annotated_frame)

        # Optional: still check for 'q' key if running in interactive window (safe to leave in)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nInterrupted by user. Saving and exiting...")

finally:
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
