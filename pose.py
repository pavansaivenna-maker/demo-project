import cv2
import mediapipe as mp
import time

model_path = "C:/Users/Pavansaivenna/Downloads/pose_model.task"

options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
    running_mode=mp.tasks.vision.RunningMode.IMAGE)

# Fixed labels (camera mirrors left/right)
LABELS = {
    0:"Nose",
    11:"L.Shoulder", 12:"R.Shoulder",
    13:"L.Elbow",    14:"R.Elbow",
    15:"L.Wrist",    16:"R.Wrist",
    23:"L.Hip",      24:"R.Hip",
    25:"L.Knee",     26:"R.Knee",
    27:"L.Ankle",    28:"R.Ankle"
}

BODY = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24)]
LEGS = [(23,25),(24,26),(25,27),(26,28)]

cap = cv2.VideoCapture(0)
prev_time = time.time()

with mp.tasks.vision.PoseLandmarker.create_from_options(options) as lm:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = lm.detect(mp_img)

        detected = False
        if result.pose_landmarks:
            detected = True
            pts = result.pose_landmarks[0]

            # Body lines - white
            for a,b in BODY:
                p1 = (int(pts[a].x*w), int(pts[a].y*h))
                p2 = (int(pts[b].x*w), int(pts[b].y*h))
                cv2.line(frame, p1, p2, (255,255,255), 2)

            # Leg lines - white
            for a,b in LEGS:
                p1 = (int(pts[a].x*w), int(pts[a].y*h))
                p2 = (int(pts[b].x*w), int(pts[b].y*h))
                cv2.line(frame, p1, p2, (255,255,255), 2)

            # Dots and labels
            for i, pt in enumerate(pts):
                x, y = int(pt.x*w), int(pt.y*h)
                # Orange dot for right side, blue for left
                color = (0,140,255) if i % 2 == 0 else (255,100,0)
                cv2.circle(frame, (x,y), 7, color, -1)
                cv2.circle(frame, (x,y), 3, (255,255,255), -1)
                if i in LABELS:
                    cv2.putText(frame, LABELS[i], (x+6, y+4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # FPS
        curr_time = time.time()
        fps = int(1/(curr_time - prev_time + 0.001))
        prev_time = curr_time

        # Top title bar
        cv2.rectangle(frame, (0,0), (w,40), (0,0,0), -1)
        status = "Pose Detected" if detected else "No Pose Detected"
        title = f"MediaPipe Pose Detection  |  {status}"
        cv2.putText(frame, title, (10,27),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"FPS: {fps}", (w-90,27),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)

        # Bottom text
        cv2.putText(frame, "Pose Detection Running... Press 'Q' to quit.",
            (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        cv2.imshow("Pose Detection - AI/ML Intern Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
