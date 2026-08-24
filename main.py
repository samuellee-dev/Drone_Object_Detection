"""
ONNX 모델 기반 실시간 객체 탐지 (온디바이스 최종본)

PC에서 학습한 PyTorch 가중치를 convert.py로 ONNX 변환한 뒤,
Jetson 환경에서 ONNX Runtime으로 추론하기 위한 스크립트.
"""

import cv2
import time
from ultralytics import YOLO

# ─── 설정 ────────────────────────────────────────────────
MODEL_PATH = "best.onnx"          # convert.py 결과물
#VIDEO_PATH = "output1_960x540_20fps.MP4"
VIDEO_PATH = "output2_960x540_20fps.MP4"
#VIDEO_PATH = "output3_960x540_20fps.MP4"

CONF = 0.3          # 신뢰도 임계값. 낮추면 소형 객체 탐지율↑, 오탐지도 함께↑
IMGSZ = 640         # convert.py에서 dynamic=False로 고정했으므로 반드시 640
MASK_TOP = 64       # 상단 UI 마스킹 높이 (960x540 기준)
MASK_BOTTOM = 485   # 하단 UI 마스킹 시작 y좌표 (960x540 기준)
# ─────────────────────────────────────────────────────────

# YOLO()는 확장자를 보고 백엔드를 자동 선택한다.
# .onnx면 PyTorch가 아닌 ONNX Runtime으로 로드됨 → 구형 Jetson에서 PyTorch 연산 제약 회피
model = YOLO(MODEL_PATH, task="detect")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("영상을 열 수 없습니다.")
    exit()

print("ONNX Runtime 추론을 시작합니다. 'q'를 누르면 종료됩니다.")

while cap.isOpened():
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    # [마스킹] 화면에 고정된 UI를 사람/차량으로 오인식하는 문제 차단.
    # 원본(frame)은 그대로 두고 복사본에만 검은 띠를 씌워 추론에 넣는다.
    detect_frame = frame.copy()
    detect_frame[0:MASK_TOP, :] = 0
    detect_frame[MASK_BOTTOM:, :] = 0

    # persist=True: 이전 프레임의 추적 ID를 유지해 프레임 간 연속성 부여.
    # verbose=False: 콘솔 로그 출력을 끄면 그만큼 오버헤드가 줄어든다.
    results = model.track(
        detect_frame, conf=CONF, persist=True, imgsz=IMGSZ, verbose=False
    )

    # img=frame: 탐지는 마스킹본으로, 시각화는 깨끗한 원본 위에 그린다.
    annotated_frame = results[0].plot(img=frame, font_size=0.3, line_width=1)

    # 프레임 읽기부터 시각화까지 전 구간을 측정 → 체감 성능과 일치하는 FPS
    inference_time = time.time() - start_time
    actual_fps = 1.0 / inference_time if inference_time > 0 else 0

    cv2.putText(
        annotated_frame,
        f"FPS: {actual_fps:.2f} (ONNX)",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Drone Detection (ONNX)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("측정이 종료되었습니다.")