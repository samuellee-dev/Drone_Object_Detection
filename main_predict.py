import cv2
import time
from ultralytics import YOLO

# 1. 모델 및 비디오 로드
model = YOLO(r'C:\Users\samuel\Desktop\object_detection\best_person.pt')
#video_path = r"C:\Users\samuel\Desktop\object_detection\output1_960x540_20fps.MP4"  
video_path = r"C:\Users\samuel\Desktop\object_detection\output2_960x540_20fps.MP4"  
#video_path = r"C:\Users\samuel\Desktop\object_detection\output3_960x540_20fps.MP4"  

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("영상을 열 수 없습니다.")
    exit()

print("마스킹과 동기식 탐지를 적용하여 실행합니다.")

while cap.isOpened():
    start_time = time.time()
    
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break
    
    # [마스킹 처리] AI가 UI를 사람/차량으로 오인식하는 것 방지
    detect_frame = frame.copy()
    detect_frame[0:64, :] = 0       # 상단 UI 영역 마스킹
    detect_frame[485:, :] = 0       # 하단 UI 영역 마스킹
    
    # [동기식 처리] 마스킹된 프레임을 즉시 AI에 넣어 분석 (밀림 제로)
    results = model.predict(detect_frame, conf=0.3, imgsz=640, verbose=False)
    
    # 분석된 결과(박스)를 검은 띠가 없는 깨끗한 원본 frame 위에 시각화
    annotated_frame = results[0].plot(img=frame, font_size=0.3, line_width=1)
    
    # 실제 처리 속도(FPS) 계산
    inference_time = time.time() - start_time
    actual_fps = 1.0 / inference_time if inference_time > 0 else 0

    # 화면에 FPS 표시
    cv2.putText(annotated_frame, f"FPS: {actual_fps:.2f} (Sync + Masking)", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Drone Detection (Sync)", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
