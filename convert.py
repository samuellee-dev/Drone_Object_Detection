from ultralytics import YOLO

model = YOLO('best.pt')
model.export(format='onnx', imgsz=640, half=True, dynamic=False)

#imgsz 모델이 640*640 크기의 이미지를 처리할 수 있도록 모델의 입력이 고정됨.
#half 모델을 FP32 -> FP16으로 변환 모델의 크기가 줄고 FP16연산을 지원하는 GPU에서 성능 향상
#dynamic False으로 onnx 모델에 이미지가 고정됨, 최적화를 위해 고정
