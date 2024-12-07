import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 저장된 모델 경로
saved_path = r"C:\Users\rhfor\Desktop\mbti_classfier"

# 모델과 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(saved_path)
model = AutoModelForSequenceClassification.from_pretrained(saved_path)

# 모델을 CPU로 강제 전환
device = torch.device("cpu")
model.to(device)

# 새로운 텍스트 예측 
text = "이 곳에 테스트하고 싶은 텍스트를 입력하세요"
inputs = tokenizer(text, return_tensors="pt", max_length=128, padding="max_length", truncation=True)

# 입력 데이터를 CPU로 이동
inputs = {key: value.to(device) for key, value in inputs.items()}

# 모델 평가 모드 전환
model.eval()

# 추론
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(dim=-1).item()

print(f"Predicted class: {predicted_class}")

# 0 = T, 1 = F, 2 = 무관
