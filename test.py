from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 저장된 모델 경로
saved_path = "./mbti_t_f_classifier_04"

# 모델과 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(saved_path)
model = AutoModelForSequenceClassification.from_pretrained(saved_path)

# 새로운 텍스트 예측
text ="이 글을 지우고 테스트 하고 싶은 텍스트를 입력하세요"
inputs = tokenizer(text, return_tensors="pt", max_length=128, padding="max_length", truncation=True)

# 모델 평가 모드 전환
model.eval()

# 추론
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(dim=-1).item()

print(f"Predicted class: {predicted_class}")

# 0 = T, 1 = F, 2 = 무관
