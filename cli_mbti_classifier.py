from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 저장된 모델 경로 설정
saved_path = "C:/project/MBTI_T_F_classifier/classifier_model"


# 모델과 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(saved_path)
model = AutoModelForSequenceClassification.from_pretrained(saved_path)



def classify_sentence(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class = logits.argmax(dim=-1).item()
        confidence = probabilities[0][predicted_class].item() * 100
    return predicted_class, confidence

def main():
    print("MBTI T/F 분류기")
    print("문장을 입력하세요 (종료: exit):")
    while True:
        text = input("입력: ")
        if text.lower() == "exit":
            print("프로그램 종료")
            break
        predicted_class, confidence = classify_sentence(text)
        labels = {0: "T", 1: "F", 2: "NO"}
        print(f"결과: {labels[predicted_class]} ({confidence:.2f}%)")

if __name__ == "__main__":
    main()
