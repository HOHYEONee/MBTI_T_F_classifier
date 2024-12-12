
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "klue/roberta-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
MBTI_T_F_classifier_04 = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

import pandas as pd

df = pd.read_csv('T_2.csv')

df['label'] = 0

print(df.head())
print(len(df))

df.to_csv("T_with_labels.csv", index=False)

import pandas as pd

df = pd.read_csv('F_2.csv')

df['label'] = 1

print(df.head())
print(len(df))

df.to_csv("F_with_labels.csv", index=False)

import pandas as pd

df = pd.read_csv('NO.csv')

df['label'] = 2

print(df.head())
print(len(df))

df.to_csv("NO_with_labels.csv", index=False)


import torch

print(torch.__version__)

print(torch.cuda.is_available())

import pandas as pd

df = pd.read_csv("T_with_labels.csv")
df = df.sample(frac=1).reset_index(drop=True)

test_data = df.tail(100)


train_data = df.iloc[:-100]


print("Train data size:", train_data.shape)
print("Test data size:", test_data.shape)


train_data.to_csv("T_with_labels_question.csv", index=False)


test_data.to_csv("T_with_labels_test.csv", index=False)

import pandas as pd

df = pd.read_csv("F_with_labels.csv")
df = df.sample(frac=1).reset_index(drop=True)

test_data = df.tail(100)


train_data = df.iloc[:-100]


print("Train data size:", train_data.shape)
print("Test data size:", test_data.shape)


train_data.to_csv("F_with_labels_question.csv", index=False)


test_data.to_csv("F_with_labels_test.csv", index=False)

import pandas as pd

df = pd.read_csv("NO_with_labels.csv")
df = df.sample(frac=1).reset_index(drop=True)

test_data = df.tail(100)


train_data = df.iloc[:700]


print("Train data size:", train_data.shape)
print("Test data size:", test_data.shape)


train_data.to_csv("NO_with_labels_question.csv", index=False)

test_data.to_csv("NO_with_labels_test.csv", index=False)

import pandas as pd

df_0 = pd.read_csv("T_with_labels_question.csv")  # label 0에 해당하는 CSV
df_1 = pd.read_csv("F_with_labels_question.csv")  # label 1에 해당하는 CSV
df_2 = pd.read_csv("NO_with_labels_question.csv")  # label 2에 해당하는 CSV

df = pd.concat([df_0, df_1, df_2], ignore_index=True)


print(df.head())
print(f"총 데이터 수: {len(df)}")


df.to_csv("mbtiDatasets.csv", index=False)

import pandas as pd

df_0 = pd.read_csv("T_with_labels_test.csv")  # label 0에 해당하는 CSV
df_1 = pd.read_csv("F_with_labels_test.csv")  # label 1에 해당하는 CSV
df_2 = pd.read_csv("NO_with_labels_test.csv")  # label 1에 해당하는 CSV

df = pd.concat([df_0, df_1, df_2], ignore_index=True)


print(df.head())
print(f"총 데이터 수: {len(df)}")


df.to_csv("mbtiDatatests.csv", index=False)

from torch.utils.data import Dataset
import torch

class MBTIDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
       
        row = self.dataframe.iloc[index]
        text, label = row['text'], row['label']

      
        inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

  
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        label = torch.tensor(label, dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label
        }
df = pd.read_csv("mbtiDatasets.csv")

train_dataset = MBTIDataset(df, tokenizer, max_len=128)

from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

import torch
from torch.optim import AdamW


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MBTI_T_F_classifier_04.to(device)


criterion = torch.nn.CrossEntropyLoss()
optimizer = AdamW(MBTI_T_F_classifier_04.parameters(), lr=5e-5)

num_epochs = 3

import pandas as pd
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    MBTI_T_F_classifier_04.train()  

    total_loss = 0
    for batch in train_loader:
    
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

 
        outputs = MBTI_T_F_classifier_04(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

     
        loss = outputs.loss
        total_loss += loss.item()

     
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

   
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

MBTI_T_F_classifier_04.save_pretrained("./mbti_t_f_classifier_04")
tokenizer.save_pretrained("./mbti_t_f_classifier_04")


df_test = pd.read_csv("mbtiDatatests.csv")  




tokenizer = AutoTokenizer.from_pretrained("./mbti_t_f_classifier_04")


test_dataset = MBTIDataset(df_test, tokenizer, max_len=128)


test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

from transformers import AutoModelForSequenceClassification


model = AutoModelForSequenceClassification.from_pretrained("./mbti_t_f_classifier_04")
model.to(device)  
model.eval()  

from sklearn.metrics import accuracy_score, classification_report

all_preds = []
all_labels = []


with torch.no_grad():  
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

       
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)  

        
        all_preds.extend(preds.cpu().numpy())  
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f"Accuracy: {accuracy:.4f}")

print(classification_report(all_labels, all_preds, target_names=["Class 0", "Class 1", "Class 2"]))