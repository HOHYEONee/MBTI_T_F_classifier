# MBTI_T_F_classifier

<br><br><br>

### MBTI T F Classifier program

<br><br><br>

***1. Project Overview and Theme***

A program that classifies a user's input sentence as being closer to the MBTI T (Thinking) type or F (Feeling) type based on Hugging Face.
It determines whether a sentence is closer to the Thinking (T) type, Feeling (F) type, or an unrelated sentence that cannot be classified as either T or F in the MBTI framework.

<br>

***2. Our Term Project Progress***

     (1) Collecting Data
          - data : It includes the sentence data file used and an explanation of how the data was collected. 
     (2) Training the Classification Model
          - classifier_model / model_generation_code.py / test.py / test_cpuVersion.py
     (3) Developing the CLI Program
          - cli_mbti_classifier.py

<br>

***3. Used Packages and Versions***
- Transformers (version: 4.46.3)
- Torch (version: 2.5.1+cu121 or 2.5.1+cpu)

<br>

***4. Execution Instruction***  

(1) Install `transformers` and `pytorch`.

(2) Download the files inside the `classifier_model` folder. (These are uploaded in our repository.)


     classifier_model contents:
        - `config.json`
        - `model.safetensors` 
           -> Download it from the link and place it in the folder.
        - `special_tokens_map.json`
        - `tokenizer.json`
        - `tokenizer_config.json`
        - `vocab.txt`

(3) Run the cli_mbti_classifier.py.

(4) Set the path of `classifier_model` in `saved_path`.

(5) Enter the text you want to test.

(6) Check the results after execution:

     The result shows which type it is associated with and the corresponding percentage.
     - result: F (xx.xx%)
     - result: T (xx.xx%)
     - result: NO (xx.xx%)

<br>

***5. Examples and Execution Screenchots***

<br>

***6. References***
- [인공지능 친해지기] Hugging Face - (1) 코딩 몰라도 가능한 AI 다루기 - 텍스트분류(감성분석)과 개체명 인식
(https://youtu.be/HwTaVeRzy5M?si=bRGrH2FPgXme4FqP)
- [인공지능 친해지기] Hugging Face (11) 토큰화(Tokenization)
(https://youtu.be/MCok4wCX29M?si=QeNw0qX5Nkdj1E56)
