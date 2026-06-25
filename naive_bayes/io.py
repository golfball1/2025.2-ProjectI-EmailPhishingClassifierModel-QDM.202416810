import pandas as pd

import joblib

best_model = joblib.load("best_model.pkl")
processor = joblib.load("processor.pkl")

stop = False    

while True:
    if stop:
        break
    
    print("===== EMAIL PHISHING DETECTOR =====")
    date = input("Date: ")
    sender = input("Sender: ")
    receiver = input("Receiver: ")
    subject = input("Subject: ")
    body_text = input("Body: ")

    while True:
        has_url = input("Contains URL? (y/n): ").strip().lower()
        if has_url == "y":
            urls = 1 
            break
        elif has_url == "n":
            urls = 0
            break
        

    input_df = pd.DataFrame({
        "subject": [subject],
        "sender": [sender],
        "receiver": [receiver],
        "date": [date],
        "body": [body_text],
        "urls": [urls]
    })

    text_cols = ["subject", "sender", "body"]
    input_df['body'] = input_df[text_cols].fillna('').agg(' '.join, axis=1)

    X_input = processor.transform(input_df)


    spam_prob = best_model.predict_proba(X_input)[0][1]

    prediction = "SPAM" if spam_prob >= 0.55 else "HAM"


    print("\n===== RESULT =====")
    print(f"Prediction: {prediction}")
    print(f"Spam Probability: {spam_prob * 100:.2f}%")

    
    while True:
        continuing = input("Continue? (y/n): ").strip().lower()
        if continuing == "y":
            stop = False
            break
        elif continuing == "n":
            stop = True
            break
        
#===== EMAIL PHISHING DETECTOR =====
#Date: 11/1/2001
#Sender: jfdks.burner.noreply@xyz
#Receiver: minh@gmail.com 
#Subject: Meet hot girls
#Body: Meet hot girls in your area at xxxvideos.com. Join now!
#Contains URL? (y/n): y

#===== EMAIL PHISHING DETECTOR =====
#Date: 11/1/2001
#Sender: hieu.th23014@sis.hust.edu.vn
#Receiver: minh@gmail.com
#Subject: Need help on the computer
#Body: Hey Minh, our computer is broken. Can you come check it out? Thanks
#Contains URL? (y/n): n
