import numpy as np
import pandas as pd
from numpy.core.fromnumeric import reshape
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , precision_score , recall_score , f1_score
df = pd.read_csv("twitter_sentiment_data.csv")

df["Formatted"] = (
    df["message"].str.replace(r"^https?:\/\/.*[\r\n]*","",regex = True).replace(r"@[A-Za-z0-9_]+","",regex = True).replace(r"#[A-Za-z0-9_]+","",regex = True)
)

x_train,x_test,y_train,y_test = train_test_split(
    df['Formatted'] , df['sentiment'] , test_size=0.20 , random_state = 1
)
count_vector = CountVectorizer()
training_data = count_vector.fit_transform(x_train)
testing_data = count_vector.transform(x_test)

#Training the model
model = LogisticRegression(random_state  = 0, max_iter= 1000)
model.fit(training_data,y_train)

prediction = model.predict(testing_data)

print(f"Accuracy Score : {accuracy_score(y_test , prediction)}")
while True :
    # Making the predictions

    con_inp = input("Enter the Message :")
    inp = np.array(con_inp)
    inp = np.reshape(inp, (1,-1))
    inp_conv = count_vector.transform(inp.ravel())
    result = model.predict(inp_conv)

    for element in result :
        if result[0] == 0:
            print("Neutral")
        elif result[0] == 1:
            print("Happy")
        else :
            print("Sad")

