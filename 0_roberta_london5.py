from transformers import pipeline, AutoTokenizer, TFAutoModelForSequenceClassification
import re
import pandas as pd

model_name="cardiffnlp/twitter-roberta-base-sentiment"
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, from_pt=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

# ---------------------------------------------- Open/Read CSV File-----------------------------------------------------
reviews_df_com = pd.read_csv("DataSet/London5_1.csv", encoding = "ISO-8859-1")  # Read data. Important!
# reviews_df_com = reviews_df[['Review Text','Review Rating']]
# reviews_df_com = reviews_df_com.sample(frac = 0.02, replace = False, random_state=42) # Take just %1 of all reviews

extraColumnName = ['Big', 'Tag', 'Score']
for index, columnName in enumerate(extraColumnName):
    reviews_df_com.insert((reviews_df_com.shape[1]), columnName, 0, True)

reviews_df_com.Score = reviews_df_com.Score.astype(float)
print(reviews_df_com.shape)

# # Yeni Bölüm

big = 0
print(reviews_df_com.shape)
reviews_df_com = reviews_df_com[:9335]  # 5609, 7878, 9333,
# reviews_df_com = reviews_df_com[reviews_df_com['Review Rating'] == 1][:50]
for t in range(len(reviews_df_com)):                        # iterate for each object
    i = str(reviews_df_com['Review Text'].values[t])
    # print(i)
    i = re.sub("\"", '', i)
    if len(i) > 2000:
        review = i[:2000]
        reviews_df_com['Big'].values[t] = 1
        print(t)
    else:
        review = i
        print(t)

    # review = i
    classified = classifier(review)
    if classified[0]['label'] == "LABEL_0":
        reviews_df_com['Tag'].values[t] = -1
    elif classified[0]['label'] == "LABEL_1":
        reviews_df_com['Tag'].values[t] = 0
    else:
        reviews_df_com['Tag'].values[t] = 1

    reviews_df_com['Score'].values[t] = round(classified[0]['score'], 3)

    # print(i, "\nTag : ",reviews_df_com['Tag'].values[t])

# dfcim = reviews_df_com[reviews_df_com['Review Rating'] == 1 ]
reviews_df_com.to_csv (r'C:\Users\Demir\PycharmProjects\2021-Makale-NLP\Dataset\exported_roberta_df1.csv', index = False, header=True)
