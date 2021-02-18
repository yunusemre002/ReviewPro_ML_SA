from transformers import pipeline, AutoTokenizer, TFAutoModelForSequenceClassification
# model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
# model_name = "bert-base-uncased"
# model_name = "distilbert-base-uncased"
# model_name = "roberta-base"
# model_name = "bert-large-uncased"
# model = TFAutoModelForSequenceClassification.from_pretrained(model_name, from_pt=True)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

classifier = pipeline('sentiment-analysis')

# results = classifier(["it looks good", "i'm hate from you", "that is awfull", "Hi, my name is knajh"])
# for result in results:
#     print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

import pandas as pd
# ---------------------------------------------- Open/Read CSV File-----------------------------------------------------
reviews_df_com = pd.read_csv("DataSet/London4.csv", encoding = "ISO-8859-1")  # Read data. Important!
# reviews_df_com = reviews_df[['Review Text','Review Rating']]
# reviews_df_com = reviews_df_com.sample(frac = 0.02, replace = False, random_state=42) # Take just %1 of all reviews

extraColumnName = ['Big', 'Tag', 'Score']
for index, columnName in enumerate(extraColumnName):
    reviews_df_com.insert((reviews_df_com.shape[1]), columnName, 0, True)

reviews_df_com.Score = reviews_df_com.Score.astype(float)
print(reviews_df_com.shape)
print()


# Yeni Bölüm

neg = 0
pos = 0
big = 0
import re
# reviews_df_com = reviews_df_com[0:]  # 12640
print(reviews_df_com.shape)

for t in range(len(reviews_df_com)):                        # iterate for each object
    i = str(reviews_df_com['Review Text'].values[t])
    # if reviews_df_com['Review Rating'].values[t] == 5:
    i = re.sub("\"", '', i)
    if len(i) > 2047:
        review = i[:2047]
        reviews_df_com['Big'].values[t] = 1
        print(t)
    else:
        review = i
        print(t)

    # review = i
    # print(t)

    classified = classifier(review)
    reviews_df_com['Tag'].values[t] = 0 if classified[0]['label'] == 'NEGATIVE' else 1
    reviews_df_com['Score'].values[t] = round(classified[0]['score'], 3)

    # print(round(classified[0]['score'], 3), reviews_df_com['Score'].values[t])

# dfcim = reviews_df_com[reviews_df_com['Review Rating'] == 1 ]
reviews_df_com.to_csv (r'C:\Users\Demir\PycharmProjects\2021-Makale-NLP\Dataset\exported_df4.csv', index = False, header=True)
















# neg = 0
# pos = 0
# big = 0
#   if len(i) < 512:
#     # re = classifier(i)
# #     if re[0]['label'] == "NEGATIVE":
# #       # print(f"{t}, label: {re[0]['label']}, with score: {round(re[0]['score'], 4)}")
# #       neg +=1
# #     elif re[0]['label'] == "POSITIVE":
# #       pos +=1
#   elif len(i) < 1024:
#     big +=1
#     # print(f"512< {i}")
#     # print(classifier(i))
#   else:
#     print(f"1024< {i}")
    # print(classifier(i))
# print(f"pos: {pos}, neg: {neg}, big: {big}")

# ## pos: 211, neg: 12, big: 239
