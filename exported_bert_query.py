import pandas as pd
import numpy as np
reviews_df = pd.read_csv("Dataset/exported_bert_df.csv", encoding ="ISO-8859-1")  # Read data. Important!
df = reviews_df.copy()
print(df.shape)
df.info()


print(
        "\n", "< 3 "
        "\n Rating  < 3  : ", df["Review Text"][df['Review Rating'] < 3].count(),
        "\n Tag     < 3  : ", df["Review Text"][df.Tag < 3].count(),
        "\n R+Tag   < 3  : ", df["Review Text"][(df['Review Rating'] < 3) & (df.Tag < 3 )].count(),

        "\n\n", "== 3 "
        "\n Rating  = 3  : ", df["Review Text"][df['Review Rating'] == 3].count(),
        "\n Tag     = 3  : ", df["Review Text"][df.Tag == 3].count(),
        "\n R+Tag   = 3  : ", df["Review Text"][(df['Review Rating'] == 3) & (df.Tag == 3)].count(),

        "\n\n", "> 3 "
        "\n Rating  > 3  : ", df["Review Text"][df['Review Rating'] > 3].count(),
        "\n Tag  > 3  : ", df["Review Text"][df.Tag > 3].count(),
        "\n R+Tag   > 3  : ", df["Review Text"][(df['Review Rating'] > 3) & (df.Tag > 3)].count(),
)


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

df.loc[df['Review Rating'] == 1, 'RR'] = -1
df.loc[df['Review Rating'] == 2, 'RR'] = -1
df.loc[df['Review Rating'] == 3, 'RR'] = 0
df.loc[df['Review Rating'] == 4, 'RR'] = 1
df.loc[df['Review Rating'] == 5, 'RR'] = 1

df.loc[df['Tag'] == 1, 'T'] = -1
df.loc[df['Tag'] == 2, 'T'] = -1
df.loc[df['Tag'] == 3, 'T'] = 0
df.loc[df['Tag'] == 4, 'T'] = 1
df.loc[df['Tag'] == 5, 'T'] = 1


y_true = df['RR']
y_pred = df['T']
print("\n", confusion_matrix(y_true, y_pred))
print("acc: ", accuracy_score(y_true, y_pred))
print('\n\nReport: ', classification_report(y_true, y_pred), sep="\n")
#
# label : 'POSITIVE' , score: 0.9898956
# label : 'NEGATIVE' , score: 0.9898956