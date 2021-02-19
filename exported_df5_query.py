import pandas as pd
import numpy as np
reviews_df = pd.read_csv("Dataset/exported_df5_1.csv", encoding = "ISO-8859-1")  # Read data. Important!
df = reviews_df.copy()
print(df.shape)
df.info()


df["Review Text"][df['Review Rating'] == 1].count()
def printit(star):
    print(
            "\n",star, "yıldızlı reviews icin"
            "\n Toplam        : ", df["Review Text"][df['Review Rating'] == star].count(),
            "\n Toplam  true  : ", df["Review Text"][(df['Review Rating'] == star) & (df['Review Rating'] == df.Tag )].count(),
            "\n Toplam  false : ", df["Review Text"][(df['Review Rating'] == star) & (df['Review Rating'] != df.Tag)].count(),
            "\n 0.6 >         : ", df["Review Text"][(df['Review Rating'] == star) & (df.Score < 0.6)].count(),
            "\n Uzun          : ", df.loc[(df['Big'] == 1) & (df['Review Rating'] == star), 'Review Text'].count(),                  # Uzun ve 1 yıldızlı yorum varmış
            "\n Uzun + true   : ", df.loc[(df['Big'] == 1) & (df['Review Rating'] == star) & (df['Review Rating'] == df.Tag), 'Review Text'].count(),  # Bunlardan bu kadarı doğru tespit edlmiş.
            "\n Uzun + false  : ", df.loc[(df['Big'] == 1) & (df['Review Rating'] == star) & (df['Review Rating'] != df.Tag), 'Review Text'].count(),  # Bu kadarı yanlış tespit gibi
            "\n Kısa          : ", df.loc[(df['Big'] == 0) & (df['Review Rating'] == star), 'Review Text'].count(),                  # Kısa ve 1 yıldızlı
            "\n Kısa + true   : ", df.loc[(df['Big'] == 0) & (df['Review Rating'] == star) & (df['Review Rating'] == df.Tag), 'Review Text'].count(),  # Bu kadarı doğru tespit gibi
            "\n Kısa + false  : ", df.loc[(df['Big'] == 0) & (df['Review Rating'] == star) & (df['Review Rating'] != df.Tag), 'Review Text'].count(),  # Bu kadarı yanlış tespit gibi
    )

for i in range(1,6):
    printit(i)


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# df.loc[df['Review Rating'] == 1, 'C'] = 0
# df.loc[df['Review Rating'] == 2, 'C'] = 0
# df.loc[df['Review Rating'] == 3, 'C'] = 0
# df.loc[df['Review Rating'] == 4, 'C'] = 1
# df.loc[df['Review Rating'] == 5, 'C'] = 1

# df.loc[df['Score'] < 0.6, 'Tag'] = 3

y_true = df['Review Rating']
y_pred = df['Tag']
print(confusion_matrix(y_true, y_pred))
print("acc: ", accuracy_score(y_true, y_pred))
print('\n\nReport: ', classification_report(y_true, y_pred), sep="\n")
#
# label : 'POSITIVE' , score: 0.9898956
# label : 'NEGATIVE' , score: 0.9898956