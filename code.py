import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy


from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv('/content/drive/MyDrive/IPB/Dataset/Dataset-2021.csv')

df_new = copy.deepcopy(df)


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_df = 0.90, min_df = 2, stop_words = 'english')
cv_fit = cv.fit_transform(df_new.Abstrak)


from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components = 5, random_state = 1)
print('Fitting the vectorzer with the LDA')
lda.fit(cv_fit)


print('Number of topics:', len(lda.components_))
print('Number of columns of the lda fit:', len(lda.components_[0]))

feature = cv.get_feature_names()

for ind, topic in enumerate(lda.components_):
    print('Top 15 words in topic {}'.format(ind))
    print('-'*25)
    top_15 = topic.argsort()[-15:]
    print([feature[i] for i in top_15], '\n\n')

from wordcloud import WordCloud, STOPWORDS
def word_cloud(topic):
    # plt.figure(figsize = (8,6))
    topic_words = [feature[i] for i in lda.components_[topic].argsort()[-15:]]
    cloud = WordCloud(stopwords = STOPWORDS, background_color = 'white',
                      width=2500, height=1800).generate(" ".join(topic_words))

    print('\nWorcloud for topic:', topic, '\n')
    plt.imshow(cloud)
    plt.axis('off')
    plt.show()

word_cloud(0)
word_cloud(1)
word_cloud(2)
word_cloud(3)
word_cloud(4)