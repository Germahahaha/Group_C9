#K'means

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import gensim
from gensim.models import Doc2Vec
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
from sklearn.externals import joblib
from string import punctuation
import tensorflow as tf
import os, json, re
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from nltk.stem.snowball import SnowballStemmer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib

matplotlib.rcParams['font.sans-serif']=['Times New Roman']   # Modify the fonttype
matplotlib.rcParams['axes.unicode_minus']=False 
nltk.download('stopwords')
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
p_stemmer = PorterStemmer()
stemmer = SnowballStemmer("english")

root_path = os.path.abspath(os.path.dirname(__file__))
file_path = root_path+'\\stopwords.txt'
print(file_path)
with open(file_path, 'r', encoding='utf8') as f:
    newstopwords = []
    for line in f:
        temp = line.strip('\n')
        newstopwords.append(temp)
for stopword in newstopwords:
    stop_words.add(str(stopword))

# Data cleaning
def data_cleaning(text):
    data = text.lower()
    # clean and tokenize document string
    data_content = data.split()
    word_list = []
    for i in data_content:
        x = 0
        if (('http' not in i) and ('@' not in i) and ('<.*?>' not in i) and ( i.isdigit() == False)  and (not i in stop_words)):   # and i.isalnum()
            i = i.replace(".","")
            word_list += [i]
    return word_list

# Data Pre-processing
def preprocessing(text):
    # remove numbers
    number_tokens = [re.sub(r'[\d]', ' ', i) for i in text]
    number_tokens = ' '.join(number_tokens).split()
    # remove empty
    length_tokens = [i for i in number_tokens if len(i) > 1]
    return length_tokens


json_body = []
with tf.device('/device:GPU:0'):
    with open(root_path+'\\cveresult.json', 'r', encoding='utf8') as f:
        line = f.readline()
        json_text = json.loads(line)

    for i in range(len(json_text)):
        json_info = json_text[str(i)]["description"]
        json_body.append(json_info)

with tf.device('/device:GPU:1'):
    LabeledSentence1 = gensim.models.doc2vec.TaggedDocument
    all_content = []
    all_content_notoken = []
    texts = []
    j = 0
    k = 0
    for em in json_body:
        # Data cleaning
        clean_content = data_cleaning(em)
        # Pre-processing
        processed_content = preprocessing(clean_content)
        # add tokens to list
        if processed_content:
            label_content = LabeledSentence1(processed_content, [j])
            all_content.append(label_content)
            j += 1
        k += 1
    totalvocab_stemmed = []
    totalvocab_tokenized = []

    print("Number of contents processed: ", k)
    print("Number of non-empty contents vectors: ", j)

    ######################## Using kmeans tfidf Cluster
    d2v_model = Doc2Vec(all_content, vector_size=100, window=10, min_count=5, workers=7, dm=1,
                        alpha=0.025, min_alpha=0.001)
    d2v_model.train(all_content, total_examples=d2v_model.corpus_count, epochs=10, start_alpha=0.002, end_alpha=-0.016)
    num_clusters = 7
    # Apply K-means clustering on the model
    kmeans_model = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=100)
    X = kmeans_model.fit(d2v_model.docvecs.doctag_syn0)
    labels = kmeans_model.labels_.tolist()
    l = kmeans_model.fit_predict(d2v_model.docvecs.doctag_syn0)
    pca = PCA(n_components=2).fit(d2v_model.docvecs.doctag_syn0)
    datapoint = pca.transform(d2v_model.docvecs.doctag_syn0)

    joblib.dump(kmeans_model, root_path+'\\doc_cluster_cve.pkl')
    km = joblib.load(root_path+'\\doc_cluster_cve.pkl')  # use the saved model
    clusters = km.labels_.tolist()
    contents = {'content': all_content, 'cluster': clusters}
    frame = pd.DataFrame(contents, index=[clusters], columns=['content', 'cluster'])

    for v in all_content:
        teststr = " ".join(str(i) for i in v)
        all_content_notoken.append(teststr)

    vectorizer = TfidfVectorizer(max_df=0.5, max_features=500, min_df=2,stop_words=stop_words)
    X = vectorizer.fit_transform(all_content_notoken)
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(num_clusters):
        print("Cluster %d words: " % i, end='')
        for ind in order_centroids[i, :20]:
            print(' %s' % terms[ind], end=' ')
        print()

    framedf = frame
    framedf.index = range(0,len(framedf))
    datapointdf = pd.DataFrame(datapoint, columns = ['x', 'y'])
    newframe = framedf.join(datapointdf, how='right')

    newframe.to_csv(root_path+'\\data_cve.csv')

    # plt.figure
    label1 = ["#ee4035", "#f37736", "#fdf498", "#7bc043", "#0392cf", "#96ceb4", "#ffeead", "#ff6f69", "#ffcc5c",
              "#88d8b0", "#76b4bd", "#ffbbee"]
    color = [label1[i] for i in labels]
    plt.scatter(datapoint[:, 0], datapoint[:, 1], c=color)

    red_patch = []
    patch0 = mpatches.Patch(color='#ee4035', label='0')
    patch1 = mpatches.Patch(color='#f37736', label='1')
    patch2 = mpatches.Patch(color='#fdf498', label='2')
    patch3 = mpatches.Patch(color='#7bc043', label='3')
    patch4 = mpatches.Patch(color='#0392cf', label='4')
    patch5 = mpatches.Patch(color='#96ceb4', label='5')
    patch6 = mpatches.Patch(color='#ffeead', label='6')
    plt.legend(
        handles=[patch0, patch1, patch2, patch3, patch4, patch5, patch6], fontsize = 18)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.savefig(root_path+'\\Kmeans.png',dpi = 240)
    plt.show()