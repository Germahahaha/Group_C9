#SVM

import re
import numpy
from sklearn.svm import SVC
import gensim
from gensim.models import Doc2Vec
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
from string import punctuation
import tensorflow as tf
import os, json
from sklearn import metrics
import matplotlib.pyplot as plt
from nltk.stem.snowball import SnowballStemmer
import random
from sklearn import metrics
from sklearn.metrics import auc
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

with open(file_path, 'r', encoding='utf8') as f:
    newstopwords = []
    for line in f:
        temp = line.strip('\n')  # 去掉每行最后的换行符'\n'
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
    length_tokens = [i for i in number_tokens if len(i) > 1]
    return length_tokens

json_body = []
json_label= []
with tf.device('/device:GPU:0'):
    with open(root_path+'\\cveresult.json', 'r', encoding='utf8') as f:
        line = f.readline()
        json_text = json.loads(line)

    train_data_len = 9600
    txt_vec_len = 100
    test_data_len = 2400
    train_index = random.sample(range(0, 50847),train_data_len)
    index_all = list(range(0, 50847))
    print('index_all type',type(index_all))
    for item in train_index:
        index_all.remove(item)
    index_test_2 = random.sample(range(1, 50847-train_data_len),test_data_len)
    index_all = numpy.array(index_all)
    test_index = index_all[index_test_2]
    for i in range(len(json_text)):
        json_info = json_text[str(i)]["description"]
        label = json_text[str(i)]["impactscore"]
        json_body.append(json_info)
        if type(label) is str:
            label =0

        if 4.0 <= label and 7.0 > label:
            json_label.append(2)
        elif 7.0 <= label and 10.0 >= label:
            json_label.append(3)
        else:
            json_label.append(1)

    json_label = numpy.array(json_label)
    train_label = json_label[train_index]
    test_label = json_label[test_index]
    print('labeled')

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

    ######################## Using sklearn.svm
    d2v_model = Doc2Vec(all_content, vector_size=txt_vec_len, window=10, min_count=5, workers=7, dm=1,
                        alpha=0.025, min_alpha=0.001)
    vecs = [numpy.array(d2v_model.docvecs[z.tags[0]]).reshape((1, txt_vec_len)) for z in all_content]
    vecs = numpy.array(vecs)
    train_data = vecs[train_index]
    test_data = vecs[test_index]

    model_svm = SVC(decision_function_shape = 'ovr',C=20000, kernel = 'poly',degree = 5, gamma = 'scale')
    vecs = numpy.array(vecs)
    train_data = numpy.array(train_data)
    test_data = numpy.array(test_data)

    train_data = numpy.reshape(train_data, (len(train_data),txt_vec_len))
    test_data = numpy.reshape(test_data, (len(test_data),txt_vec_len))

    train_label = numpy.array(train_label)
    test_label = numpy.array(test_label)
    model_svm.fit(train_data, train_label)
    
    print(model_svm.predict(numpy.array(train_data[0:200])))
    print('train data size:',train_data.shape)
    print('test data size:',test_data.shape)
    print('train correct rate:',model_svm.score(train_data, train_label))
    print(model_svm.score(test_data, test_label))
    print('multiclass classification')

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    true_label_1 = []
    true_label_2 = []
    true_label_3 = []
    for item in range(len(test_label)):
        if test_label[item] == 1:
            true_label_1.append(1)
            true_label_2.append(0)
            true_label_3.append(0)
        elif test_label[item] == 2:
            true_label_1.append(0)
            true_label_2.append(1)
            true_label_3.append(0)
        else:
            true_label_1.append(0)
            true_label_2.append(0)
            true_label_3.append(1)

    pre_label = numpy.array(model_svm.decision_function(test_data))
    true_label_1 = numpy.array(true_label_1)
    true_label_2 = numpy.array(true_label_2)
    true_label_3 = numpy.array(true_label_3)

    fpr['1'], tpr['1'], thersholds = metrics.roc_curve(true_label_1, pre_label[:,0])
    roc_auc['1'] = auc(fpr['1'], tpr['1'])
    fpr['2'], tpr['2'], thersholds = metrics.roc_curve(true_label_2, pre_label[:,1])
    roc_auc['2'] = auc(fpr['2'], tpr['2'])
    fpr['3'], tpr['3'], thersholds = metrics.roc_curve(true_label_3, pre_label[:,2])
    roc_auc['3'] = auc(fpr['3'], tpr['3'])
    print(roc_auc['1'])
    print(roc_auc['2'])
    print(roc_auc['3'])

    colors = ['aqua', 'darkorange', 'deeppink', 'navy', 'cornflowerblue']
    lw=2
    plt.figure()
    n_classes = ['1', '2', '3']
    for i in n_classes:
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=colors[int(i)], lw=lw,
                label='ROC curve of class {0} (AUC = {1:0.2f})'
                ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('SVM')
    plt.legend(loc="lower right")
    plt.savefig(root_path+'\\ROC_Curve.png',dpi = 240)
    plt.show()
