import pickle
import codecs
import numpy as np
import pandas as pd
import utils
from gensim import corpora
from gensim.utils import simple_preprocess
from gensim.models.coherencemodel import CoherenceModel

def load_vocabulary(data_url):
    vocabulary = []
    #vocabulary = {}
    with open(data_url) as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            word_data = line.split()
            #vocabulary[str(word_data[1])] = word_data[0]
            vocabulary.append(word_data[0])
    return vocabulary

def data_set(data_url, vocab_size):
    """process data input."""
    data_list = []
    word_count = []
    with open(data_url) as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            id_freqs = line.split()
            # id_freqs = line.split()[1:]
            doc = {}
            count = 0
            for id_freq in id_freqs:
                items = id_freq.split(':')
                # python starts from 0
                doc[int(items[0]) - 1] = int(items[1])
                # doc[int(items[0])] = int(items[1])
                count += int(items[1])
            if count > 0:
                data_list.append(doc)
                word_count.append(count)

    data_mat = np.zeros((len(data_list), vocab_size), dtype=np.float)
    for doc_idx, doc in enumerate(data_list):
        for word_idx, count in doc.items():
            data_mat[doc_idx, word_idx] += count

    return data_mat


def compute_coherence(doc_word, topic_word, N):
    topic_size, word_size = np.shape(topic_word)
    doc_size = np.shape(doc_word)[0]
    # find top words'index of each topic
    topic_list = []
    for topic_idx in range(topic_size):
        top_word_idx = np.argpartition(topic_word[topic_idx, :], -N)[-N:]
        topic_list.append(top_word_idx)

    # compute coherence of each topic
    sum_coherence_score = 0.0
    for i in range(topic_size):
        word_array = topic_list[i]
        sum_score = 0.0
        for n in range(N):
            for l in range(n + 1, N):
                p_n = 0.0
                p_l = 0.0
                p_nl = 0.0
                for j in range(doc_size):
                    if doc_word[j, word_array[n]] != 0:
                        p_n += 1
                    # whether l^th top word in doc j^th
                    if doc_word[j, word_array[l]] != 0:
                        p_l += 1
                    # whether l^th and n^th top words both in doc j^th
                    if doc_word[j, word_array[n]] != 0 and doc_word[j, word_array[l]] != 0:
                        p_nl += 1
                if p_n > 0 and p_l > 0 and p_nl > 0:
                    p_n = p_n / doc_size
                    p_l = p_l / doc_size
                    p_nl = p_nl / doc_size
                    sum_score += np.log(p_nl / (p_l * p_n)) / -np.log(p_nl)
        sum_coherence_score += sum_score * (2 / (N * N - N))
    sum_coherence_score = sum_coherence_score / topic_size
    return sum_coherence_score

def print_coherence(model='cr_nvctm', url='./data/Snippets/train.feat', vocab_size=30642):
    with codecs.open('./{}_train_beta'.format(model), 'rb') as fp:
        beta = pickle.load(fp)
    fp.close()

    test_mat = data_set(url, vocab_size)

    top_n = [5, 10, 15]
    coherence = 0.0
    for n in top_n:
        coherence += compute_coherence(test_mat, np.array(beta), n)
    coherence /= len(top_n)

    print('| NPMI score: {:f}'.format(coherence))


### Functions Diana for coherence with Gensim and store topics




def compute_coherence_gensim(topic_words, url_data='..\\data\\reviews\\translated_text.csv'):
    print("in coherence")
    df = pd.read_csv(url_data,sep=",")
    print("after df")
    texts = df["deep_translation"].apply(simple_preprocess)
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    print("create dictionary and corpus")
    topic_size, word_size = np.shape(topic_words)
    topic_list = []
    topic_ids = []
    for topic_idx in range(topic_size):
        top_word_idx = np.argpartition(topic_words[topic_idx, :], -10)[-10:]
        top_words = [dictionary.get(item) for item in top_word_idx]
        topic_list.append(top_words)
        topic_ids.append(top_word_idx.tolist())
    print("----------------topic list----------------")
    print(topic_list)
    print("----------------topic ids-----------------")
    print(topic_ids)
    topics_test = [['items','place','shop','stock','large'],
                    ['products','food','good','market','crowded'],
                    ['less','comparitely','price','shopping','sorted'],
                    ['always','space','clear','salespeople','incredible'],
                    ['young','products','space','crowded','language']]
    c_npmi = CoherenceModel(topics=topics_test, texts=texts, corpus=corpus, dictionary=dictionary, coherence='c_npmi')
    return c_npmi.get_coherence()

def print_coherence_gensim(beta, model='cr_nvctm'):
    # with codecs.open('./{}_train_beta'.format(model), 'rb') as fp:
    #     beta = pickle.load(fp)
    # fp.close()
    topic_matrix = np.array(beta)
    topic_size, word_size = np.shape(topic_matrix)
    print("topic size:"+str(topic_size))
    print("word size:"+ str(word_size))
    print("")
    coherence = 1
    #coherence = compute_coherence_gensim(np.array(beta))
    print('| NPMI score: {:f}'.format(coherence))
    
def save_topics(doc_word, topic_word, N, vocabulary,file_name):
    topic_size, word_size = np.shape(topic_word)
    doc_size = np.shape(doc_word)[0]
    # find top words'index of each topic
    topic_list = []
    for topic_idx in range(topic_size):
        print("topic: "+str(topic_idx))
        top_word_idx = np.argpartition(topic_word[topic_idx, :], -N)[-N:]
        print(str(top_word_idx))
        top_words = [vocabulary[idx-1] for idx in top_word_idx]
        topic_list.append(str(top_words))
    f = open(file_name, "w")
    #f.write('---------------Printing the Topics------------------\n')
    for i in range(len(topic_list)):
        f.write(topic_list[i]+'\n')
    #f.write('---------------End of Topics------------------\n')
    f.close()

def print_topics(beta, model='cr_nvctm',url='./data/Snippets/train.feat', vocab_size=5000, url_dictionary = './data/reviews/reviews.vocab', file_name = 'topics.txt'):
    with codecs.open('./{}_train_beta'.format(model), 'rb') as fp:
        beta = pickle.load(fp)
    fp.close()
    print(vocab_size)
    test_mat = data_set(url,vocab_size)
    #load vocabulary as dictionary
    vocabulary = load_vocabulary(url_dictionary)
    top_n = [10]
    for n in top_n:
        save_topics(test_mat, np.array(beta), n,vocabulary, file_name)
    #print(vocabulary)
