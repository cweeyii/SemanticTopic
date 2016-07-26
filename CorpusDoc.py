#coding:utf-8

import jieba
import re
import os
from gensim import corpora, models, similarities
import datetime
from six import iteritems
import random

regex = re.compile(ur"[^\u4e00-\u9f5aa-zA-Z0-9]")

class CorpusStopWord(object):
    def __init__(self, corpus_dir='data/stopword'):
        self.root_dir = corpus_dir

    def get_stop_words(self):
        stop_words=set()
        for name in os.listdir(self.root_dir):
            if os.path.isfile(os.path.join(self.root_dir, name)):
                data = open(os.path.join(self.root_dir, name), 'rb').read()
                data=data.split('\n')
                stop_words.update(data)
        return stop_words


class CorpusDocument(object):
    '''
    该类通过获取指定陌路下的文件,并读取文件内容,经过结巴分词返回分词后的列表
    '''
    def __init__(self, corpus_dir='data/corpus',stop_words=CorpusStopWord().get_stop_words(), is_cut=True):
        self.root_dir = corpus_dir
        self.stop_words=stop_words
        self.is_cut=is_cut
        #jieba.enable_parallel(6)

    def __iter__(self):
        def etl(s):  # remove 标点和特殊字符
            return regex.sub('', s)
        for name in os.listdir(self.root_dir):
            if os.path.isfile(os.path.join(self.root_dir, name)):
                data = open(os.path.join(self.root_dir, name), 'rb').read()
                title = data[:data.find('\r\n')]
                if self.is_cut:
                    data = filter(lambda x: len(x) > 0, map(etl, jieba.cut(data, cut_all=False)))
                    data=[ d.lower() for d in data if d.lower() not in self.stop_words ]
                yield (name, title, data)

class CorpusDictionary(object):

    def __init__(self, dic_file_name='result/normal_dictionary.dict'):
        self.dic_file_name = dic_file_name

    def get_dictionary(self, doc_len=17562, stop_ids=set()):
        if os.path.exists(os.path.join(self.dic_file_name)):
            return corpora.Dictionary.load(os.path.join(self.dic_file_name))
        starttime = datetime.datetime.now()
        COMMON_LINE = doc_len / 10
        dictionary = corpora.Dictionary(data for name, title, data in CorpusDocument())
        too_common_words = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq > COMMON_LINE]
        once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
        stop_ids.update(too_common_words)
        stop_ids.update(once_ids)
        dictionary.filter_tokens(stop_ids)
        dictionary.compactify()
        dictionary.save(os.path.join(self.dic_file_name))
        endtime = datetime.datetime.now()
        print('建立Dictionary花费时间%d秒' % (endtime - starttime).seconds)
        return dictionary


class CorpusDict(object):
    def __init__(self, document=CorpusDocument(),dictionary=CorpusDictionary().get_dictionary()):
        self.document = document
        self.dictionary = dictionary

    def __iter__(self):
        for name, title, data in self.document:
            yield self.dictionary.doc2bow(data)

class CorpusModels(object):

    def get_corpus(self, corpus_file='result/corpus.mm'):
        if os.path.exists(os.path.join(corpus_file)):
            return corpora.MmCorpus(os.path.join(corpus_file))
        starttime = datetime.datetime.now()
        corpus = list(CorpusDict())
        corpora.MmCorpus.serialize(os.path.join(corpus_file),corpus)
        endtime = datetime.datetime.now()
        print('建立corpus花费时间%d秒' % (endtime - starttime).seconds)
        return corpus

    def get_tfidf_model(self,tfidf_model_file='result/tf_idf.model',document=CorpusDocument(), dictionary=CorpusDictionary().get_dictionary()):
        if os.path.exists(os.path.join(tfidf_model_file)):
            return models.TfidfModel.load(os.path.join(tfidf_model_file))
        starttime = datetime.datetime.now()
        tfidf = models.TfidfModel(CorpusDict(document,dictionary))
        tfidf.save(tfidf_model_file)
        endtime = datetime.datetime.now()
        print('建立tf-idf花费时间%d秒' % (endtime - starttime).seconds)
        return tfidf

    def random_doc(self):
        name = random.choice(os.listdir('data/corpus'))
        data = open('data/corpus/%s' % name, 'rb').read()
        print 'random choice ', name
        return name, data

