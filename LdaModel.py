#coding:utf-8
from CorpusDoc import *
import numpy as np
from scipy import spatial

class LdaModel(CorpusModels):
    '''
    该类通过对文档-单词的TF-IDF矩阵进行分解,得到单词-隐主题分布矩阵和隐主题-文档分布矩阵
    M = UΣV
    应用: 1.通过U可以进行单词聚类,发现同义词单词 2. 通过V可以进行文档相似性计算
    '''

    def get_lda_model(self, lda_model_file='result/lda.model', corpus_dict=CorpusDict(),
                           dictionary=CorpusDictionary().get_dictionary()):
        if os.path.exists(os.path.join(lda_model_file)):
            return models.LdaModel.load(os.path.join(lda_model_file))
        starttime = datetime.datetime.now()
        tfidf = self.get_tfidf_model()
        # corpus_tfidf = tfidf[corpus]
        lda = models.LdaModel(tfidf[corpus_dict], id2word=dictionary, num_topics=30)
        lda.save(os.path.join(lda_model_file))
        endtime = datetime.datetime.now()
        print('建立LDA花费时间%d秒' % (endtime - starttime).seconds)
        return lda

    def get_predict_result(self):
        lda = self.get_lda_model()
        corpus = self.get_corpus()
        index = similarities.MatrixSimilarity(lda[corpus])

        select_name, data = self.random_doc()

        dictionary = CorpusDictionary().get_dictionary()
        vec_bow = dictionary.doc2bow(jieba.cut(data, cut_all=False))
        vec_lsi = lda[vec_bow]
        print 'topic probability:'
        print(vec_lsi)
        sims = sorted(enumerate(index[vec_lsi]), key=lambda item: -item[1])
        print 'top 10 similary notes:'
        names = [name for name, title, data in CorpusDocument(is_cut=False)]
        for sim_pair in sims[:5]:
            print('%s vs %s sim=%f' % (select_name, names[sim_pair[0]], sim_pair[1]))
        return sims[0][0]

    def evaluate_model(self):
        """
        計算模型在test data的Perplexity
        :param model:
        :return:model.log_perplexity float
        """
        test_corpus = self.get_corpus()
        model=self.get_lda_model()
        return model.log_perplexity(test_corpus)



if __name__ == '__main__':
    starttime = datetime.datetime.now()
    corpus_models=LdaModel()
    lda=corpus_models.get_lda_model()
    i=0
    for t in lda.print_topics(30):
        print '[topic #%s]: ' % i, t
        i += 1
    test_doc_id=corpus_models.get_predict_result()
    print corpus_models.evaluate_model()
    #for closest_doc_id in closest_doc_ids[:topn]:
        #print('closest doc %s  % (closest_doc_id, texts[closest_doc_id]))
