import datetime    # 用于显示时间
import math    # 用于数学计算，如计算相似度
import os    # 用于系统操作，如获取目录
import re    # 用于处理字符串
import sys    # 用于系统操作
import time    # 用于计时

import functools    # 用于比较关键词重要程度
from gensim import corpora, models    # 调用gensim主题模型接口
import html    # 用于处理html转义字符
import numpy as np    # 数据处理包numpy
import pandas as pd    # 数据处理包pandas

import jieba    # 中文分词
import nltk, spacy    # 英文分词    # import pke 英文关键词提取
import MeCab    # 日文分词


def LOG(comment, head='', tail=''):  # 用于显示程序运行的时刻

    print('%s%s %s...%s' % (head, str(datetime.datetime.now())[:23], comment, tail))


def compare(element_1, element_2) -> int:  # 比较函数，用于排序关键词

    res = np.sign(element_1[1] - element_2[1])

    if res != 0:
        return res

    else:

        a = element_1[0] + element_2[0]
        b = element_2[0] + element_1[0]

        if a > b:
            return 1
        elif a == b:
            return 0
        else:
            return -1


def map_keywords(language, keywords: list):  # 将算法得到的关键词归类为具体标签

    #if language == 'english':  # 英文标签匹配尚未完成，直接跳出
    #    return keywords

    mapping_list = pd.read_excel(
        io=''.join([os.getcwd(), '/alternative_tags/', language, '_alternative_tags.xlsx']),
        usecols=[
            'Tag', '关键词2', '关键词3', '关键词4', '关键词5',
            '关键词6', '关键词7', '关键词8', '关键词9', '关键词10',
            '关键词11', '关键词12', '关键词13', '关键词14', '关键词15',
            '关键词16', '关键词17', '关键词18', '关键词19'
        ]
    )  # 读取标签对应表，注意文件更新后需要修改列数

    mapping_dict = {
        record['Tag']: [
            word.lower()
            for word in record.values()
            if not pd.isnull(word)
        ]
        for record in mapping_list.to_dict('records')
        if not pd.isnull(record['Tag'])
    }  # 将标签对应表处理为标签字典

    mapping = {}
    for key, values in mapping_dict.items():  # 将标签字典拆分为词语单独映射
        for value in values:
            mapping[value] = mapping.get(value, []) + [key]

    tags = [word for words in keywords for word in mapping.get(words, [''])]  # 生成标签
    tags_pro = sorted(list(set(tags)), key=tags.index)  # 标签有序去重，保留重要程度排序

    if language == 'english':
        if tags_pro == ['']:
            return ['区块链']  # 无标签则标记为'区块链'

    if language == 'chinese':
        if tags_pro == ['']:
            return ['Blockchain']  # 无标签则标记为'区块链'

    return tags_pro


class TagExtraction(object):

    def __init__(self, language, load_from_saved=False, test=False, save=False):

        # load_from_saved: True表示读取上次程序已读取的语料库，可以节省时间
        # test: True表示使用手动复制的文章进行调试
        # save: True表示将得到的关键词存成文件形式

        self.language = language  # 中文/英文/日文
        self.save = save  # 是否将得到的关键词写入文件

        self.stopwords = self.get_stopwords()  # 停用词词典
        self.userdict = self.get_userdict()  # 自定义词典

        self.corpus = self.get_corpus(load_from_saved)  # 语料库
        self.text = self.get_text(test)  # 文章

    def get_stopwords(self):  # 读取停用词词典，路径为 ./stopwords/***_stopwords.txt（中文、英文、日文）

        with open(file=''.join([os.getcwd(), '/stopwords/', self.language, '_stopwords.txt']), mode='r',
                  encoding='utf-8') as stop_words:
            return [stop_word.strip() for stop_word in stop_words.readlines()]  # 返回停用词列表

    def get_userdict(self):  # 读取自定义词典，路径为 ./userdict/***_userdict.txt（中文、英文），日文的储存方式为csv

        if self.language == 'chinese':  # 返回值为None，直接调用接口加载自定义词典

            jieba.load_userdict(
                ''.join([os.getcwd(), '/userdict/', self.language, '_userdict.txt']))  # 使用jieba的load_userdict方法

        elif self.language == 'english':  # 返回值为存有自定义词典的list

            with open(file=''.join([os.getcwd(), '/userdict/', self.language, '_userdict.txt']), mode='r',
                      encoding='utf-8') as words:
                return [tuple(line.lower().strip().split()) for line in words.readlines()]  # 配合使用nltk的MWE分词器

        elif self.language == 'japanese':  # 返回值为None，生成csv与dic文件

            pass

            # userdic = /data/workdir/dev/nlp/tag_extraction/tag_extraction_api/userdict/userdictjp.dic
            # ビットコイン,1285,1285,5,名詞,固有名詞,*,*,*,*,ビットコイン,ビットコイン,ビットコイン

            # ctg = os.getcwd()
            # os.chdir(''.join([ctg, '/userdict']))
            # os.system('/usr/lib/mecab/mecab-dict-index -d /usr/share/mecab/dic/ipadic -u userdictjp.dic -f euc-jp -t euc-jp japanese_userdict.csv')
            # # os.system('/usr/lib/mecab/mecab-dict-index -d /data/workdir/dev/nlp/lib/python3.6/site-packages/MeCab/dic -u userdictjp.dic -f euc-jp -t euc-jp japanese_userdict.csv')
            # os.chdir(ctg)
            # with open(file='/data/workdir/dev/nlp/lib/python3.6/site-packages/MeCab/mecabrc.in', mode='a', encoding='utf-8') as upload:
            #     upload.write('userdic = /data/workdir/dev/nlp/tag_extraction/tag_extraction_api/userdict/userdictjp.dic\n')

    def sentence_filter(self, sentence):  # 对句子进行初步的分词和清洗

        if self.language == 'chinese':

            import jieba.posseg as psg

            return psg.cut(sentence)  # 使用jieba的分词接口直接完成分词和清洗

        elif self.language == 'english':

            from nltk.tokenize import MWETokenizer  # 使用MWE分词器

            tokenizer = MWETokenizer(self.userdict)  # 添加自定义词组，以下划线'_'为词组连接

            nlp = spacy.load('en_core_web_sm')  # 生成spacy分词器

            # for word in self.userdict:    # spacy添加自定义词语，貌似无效
            #     lex = nlp.vocab[word]

            # 清洗标点符号
            quote_double_pattern = re.compile('“|”')
            quote_single_pattern = re.compile('‘|’')
            punc_pattern = re.compile(
                "\"|\xa0|\t|\n|\:|\;| — | - |–-|\!|\@|\#|\$|\%|\^|\*|\_|\?|？|\(|\)|\[|\]|\{|\}|\<|\>|\||\+|\=|\~|\`|°|\\|\/|，")

            sentence = re.sub(quote_double_pattern, '"', sentence)
            sentence = re.sub(quote_single_pattern, "'", sentence)  # 考虑's和s'的情况，不能直接删掉
            sentence = re.sub(punc_pattern, ' ', sentence)

            # 使用nltk和spacy得到分词结果，使用pke则得到完整句子
            # return nlp(' '.join(sentence.split()))    # spacy
            return nlp(' '.join(tokenizer.tokenize(sentence.lower().split())))  # nltk + spacy: 先用nltk添加词组，再用spacy分词
            # return sentence    # pke

        elif self.language == 'japanese':

            mecab = MeCab.Tagger('')  # 使用mecab的分词器直接得到结果，暂时不能添加自定义词典，有些专有名词识别不出来（如: 比特/币）

            # 清洗标点符号
            punc_pattern = re.compile(
                "\xa0|\t|\n|\:|\;| — | - |\!|\@|\#|\$|\%|\^|\&|\*|\_|\?|\(|\)|\[|\]|\{|\}|\<|\>|\||\+|\=|\~|\`|°|\\|/|・|「|」|•|※")
            sentence = re.sub(punc_pattern, ' ', sentence)

            sentence = [
                (
                    chunk.split('\t')[0],
                    chunk.split('\t')[1].split(',')[0]
                )
                for chunk in mecab.parse(sentence).splitlines()[:-1]
            ]  # 根据词条结构获取词根和词型

            return sentence

    def word_filter(self, sentence):  # 筛选词语的词性和构成

        words: list = []

        def notnum(content):  # 去掉含数字的词语

            if '0' in content or '1' in content or '2' in content or '3' in content or '4' in content or \
                    '5' in content or '6' in content or '7' in content or '8' in content or '9' in content:
                return False
            else:
                return True

        def notadd(content):  # 去掉含网址的词语

            if '.' in content:  # example: '.com' in content or '.org' in content or '.io' in content:
                return False
            else:
                return True

        if self.language == 'chinese':

            for word in sentence:

                if not word.flag.startswith('n'):  # 只保留名词，可以额外添加名词（在userdict中用'n'标注）
                    continue

                if word.word not in self.stopwords and len(word.word) > 1:  # 去除停用词和单个中文字
                    words.append(word.word)

            return words

        elif self.language == 'english':  # spacy进行手动筛选，pke不需要手动筛选

            for word in sentence:

                if '_' in str(word):
                    words.append(str(word.lemma_).lower())  # 仅有词组中含有下划线，识别到含下划线的词组即将其加入到语料库中
                    continue

                if not word.tag_ in ['NN', 'NNS', 'NNP', 'NNPS']:  # 保留英文名词和专用名词以及其复数形式
                    continue

                # 单独使用spacy只能分单词不能分词组，pke可以分词和词组但不能添加自定义词组
                # 解决方案: 使用nltk添加词组，再使用spacy分词
                # and notnum(str(word)) 删除含数字的词组，不采用

                # 去除停用词、含网站地址以及单词长度小于3的单词
                if str(word).lower() not in self.stopwords and notadd(str(word)) and len(str(word.lemma_)) > 2:
                    words.append(str(word.lemma_).lower())

            return words

        elif self.language == 'japanese':

            for word, tag in sentence:

                if not tag == '名詞':  # 只保留名词，包括中文、英文和日文，但效果不太好，要补充停用词及自定义词
                    continue

                if word not in self.stopwords and notnum(word) and len(word) > 1:  # 去除停用词、含数字以及单字日文词
                    words.append(word)

            return words

    def get_corpus(self, load_from_saved):  # 读取语料库

        LOG('Loading corpus', head='\n')

        if load_from_saved:  # 读取上一次程序保存的已处理好的语料库

            # if self.language == 'english':
            #     return ''.join([os.getcwd(), '/corpus/english_text.csv'])    # 使用pke，读取csv格式文件

            with open(file=''.join([os.getcwd(), '/corpus/', self.language, '_corpus_vocabulary.txt']), mode='r',
                      encoding='utf-8') as load:
                return [doc.strip().split() for doc in load.readlines()]  # 从文件中读取已保存的划分好的词组

        else:

            corpus_excel = pd.read_excel(
                io=''.join([os.getcwd(), '/corpus/longhash_news.xlsx']),
                sheet_name=self.language,
                usecols=['title', 'shorttitle', 'summary', 'content']
            )  # 读取longhash_news文件

            if self.language == 'japanese':  # 日文中含有未处理的html标签，需要清洗掉
                corpus_excel['content'] = corpus_excel['content'].map(lambda x: re.sub('<.+?>', '', x))

            corpus_excel['content'] = corpus_excel['content'].map(lambda x: html.unescape(str(x)))  # 转换html转义字符
            corpus_excel.fillna('', inplace=True)  # 将缺失值转化为空串

            func = lambda x: '. '.join([x['title'], x['shorttitle'], x['summary'], x['content']])
            corpus_line = corpus_excel.apply(func, axis='columns')  # 将四列合并为一列，便于处理

            corpus: list = []
            for num, line in enumerate(corpus_line):  # 遍历语料库中每一篇文章
                print(num + 1, end=' ')
                content = line.strip()  # 去掉前后的空格
                sentence = self.sentence_filter(content)  # 清洗句子
                # if self.language == 'english':
                #     corpus.append(sentence)    # 使用pke，直接将句子储存起来，不需要筛选词性
                #     continue
                words = self.word_filter(sentence)  # 对词组进行词性筛选
                corpus.append(words)  # 将词组存入corpus中
            print('\n')

            # if self.language == 'english':
            #     pd.DataFrame(corpus).to_csv(path_or_buf=''.join([os.getcwd(), '/corpus/english_text.csv']), index=False, header=False)
            #     return ''.join([os.getcwd(), '/corpus/english_text.csv'])    # 使用pke，将储存的句子输出至csv文件中

            corpus_wash = [line for line in corpus if line != []]  # 将空行删掉

            with open(file=''.join([os.getcwd(), '/corpus/', self.language, '_corpus_vocabulary.txt']), mode='w',
                      encoding='utf-8') as save:

                # 把语料库保存至corpus文件夹的txt文件中（中文、日文、英文nltk）
                for doc in corpus_wash:
                    save.write(' '.join(doc))
                    save.write('\n')

            return corpus_wash

    def get_text(self, test):  # 读取文章（待提取标签的文本）

        LOG('Loading text')

        if test:  # 使用手动复制文章进行调试

            text_dict: dict = {}
            for text_path in sorted(
                    list(os.walk(''.join([os.getcwd(), '/test/', self.language, '_text'])))[0][2]):  # 遍历测试文章

                # if self.language == 'english':    # 使用pke，只需要存下测试文章的路径
                #     text_dict[text_path] = ''.join([os.getcwd(), '/test/', self.language, '_text/', text_path])
                #     continue

                print('%s: ' % text_path, end='')

                text: list = []
                with open(file=''.join([os.getcwd(), '/test/', self.language, '_text/', text_path]), mode='r',
                          encoding='utf-8') as text_content:

                    # 对调试文章进行逐行清洗，过程和清洗语料库类似
                    for num, line in enumerate(text_content):
                        print(num + 1, end=' ')
                        content = line.strip()
                        sentence = self.sentence_filter(content)
                        words = self.word_filter(sentence)
                        text += words  # 存入的是每篇文章经过清洗后的单词，而语料库原文件是一行一篇文章，text与corpus不同
                    print()

                text_wash = [text_line for text_line in text if text_line != []]  # 删掉空行
                text_dict[text_path] = text_wash  # 存入字典，键值为文章的路径

            return text_dict

        else:  # 不进行调试，中文、日文和英文nltk可以直接使用语料库的处理结果

            return dict(enumerate(self.corpus))  # 字典键值为序号

        # else:    # 不进行调试，英文，使用pke，读取已处理好的csv文件，可以节约处理时间
        #     text_csv = pd.read_csv(filepath_or_buffer=''.join([os.getcwd(), '/corpus/english_text.csv']), header=None, encoding='utf-8')
        #     return text_csv.to_dict()[0]    # 转化为字典后得到词语列表

    def tfidf_extract(self, keyword_num=10):  # 使用tfidf算法，默认为10个关键词

        def train_idf(corpus) -> (dict, float):  # 计算idf值

            idf: dict = {}
            total_count: int = len(corpus)  # 总文档数

            # 每个词出现的文档数
            for line in corpus:
                for word in set(line):
                    idf[word] = idf.get(word, 0.0) + 1.0

            # 按公式转换为idf值，分母加1进行平滑处理
            for key, value in idf.items():
                idf[key] = math.log(total_count / (1.0 + value))

            # 对于没有在字典中的词，默认其仅在一个文档出现，得到默认idf值
            default_idf = math.log(total_count / (1.0))

            return idf, default_idf

        LOG('Loading TF-IDF', head='\n')

        idf, default_idf = train_idf(self.corpus)  # 得到idf值
        for title, content in sorted(self.text.items(), key=lambda x: x[0]):  # 排序文章路径
            tfidf_model = TfIdf(idf, default_idf, content, keyword_num)  # TFIDF类的实例化，得到tfidf模型
            tfidf_model.get_tfidf(title, self.language, self.save)  # 使用TFIDF类的方法，得到tfidf关键词

    def topic_extract(self, model='LDA', keyword_num=10):  # 使用lda算法，默认为10个关键词

        LOG('Loading LDA', head='\n')

        topic_model = TopicModel(self.corpus, keyword_num, model=model)  # 主题模型类的实例化，得到主题模型（LDA）
        topic_model.get_simword(self.text, self.language, self.save)  # 使用主题模型类的方法，得到lda关键词

    ###############################
    # def pke_tfidf_extract(self):    # 使用pke中tfidf算法

    #     LOG('Loading TF-IDF', head='\n')    # Python Keyphrases Extraction

    #     pke.compute_document_frequency(
    #         input_dir=self.corpus,
    #         output_file=''.join([os.getcwd(), '/corpus/english_document_frequency.tsv.gz']),
    #         extension='csv',    # input file extension
    #         language='en',    # language of files
    #         normalization=None,    # use porter stemmer
    #         stoplist=self.stopwords
    #     )    # 计算文档频率

    #     for title, path in sorted(self.text.items(), key=lambda x: x[0]):

    #         extractor = pke.unsupervised.TfIdf()    # 创建tfidf关键词提取器
    #         # for key, value in self.userdict.items():    # 尝试添加自定义词条，但无效
    #         #     extractor.add_candidate(list(key), value[0], value[1], value[2], 100)
    #         extractor.load_document(input=path, language='en', encoding='utf-8', normalization=None)    # 读取待提取文章
    #         extractor.candidate_selection(pos=['NOUN', 'PROPN'], stoplist=self.stopwords)    # 筛选待提取关键词，规定词性和去停用词
    #         extractor.candidate_weighting(
    #             df=pke.load_document_frequency_file(input_file=''.join([os.getcwd(), '/corpus/english_document_frequency.tsv.gz']))
    #         )    # 给每个词组或单词计算权值，得到关键词
    #         keyphrases = extractor.get_n_best(n=10)    # 得到前十关键词

    #        if self.save:    # 保存关键词至文件中
    #             with open(file=''.join([os.getcwd(), '/result/', self.language, '_TFIDF_tags.txt']), mode='a', encoding='utf-8') as tags:
    #                 tags.write(', '.join([key.lower() for key, value in keyphrases]))
    #                 tags.write('\n')

    #         print('%s:' % title, ' / '.join([key.lower() for key, value in keyphrases]))   # 输出关键词

    ###############################
    # def pke_topic_extract(self, model='LDA'):    # 使用pke的主题模型算法

    #     LOG('Loading %s' % model, head='\n') # Python Keyphrases Extraction

    #     for title, path in sorted(self.text.items(), key=lambda x: x[0]):

    #         extractor = pke.unsupervised.TopicRank()    # 创建主题模型关键词提取器
    #         # for key, value in self.userdict.items():    # 尝试添加自定义词条，但无效
    #         #     extractor.add_candidate(list(key), value[0], value[1], value[2], 100)
    #         extractor.load_document(input=path, language='en', encoding='utf-8', normalization=None)    # 读取待提取文章
    #         extractor.candidate_selection(pos=['NOUN', 'PROPN'], stoplist=self.stopwords)    # 筛选待提取关键词，规定词性和去停用词
    #         extractor.candidate_weighting(threshold=0.74, method='average')    # 给每个词组或单词计算权值，得到关键词，threshold为聚类相似度阈值
    #         keyphrases = extractor.get_n_best(n=20, redundancy_removal=True)    # 得到前二十关键词

    #         if self.save:    # 保存关键词至文件中
    #             with open(file=''.join([os.getcwd(), '/result/', self.language, '_LDA_tags.txt']), mode='a', encoding='utf-8') as tags:
    #                 tags.write(', '.join([key.lower() for key, value in keyphrases]))
    #                 tags.write('\n')

    #         print('%s:' % title, ' / '.join([key.lower() for key, value in keyphrases]))   # 输出关键词


class TfIdf(object):  # 参考书上的算法模型

    def __init__(self, idf, default_idf, text, keyword_num):

        self.text = text  # 训练好的idf字典
        self.idf, self.default_idf = idf, default_idf  # 默认idf值
        self.tf = self.get_tf()  # 处理后的待提取文本
        self.keyword_num = keyword_num  # 关键词数量

    def get_tf(self) -> dict:  # 统计tf值

        tf: dict = {}
        for word in self.text:
            tf[word] = tf.get(word, 0.0) + 1.0

        total_count = len(self.text)
        for key, value in tf.items():
            tf[key] = float(value) / total_count

        return tf

    def get_tfidf(self, title, language, save):  # 按公式计算tf-idf

        tfidf: dict = {}
        for word in self.text:
            idf_value = self.idf.get(word, self.default_idf)
            tf_value = self.tf.get(word, 0)
            tfidf_value = tf_value * idf_value
            tfidf[word] = tfidf_value

        # 根据tf-idf排序，去排名前keyword_num的词作为关键词
        print('%s: ' % title, end='')
        keywords = [key.lower() for key, value in
                    sorted(tfidf.items(), key=functools.cmp_to_key(compare), reverse=True)[:self.keyword_num]]
        tags = map_keywords(language, keywords)  # 将关键词归类得到tags
        print(' / '.join(' '.join(tags).split()))  # 输出归类标签

        ################## test
        # with open(file=''.join([os.getcwd(), '/result/', language, '_TFIDF_tags_modified.txt']), mode='a', encoding='utf-8') as tags_file:
        #     tags_file.write(' '.join(['tags:', ' '.join(tags), '||| keywords:', ' '.join(keywords)]))
        #     tags_file.write('\n')
        #     if ' '.join(tags).strip() == '':
        #         with open(file=''.join([os.getcwd(), '/result/', language, '_unknown.txt']), mode='a', encoding='utf-8') as tags_file:
        #             tags_file.write(' '.join(keywords))
        #             tags_file.write('\n')
        ##################

        if save:  # 保存关键词
            with open(file=''.join([os.getcwd(), '/result/', language, '_tags_TFIDF.txt']), mode='a',
                      encoding='utf-8') as tags_file:
                tags_file.write(' '.join(keywords))  # 考虑是保存keywords（存在新的词语）还是tags（只会出现表中的词语）
                tags_file.write('\n')


class TopicModel(object):  # 参考书上的主题模型算法

    # 三个传入参数：处理后的数据集，关键词数量，具体模型（LSI、LDA），主题数量
    def __init__(self, corpus, keyword_num, model='LDA', num_topics=4):

        # 使用gensim的接口，将文本转为向量化表示
        self.dictionary = corpora.Dictionary(corpus)  # 先构建词空间
        corpus_vec = [self.dictionary.doc2bow(line) for line in corpus]  # 使用BOW模型向量化

        # 对每个词，根据tf-idf进行加权，得到加权后的向量表示
        self.tfidf_model = models.TfidfModel(corpus_vec)
        self.corpus_tfidf = self.tfidf_model[corpus_vec]

        self.keyword_num = keyword_num
        self.num_topics = num_topics

        # 选择加载的模型
        if model == 'LSI':
            self.model = self.train_lsi()
        else:
            self.model = self.train_lda()

        # 得到数据集的主题-词分布
        words: dict = self.word_dictionary(corpus)
        self.wordtopic_dic = self.get_wordtopic(words)

    def word_dictionary(self, corpus):  # 词空间构建方法和向量化方法，在没有gensim接口时的一般处理方法

        dictionary = []
        for line in corpus:
            dictionary.extend(line)

        return list(set(dictionary))

    def train_lsi(self):  # lsi模型
        return models.LsiModel(self.corpus_tfidf, id2word=self.dictionary, num_topics=self.num_topics)

    def train_lda(self):  # lda模型
        return models.LdaModel(self.corpus_tfidf, id2word=self.dictionary, num_topics=self.num_topics)

    def get_wordtopic(self, words):

        wordtopic_dic = {}
        for word in words:
            single_list = [word]
            wordcorpus = self.tfidf_model[self.dictionary.doc2bow(single_list)]
            wordtopic = self.model[wordcorpus]
            wordtopic_dic[word] = wordtopic

        return wordtopic_dic

    def get_simword(self, text, language, save):  # 计算词的分布和文档的分布的相似度，取相似度最高的keyword_num个词作为关键词

        for title, content in sorted(text.items(), key=lambda x: x[0]):

            sentcorpus = self.tfidf_model[self.dictionary.doc2bow(content)]
            senttopic = self.model[sentcorpus]

            def calsim(l1, l2):  # 余弦相似度计算
                a, b, c = 0.0, 0.0, 0.0
                for t1, t2 in zip(l1, l2):
                    x1 = t1[1]
                    x2 = t2[1]
                    a += x1 * x1
                    b += x1 * x1
                    c += x2 * x2
                sim = a / math.sqrt(b * c) if not (b * c) == 0.0 else 0.0
                return sim

            # 计算输入文本和每个词的主题分布相似度
            sim_dic = {}
            for key, value in self.wordtopic_dic.items():
                if key not in content:
                    continue
                sim = calsim(value, senttopic)
                sim_dic[key] = sim

            print('%s: ' % title, end='')
            keywords = [key.lower() for key, value in
                        sorted(sim_dic.items(), key=functools.cmp_to_key(compare), reverse=True)[:self.keyword_num]]
            tags = map_keywords(language, keywords)  # 此处同tfidf模型
            print(' / '.join(' '.join(tags).split()))  # 得到归类标签

            #################### test
            #   with open(file=''.join([os.getcwd(), '/result/', language, '_LDA_tags_modified.txt']), mode='a', encoding='utf-8') as tags_file:
            #       tags_file.write(' '.join(['tags:', ' '.join(tags), '||| keywords:', ' '.join(keywords)]))
            #       tags_file.write('\n')
            ####################

            if save:  # 保存标签
                with open(file=''.join([os.getcwd(), '/result/', language, '_tags_LDA.txt']), mode='a',
                          encoding='utf-8') as tags_file:
                    tags_file.write(' '.join(keywords))  # 同样需要注意储存问题
                    tags_file.write('\n')


def run():  # 中文、英文、日文，先实例化再调用不同算法提取关键词

    print('Chinese tag extraction:')    # 中文
    chinese_model = TagExtraction('chinese', load_from_saved=False, test=False, save=False)
    chinese_model.tfidf_extract(keyword_num=20)
    # chinese_model.topic_extract(model='LDA'， keyword_num=20)    # 不稳定，每次结果不一样，不建议使用

    # print('\n')

    print('English tag extraction:')  # 英文
    english_model = TagExtraction('english', load_from_saved=False, test=False, save=False)
    english_model.tfidf_extract(keyword_num=20)  # 使用spacy，与pke的区别：存在sentence和word的filter，不需要额外读取，需要取消相关注释
    # english_model.topic_extract(model='LDA', keyword_num=20)    # 结果可能会不稳定
    # english_model.pke_tfidf_extract()    # 使用pke，弃用
    # english_model.pke_topic_extract('LDA')    # 使用pke，弃用

    # print('\n')

    # print('Japanese tag extraction:')    # 日文，暂未完善
    # japanese_model = TagExtraction('japanese', load_from_saved=True, test=False, save=False)
    # japanese_model.tfidf_extract(keyword_num=20)
    # japanese_model.topic_extract(model='LDA', keyword_num=20)

if __name__ == "__main__":
    start = time.time()
    run()  # 主程序入口
    end = time.time()
    print('\nDuration: %.3fs' % (end - start))  # 程序运行时间
