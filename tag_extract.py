import datetime  # 用于显示时间
import math  # 用于数学计算，如计算相似度
import os  # 用于系统操作，如获取目录
import re  # 用于处理字符串
import sys  # 用于系统操作
import time  # 用于计时

import functools  # 用于比较关键词重要程度
from gensim import corpora, models  # 调用gensim主题模型接口
import html  # 用于处理html转义字符
import numpy as np  # 数据处理包numpy
import pandas as pd  # 数据处理包pandas

import jieba  # 中文分词
import nltk, spacy  # 英文分词    # import pke 英文关键词提取
# import MeCab    # 日文分词


from Models.Longhash.ContentNewsModel import ContentNewsModel
from Util.ConfigParser import config
from Util.Env import env


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

    # if language == 'english':  # 英文标签匹配尚未完成，直接跳出
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

    if language == 'chinese':
        if tags_pro == ['']:
            return ['区块链']  # 无标签则标记为'区块链'

    if language == 'english':
        if tags_pro == ['']:
            return ['Blockchain']  # 无标签则标记为'区块链'

    return tags_pro


class TagExtraction(object):

    def __init__(self, language, load_from_saved=False, latest=True, save=False):

        # load_from_saved: True表示读取上次程序已读取的语料库，可以节省时间
        # latest: True表示为最新文章打标签
        # save: True表示将得到的关键词存成文件形式

        self.language = language  # 中文/英文/日文
        self.save = save  # 是否将得到的关键词写入文件

        self.userdict = self.get_userdict()  # 自定义词典

        self.corpus = self.get_corpus(load_from_saved)  # 语料库
        self.text = self.get_text(latest)  # 文章
        self.tags = dict()



    def get_userdict(self):
        '''
        # 读取自定义词典，路径为 ./userdict/***_userdict.txt（中文、英文），日文的储存方式为csv
        '''
        if self.language == 'chinese':                                                                  # 返回值为None，直接调用接口加载自定义词典
            return jieba.load_userdict(env.userdict_path + '{}_userdict.txt'.format(self.language))     # 使用jieba的load_userdict方法
        elif self.language == 'english':                                                                # 返回值为存有自定义词典的list
            with open(env.userdict_path + '{}_userdict.txt'.format(self.language), mode='r', encoding='utf-8') as words:
                return [tuple(line.lower().strip().split()) for line in words.readlines()]              # 配合使用nltk的MWE分词器
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

                if word.word not in config('stopwords.chinese') and len(word.word) > 1:  # 去除停用词和单个中文字
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
                if str(word).lower() not in config('stopwords.english') and notadd(str(word)) and len(str(word.lemma_)) > 2:
                    words.append(str(word.lemma_).lower())

            return words

        elif self.language == 'japanese':

            for word, tag in sentence:

                if not tag == '名詞':  # 只保留名词，包括中文、英文和日文，但效果不太好，要补充停用词及自定义词
                    continue

                if word not in config('stopwords.japanese') and notnum(word) and len(word) > 1:  # 去除停用词、含数字以及单字日文词
                    words.append(word)

            return words

    def get_data(self):
        '''
        # 从数据库读取所有Longhash新闻，格式为Dataframe，包括4个Column ['title', 'shorttitle', 'summary', 'content']
        '''
        return ContentNewsModel().where('type', 1).select('title', 'shorttitle', 'summary', 'content').take(5).data()

    def get_corpus(self, load_from_saved):  # 读取语料库
        LOG('Loading corpus', head='\n')
        if load_from_saved:  # 读取上一次程序保存的已处理好的语料库

            # if self.language == 'english':
            #     return ''.join([os.getcwd(), '/corpus/english_text.csv'])    # 使用pke，读取csv格式文件

            with open(file=''.join([os.getcwd(), '/corpus/', self.language, '_corpus_vocabulary.txt']), mode='r',
                      encoding='utf-8') as load:
                return [doc.strip().split() for doc in load.readlines()]  # 从文件中读取已保存的划分好的词组

        else:
            corpus_excel = self.get_data()

            if self.language == 'japanese':  # 日文中含有未处理的html标签，需要清洗掉
                corpus_excel['content'] = corpus_excel['content'].map(lambda x: re.sub('<.+?>', '', x))

            corpus_excel['content'] = corpus_excel['content'].map(lambda x: html.unescape(str(x)))  # 转换html转义字符
            corpus_excel.fillna('', inplace=True)  # 将缺失值转化为空串

            func = lambda x: '. '.join([x['title'], x['shorttitle'], x['summary'], x['content']])
            corpus_line = corpus_excel.apply(func, axis='columns')  # 将四列合并为一列，便于处理

            corpus: list = []
            for num, line in enumerate(corpus_line):                # 遍历语料库中每一篇文章
                print(num + 1, end=' ')
                content = line.strip()                              # 去掉前后的空格
                sentence = self.sentence_filter(content)            # 清洗句子
                # if self.language == 'english':
                #     corpus.append(sentence)                       # 使用pke，直接将句子储存起来，不需要筛选词性
                #     continue
                words = self.word_filter(sentence)                  # 对词组进行词性筛选
                corpus.append(words)                                # 将词组存入corpus中

            corpus_wash = [line for line in corpus if line != []]   # 将空行删掉
            with open(file=env.corpus_path + '{}_corpus_vocabulary.txt'.format(self.language), mode='w', encoding='utf-8') as save:
                # 把语料库保存至corpus文件夹的txt文件中（中文、日文、英文nltk）
                for doc in corpus_wash:
                    save.write(' '.join(doc))
                    save.write('\n')
            return corpus_wash

    def get_text(self, latest):                 # 读取文章（待提取标签的文本）
        LOG('Loading text')
        print("Corpus size: ",len(self.corpus))
        all_text = dict(enumerate(self.corpus))
        if latest:
            return {max(all_text.keys()):self.corpus[max(all_text.keys())]}
        else:                                   # 不进行调试，中文、日文和英文nltk可以直接使用语料库的处理结果
            return all_text  # 字典键值为序号

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
            tags = tfidf_model.get_tfidf(title, self.language)  # 使用TFIDF类的方法，得到tfidf关键词
            self.tags[title] = tags
            return self.tags[title]


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

    def get_tfidf(self, title, language):  # 按公式计算tf-idf

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
        return tags


        ################## test
        # with open(file=''.join([os.getcwd(), '/result/', language, '_TFIDF_tags_modified.txt']), mode='a', encoding='utf-8') as tags_file:
        #     tags_file.write(' '.join(['tags:', ' '.join(tags), '||| keywords:', ' '.join(keywords)]))
        #     tags_file.write('\n')
        #     if ' '.join(tags).strip() == '':
        #         with open(file=''.join([os.getcwd(), '/result/', language, '_unknown.txt']), mode='a', encoding='utf-8') as tags_file:
        #             tags_file.write(' '.join(keywords))
        #             tags_file.write('\n')
        ##################

        #if save:  # 保存关键词
        #    with open(file=''.join([os.getcwd(), '/result/', language, '_tags_TFIDF.txt']), mode='a',
        #              encoding='utf-8') as tags_file:
        #        tags_file.write(' '.join(keywords))  # 考虑是保存keywords（存在新的词语）还是tags（只会出现表中的词语）
        #        tags_file.write('\n')


def run():  # 中文、英文、日文，先实例化再调用不同算法提取关键词

    print('Chinese tag extraction:')  # 中文
    chinese_model = TagExtraction('chinese', load_from_saved=False, latest=True, save=False)
    chinese_model.tfidf_extract(keyword_num=20)

    # print('\n')

    print('English tag extraction:')  # 英文
    english_model = TagExtraction('english', load_from_saved=False, latest=True, save=False)
    english_model.tfidf_extract(keyword_num=20)  # 使用spacy，与pke的区别：存在sentence和word的filter，不需要额外读取，需要取消相关注释

    # print('\n')

    # print('Japanese tag extraction:')    # 日文，暂未完善
    # japanese_model = TagExtraction('japanese', load_from_saved=True, test=False, save=False)
    # japanese_model.tfidf_extract(keyword_num=20)
    # japanese_model.topic_extract(model='LDA', keyword_num=20)


def process(language, latest):
    print(language + ' tag extraction:\n')
    model = TagExtraction(language, latest)
    return model.tfidf_extract(keyword_num=20)


if __name__ == "__main__":
    # print(TagExtraction('english', True).get_stopwords())
    # print(TagExtraction('english', False).get_userdict())
    # exit(0)
    # print(TagExtraction('english', False).get_corpus(False))



    print(process('chinese', True))
    # print(get_data(language=1))
    exit(0)
    #start = time.time()
    #run()  # 主程序入口
    #end = time.time()
    #print('\nDuration: %.3fs' % (end - start))  # 程序运行时间
