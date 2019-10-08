import math  # 用于数学计算，如计算相似度
import html  # 用于处理html转义字符
import pandas as pd  # 数据处理包pandas
import jieba  # 中文分词
import jieba.posseg as psg
from Models.Longhash.ContentNewsModel import ContentNewsModel
from Util.ConfigParser import config
from Util.Env import env
from Util.Redis import Redis
import collections
import json

class TagExtraction(object):
    __language = 'chinese'
    __userdict = None
    __corpus = None
    __text = None
    __id = 518
    LH_NEWS_TAG_CN = 'lh:news:tag:cn'

    def handle(self):
        return self.tfidf_extract(20)

    @property
    def userdict(self):
        if self.__userdict is None:
            self.__userdict = jieba.load_userdict(self.userdict_path())
        return self.__userdict

    @property
    def corpus(self):   # 读取语料库
        if self.__corpus is None:
            temp = Redis.get(self.LH_NEWS_TAG_CN)
            if temp:
                self.__corpus = json.loads(temp)
            else:
                self.__corpus = self.format_content(self.get_data())
                Redis.set(self.LH_NEWS_TAG_CN, json.dumps(self.__corpus))
                Redis.expire(self.LH_NEWS_TAG_CN, 86400 * 7)
        return self.__corpus

    @property
    def text(self):
        temp = ContentNewsModel().where('id', self.__id).select('title', 'shorttitle', 'summary', 'content').take(1).data()
        return self.format_content(temp)

    def get_data(self):
        return ContentNewsModel().where('type', 1).select('title', 'shorttitle', 'summary', 'content').take(5).data()

    def format_content(self, data):
        data['content'] = data['content'].map(lambda x: html.unescape(str(x)))  # 转换html转义字符
        data.fillna('', inplace=True)  # 将缺失值转化为空串
        func = lambda x: '.'.join([x['title'], x['shorttitle'], x['summary'], x['content']])
        corpus_line = data.apply(func, axis='columns')  # 将四列合并为一列，便于处理
        corpus: list = []
        for num, line in enumerate(corpus_line):  # 遍历语料库中每一篇文章
            content = line.strip()  # 去掉前后的空格
            sentence = self.sentence_filter(content)  # 清洗句子
            words = self.word_filter(sentence)  # 对词组进行词性筛选
            corpus.append(words)  # 将词组存入corpus中
        del data
        return [line for line in corpus if line != []]  # 将空行删掉

    def userdict_path(self):
        return env.userdict_path + '{}_userdict.txt'.format(self.__language)

    def sentence_filter(self, sentence):
        return psg.cut(sentence)

    def word_filter(self, sentence):  # 筛选词语的词性和构成
        words: list = []
        for word in sentence:
            if not word.flag.startswith('n'):  # 只保留名词，可以额外添加名词（在userdict中用'n'标注）
                continue
            if word.word not in config('stopwords.chinese') and len(word.word) > 1:  # 去除停用词和单个中文字
                words.append(word.word)
        return words

    def tfidf_extract(self, keyword_num=10):  # 使用tfidf算法，默认为10个关键词
        self.idf, self.default_idf = self.train_idf(self.corpus)  # 得到idf值
        self.keyword_num = keyword_num
        for index in self.text:  # 排序文章路径
            self.tf = self.get_tf(index)  # 处理后的待提取文本
            return [i for i in self.get_tfidf(index) if i != '']  # 使用TFIDF类的方法，得到tfidf关键词

    def train_idf(self, corpus) -> (dict, float):  # 计算idf值
        idf: dict = {}
        total_count: int = len(corpus)  # 总文档数
        for line in corpus:  # 每个词出现的文档数
            for word in set(line):
                idf[word] = idf.get(word, 0.0) + 1.0
        for key, value in idf.items():  # 按公式转换为idf值，分母加1进行平滑处理
            idf[key] = math.log(total_count / (1.0 + value))
        default_idf = math.log(total_count / (1.0))  # 对于没有在字典中的词，默认其仅在一个文档出现，得到默认idf值
        return idf, default_idf

    def get_tf(self, content) -> dict:  # 统计tf值
        tf: dict = {}
        for word in content:
            tf[word] = tf.get(word, 0.0) + 1.0
        total_count = len(content)
        for key, value in tf.items():
            tf[key] = float(value) / total_count
        return tf

    def get_tfidf(self, content):  # 按公式计算tf-idf
        tfidf = {index: self.get_tfidf_value(index) for index in content}
        keywords = [i for i, v in collections.Counter(tfidf).most_common()][:20]
        tags = self.map_keywords(keywords)  # 将关键词归类得到tags
        return tags

    def get_tfidf_value(self, word):
        idf_value = self.idf.get(word, self.default_idf)
        tf_value = self.tf.get(word, 0)
        return tf_value * idf_value

    def get_alternative_tags_path(self):
        return env.alternative_tags_path + '{}_alternative_tags.xlsx'.format(TagExtraction.__language)

    def map_keywords(self, keywords):  # 将算法得到的关键词归类为具体标签
        mapping_list = pd.read_excel(io=self.get_alternative_tags_path())
        mapping = {
            word: record['Tag'] for record in mapping_list.to_dict('records') for word in record.values() if
            not pd.isnull(word) and word != record['Tag']
        }
        tags = [mapping.get(word) for word in keywords if mapping.get(word) != None]
        tags = sorted(list(set(tags)), key=tags.index)  # 标签有序去重，保留重要程度排序
        return ['区块链'] if tags == [''] else tags


if __name__ == "__main__":
    print(TagExtraction().handle())
    exit(0)
