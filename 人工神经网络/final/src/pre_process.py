import json
import re
import jieba
import nltk
from nltk.tokenize import word_tokenize


# 读取jsonl文件
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data


# 提取需要的字段
def extract_fields(data):
    en_texts = [item['en'] for item in data]
    zh_texts = [item['zh'] for item in data]
    return en_texts, zh_texts


# 去除标点符号和拆分连续数字的函数
def remove_punctuation_and_split_digits(text, lang='en'):
    if lang == 'en':
        # 删除英文标点符号
        text = re.sub(r'[.,?!–\-":;()"’“”\'·/‘’\[\]…`¾]', '', text)
    else:
        # 对于括号加英文的形式进行删除
        text = re.sub(r'\([A-Za-z\s]+\)', '', text)
        text = re.sub(r'（[A-Za-z\s]+）', '', text)
        # 删除中文标点符号
        text = re.sub(r'[。？，！–\-“”：；（()）、—《》<>·,:;?!./‘’…\[\]"¾]', '', text)
        # 在百分号前加空格
        text = re.sub(r'%', ' %', text)

    # 拆分连续数字
    text = re.sub(r'(\d)', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text)  # 将多个空格合并为一个空格

    return text


# 分词函数
def tokenize(text, lang='en'):
    if lang == 'en':
        tokens = word_tokenize(text)
    else:
        tokens = jieba.lcut(text)
    result = ' '.join(tokens)
    result = re.sub(r'\s+', ' ', result)  # 将多个空格合并为一个空格
    return result


# 保存处理后的数据为txt文件，使用@@分割句子对
def save_to_txt(en_texts, zh_texts, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for en_text, zh_text in zip(en_texts, zh_texts):
            # 转化为小写
            en_text = en_text.lower()
            zh_text = zh_text.lower()
            # 去除标点符号和拆分连续数字
            en_text = remove_punctuation_and_split_digits(en_text, lang='en')
            zh_text = remove_punctuation_and_split_digits(zh_text, lang='zh')
            # 分词
            en_text = tokenize(en_text, lang='en')
            zh_text = tokenize(zh_text, lang='zh')
            file.write(f'{en_text}@@{zh_text}\n')


# 主函数
def preprocess_jsonl(file_path, output_file):
    data = read_jsonl(file_path)
    en_texts, zh_texts = extract_fields(data)
    save_to_txt(en_texts, zh_texts, output_file)

load_data = 'train_100k.jsonl'
output_file = 'eng-zh-100k.txt'

# 将原始数据集转化为中英文txt文件，使用@@分割句子对
preprocess_jsonl(f'data/raw_data/{load_data}', f'data/sentence_pairs/{output_file}')