import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas
import spacy
from spacy.lang.en import English
from spacy.lang.de import German
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm_notebook
import random
from collections import Counter
import unicodedata
import re
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from torch.utils.tensorboard import SummaryWriter

if torch.cuda.is_available:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(device)

def load_top_k_zh_embeddings(glove_file_path, k):
    embeddings_index = {}
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= k:  # 只读取前k个词
                break
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

eng_embeddings = load_top_k_zh_embeddings('data/word2vec/tencent-ailab-embedding-en-d200-v0.1.0-s.txt', k=500000)

def load_top_k_zh_embeddings(tencent_file_path, k=500000):
    embeddings_index = {}
    with open(tencent_file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= k:  # 只读取前k个词
                break
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

zh_embeddings = load_top_k_zh_embeddings('data/word2vec/tencent-ailab-embedding-zh-d200-v0.2.0-s.txt', k=500000)


# 定义起始和结束标记的索引
SOS_token = 0  # Start of Sentence
EOS_token = 1  # End of Sentence
UNK_token = 2  # Unknown word

class Lang:
    def __init__(self, name):
        self.name = name  # 语言的名称
        self.word2index = {}  # 单词到索引的映射
        self.word2count = {}  # 单词出现次数的统计
        self.index2word = {0: "<SOS>", 1: "<EOS>", 2:"<UNK>"}  # 索引到单词的映射，初始包含SOS和EOS以及UNK
        self.n_words = 3  # 单词数目初始化为3

    def addSentence(self, sentence):
        # 将句子拆分成单词，并逐个添加到字典中
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        # 如果单词不在字典中，添加该单词
        if word not in self.word2index:
            self.word2index[word] = self.n_words  # 给单词分配一个新的索引
            self.word2count[word] = 1  # 初始化单词的计数为1
            self.index2word[self.n_words] = word  # 将新索引和单词添加到索引到单词的映射中
            self.n_words += 1  # 增加单词数目
        else:
            self.word2count[word] += 1  # 如果单词已存在，增加该单词的计数


"""读取指定语言对的文本文件，创建相应的语言对象，并返回这些对象和句子对"""


def readLangs(lang1, lang2, reverse=False, text_type='train'):
    print("Reading lines...")

    # 读取文件并按行分割
    lines = open('data/sentence_pairs/%s-%s-%s.txt' % (lang1, lang2, text_type), encoding='utf-8'). \
        read().strip().split('\n')

    # 将每一行分割成对并进行规范化
    pairs = [[s for s in l.split('@@')] for l in lines]  # 将每一行按照制表符 (\t) 分割成句子对,并且每个句子对都通过标准化处理

    # 如果需要反转语言对，则反转每对句子，并交换语言对象
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

"""过滤句子对，以确保它们符合一定的条件。这些条件包括句子长度和句子前缀。"""

# 定义最大句子长度
MAX_LENGTH = 50

def filterPair(p):
    # 检查句子对是否符合长度限制
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    # 对所有句子对进行过滤，只保留符合条件的句子对
    return [pair for pair in pairs if filterPair(pair)]


"""定义了prepareData函数，函数通过读取文件、过滤句子对并统计词汇，最终返回处理后的语言对象和句子对。"""


def prepareData(lang1, lang2, reverse=False, text_type='train'):
    # 读取语言对文件并创建语言对象
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse, text_type)
    print("Read %s sentence pairs" % len(pairs))

    # 过滤句子对，只保留符合条件的句子对
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))

    # 统计词汇
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    # 输出词汇统计结果
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)

    # 返回处理后的语言对象和句子对
    return input_lang, output_lang, pairs

# 准备英语到中文的数据，语言对顺序反转
input_lang, output_lang, pairs = prepareData('eng', 'zh', True, '100k')

# 根据eng_embeddings创建词向量字典，在output_lang的词汇表中查找对应的词向量，不在的用随机初始化的词向量代替
output_embeddings_dim = eng_embeddings['1'].shape[0]
output_lang_embeddings = {}
# 为SOS和EOS添加词向量
output_lang_embeddings['<SOS>'] = np.random.uniform(-np.sqrt(1/output_embeddings_dim), np.sqrt(1/output_embeddings_dim), output_embeddings_dim)
output_lang_embeddings['<EOS>'] = np.random.uniform(-np.sqrt(1/output_embeddings_dim), np.sqrt(1/output_embeddings_dim), output_embeddings_dim)
output_lang_embeddings['<UNK>'] = np.random.uniform(-np.sqrt(1/output_embeddings_dim), np.sqrt(1/output_embeddings_dim), output_embeddings_dim)
# 遍历output_lang的所有token，在eng_embeddings中查找对应的词向量
for word in output_lang.word2index.keys():
    if word in eng_embeddings:
        output_lang_embeddings[word] = eng_embeddings[word]
    else:
        # 在-sqrt(1/dim)和sqrt(1/dim)之间均匀采样
        output_lang_embeddings[word] = np.random.uniform(-np.sqrt(1/output_embeddings_dim), np.sqrt(1/output_embeddings_dim), output_embeddings_dim)

# 根据zh_embeddings创建词向量字典，在input_lang的词汇表中查找对应的词向量，不在的用随机初始化的词向量代替
input_embeddings_dim = zh_embeddings['1'].shape[0]
input_lang_embeddings = {}
# 为SOS和EOS添加词向量
input_lang_embeddings['<SOS>'] = np.random.uniform(-np.sqrt(1/input_embeddings_dim), np.sqrt(1/input_embeddings_dim), input_embeddings_dim)
input_lang_embeddings['<EOS>'] = np.random.uniform(-np.sqrt(1/input_embeddings_dim), np.sqrt(1/input_embeddings_dim), input_embeddings_dim)
input_lang_embeddings['<UNK>'] = np.random.uniform(-np.sqrt(1/input_embeddings_dim), np.sqrt(1/input_embeddings_dim), input_embeddings_dim)
# 遍历input_lang的所有token，在zh_embeddings中查找对应的词向量
for word in input_lang.word2index.keys():
    if word in zh_embeddings:
        input_lang_embeddings[word] = zh_embeddings[word]
    else:
        # 在-sqrt(1/dim)和sqrt(1/dim)之间均匀采样
        input_lang_embeddings[word] = np.random.uniform(-np.sqrt(1/input_embeddings_dim), np.sqrt(1/input_embeddings_dim), input_embeddings_dim)

# 将词向量字典转换为嵌入矩阵
def create_embeddings_matrix(embeddings, lang):
    embeddings_matrix = np.zeros((lang.n_words, embeddings['1'].shape[0]))
    for word, idx in lang.word2index.items():
        embeddings_matrix[idx] = embeddings[word]
    return torch.FloatTensor(embeddings_matrix)

input_lang_embeddings_matrix = create_embeddings_matrix(input_lang_embeddings, input_lang)
output_lang_embeddings_matrix = create_embeddings_matrix(output_lang_embeddings, output_lang)


# 编码器，将输入序列编码为隐藏状态，供解码器进一步处理
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_matrix, num_layers=2, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 使用预训练的词向量初始化嵌入层
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)

        # GRU层，用于处理嵌入向量序列
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True,
                          dropout=dropout_p if num_layers > 1 else 0)

        # Dropout层，防止过拟合
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        # 将输入单词索引通过嵌入层和Dropout层
        embedded = self.dropout(self.embedding(input))

        # 将嵌入向量序列输入GRU层，得到输出和隐藏状态
        output, hidden = self.gru(embedded)

        # 返回GRU的输出和最后的隐藏状态
        return output, hidden

"""注意力机制类"""

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)  # 定义线性层Wa
        self.Ua = nn.Linear(hidden_size, hidden_size)  # 定义线性层Ua
        self.Va = nn.Linear(hidden_size, 1)  # 定义线性层Va

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))  # 计算注意力分数
        scores = scores.squeeze(2).unsqueeze(1)  # 调整scores的形状

        weights = F.softmax(scores, dim=-1)  # 计算注意力权重
        context = torch.bmm(weights, keys)  # 计算上下文向量

        return context, weights  # 返回上下文向量和注意力权重

"""带注意力机制的解码器"""
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, embedding_matrix, num_layers=2, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)  # 定义嵌入层
        self.attention = BahdanauAttention(hidden_size)  # 定义注意力机制
        self.gru = nn.GRU(2 * hidden_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout_p if num_layers > 1 else 0)  # 定义GRU层
        self.out = nn.Linear(hidden_size, output_size)  # 定义输出线性层
        self.dropout = nn.Dropout(dropout_p)  # 定义Dropout层

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)  # 获取批次大小
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)  # 初始化解码器的输入为SOS_token
        decoder_hidden = encoder_hidden  # 初始化解码器的隐藏状态为编码器的隐藏状态
        decoder_outputs = []  # 用于存储解码器的输出
        attentions = []  # 用于存储注意力权重

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(   # 进行一步解码
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)  # 将解码输出添加到列表中
            attentions.append(attn_weights)  # 将注意力权重添加到列表中

            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # 使用自己的预测作为下一步的输入

        decoder_outputs = torch.cat(decoder_outputs, dim=1)  # 将所有时间步的输出连接起来
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)  # 对输出进行log softmax处理
        attentions = torch.cat(attentions, dim=1)  # 将所有时间步的注意力权重连接起来

        return decoder_outputs, decoder_hidden, attentions  # 返回输出、隐藏状态和注意力权重

    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))  # 将输入通过嵌入层并进行Dropout

        query = hidden[-1].unsqueeze(0).permute(1, 0, 2)  # 调整隐藏状态的形状以适应注意力机制，仅使用最后一层的隐藏状态
        context, attn_weights = self.attention(query, encoder_outputs)  # 计算上下文向量和注意力权重
        input_gru = torch.cat((embedded, context), dim=2)  # 将嵌入向量和上下文向量连接起来

        output, hidden = self.gru(input_gru, hidden)  # 通过GRU层
        output = self.out(output)  # 通过线性层

        return output, hidden, attn_weights  # 返回输出、隐藏状态和注意力权重


def indexesFromSentence(lang, sentence):
    # 将句子中的每个单词转换为相应的索引
    ret = []
    for word in sentence.split(' '):
        if word in lang.word2index:
            ret.append(lang.word2index[word])
        else:
            ret.append(UNK_token)  # 如果单词不在词汇表中，用UNK_token代替

    return ret


def tensorFromSentence(lang, sentence):
    # 将句子转换为索引并附加EOS标记，然后转换为张量
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1,
                                                                       -1)  # 该函数将句子转换为索引列表，并附加结束标记 EOS_token，然后将索引列表转换为张量，并返回形状为 (1, 句子长度) 的张量。


def tensorsFromPair(pair):
    # 将句子对中的两个句子分别转换为张量
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)  # 该函数将句子对转换为张量对，并返回输入张量和目标张量。

"""data loader"""

def get_dataloader(batch_size, text_type='train'):
    # 准备数据并创建DataLoader
    input_lang, output_lang, pairs = prepareData('eng', 'zh', True, text_type)
    # 调用prepareData函数，获取输入语言对象、输出语言对象和句子对列表

    n = len(pairs)
    # 获取句子对的数量

    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    # 初始化用于存储输入和目标句子索引的数组，形状为（句子对数量, 最大句子长度）

    for idx, (inp, tgt) in enumerate(pairs):        # idx: 索引，inp: 输入句子，tgt: 目标句子
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        # 将每个句子转换为索引列表

        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        # 在每个索引列表末尾添加结束标记EOS_token

        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids
        # 将索引列表填充到数组中

    # 创建TensorDataset对象，将输入和目标数组转换为张量并存储在设备上
    train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                               torch.LongTensor(target_ids).to(device))

    # 创建随机采样器
    train_sampler = RandomSampler(train_data)

    # 创建DataLoader对象，用于批量加载数据
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # 返回输入语言对象、输出语言对象和DataLoader对象
    return input_lang, output_lang, train_dataloader

"""训练函数"""

def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
                decoder_optimizer, criterion):
    total_loss = 0  # 初始化总损失

    for data in dataloader:
        input_tensor, target_tensor = data  # 获取输入和目标张量

        encoder_optimizer.zero_grad()  # 清零编码器的梯度
        decoder_optimizer.zero_grad()  # 清零解码器的梯度

        encoder_outputs, encoder_hidden = encoder(input_tensor)  # 前向传播，通过编码器
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)  # 前向传播，通过解码器

        # 计算损失
        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()  # 反向传播计算梯度

        encoder_optimizer.step()  # 更新编码器的参数
        decoder_optimizer.step()  # 更新解码器的参数

        total_loss += loss.item()  # 累加损失

    return total_loss / len(dataloader)  # 返回平均损失

"""计算时间"""

import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001,
          print_every=1, test_every=5):
    start = time.time()  # 记录训练开始时间
    plot_losses = []  # 用于存储绘图用的损失值
    print_loss_total = 0  # 每个print_every轮重置
    plot_loss_total = 0  # 每个plot_every轮重置

    # 初始化优化器
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()  # 定义损失函数

    min_loss = float('inf')  # 初始化最小损失值

    for epoch in range(1, n_epochs + 1):
        # 训练一个epoch
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss  # 累加当前epoch的损失
        plot_loss_total += loss  # 累加当前epoch的损失

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every  # 计算平均损失
            print_loss_total = 0  # 重置损失累计
            # 打印当前时间、epoch数、进度百分比和平均损失
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                         epoch, epoch / n_epochs * 100, print_loss_avg))
            writer.add_scalar('loss', loss, epoch)  # 将损失写入TensorBoard
            if loss < min_loss:
                min_loss = loss
                torch.save(encoder, f'saved_models/{running_model}/encoder_model.pth')
                torch.save(decoder, f'saved_models/{running_model}/decoder_model.pth')
                print('---------------------save model---------------------')

        if epoch % test_every == 0:
            # 每隔test_every轮评估一次模型
            bleu_scores, avg_bleu_score = evaluate_bleu_sacrebleu(encoder, decoder, test_pairs, input_lang, output_lang)
            print('BLEU Score: %.4f' % avg_bleu_score)
            writer.add_scalar('BLEU Score', avg_bleu_score, epoch)  # 将BLEU分数写入TensorBoard

"""绘制损失曲线"""

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        # 将输入句子转换为张量
        input_tensor = tensorFromSentence(input_lang, sentence)

        # 前向传播，通过编码器
        encoder_outputs, encoder_hidden = encoder(input_tensor)

        # 前向传播，通过解码器，不使用目标张量
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        # 从解码器输出中选择概率最高的单词索引
        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        # 将索引转换为单词
        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')  # 如果遇到结束标记，则停止翻译
                break
            decoded_words.append(output_lang.index2word[idx.item()])

    # 返回翻译的单词列表和注意力权重
    return decoded_words, decoder_attn

def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, _ = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        # 将输入句子转换为张量
        input_tensor = tensorFromSentence(input_lang, sentence)

        # 前向传播，通过编码器
        encoder_outputs, encoder_hidden = encoder(input_tensor)

        # 前向传播，通过解码器，不使用目标张量
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        # 从解码器输出中选择概率最高的单词索引
        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        # 将索引转换为单词
        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')  # 如果遇到结束标记，则停止翻译
                break
            decoded_words.append(output_lang.index2word[idx.item()])

    # 返回翻译的单词列表和注意力权重
    return decoded_words, decoder_attn

def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, _ = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


import sacrebleu


def evaluate_bleu_sacrebleu(encoder, decoder, pairs, input_lang, output_lang):
    bleu_scores = []

    for pair in pairs:
        input_sentence = pair[0]
        target_sentence = pair[1]
        output_words, _ = evaluate(encoder, decoder, input_sentence, input_lang, output_lang)
        candidate = " ".join(output_words[:-1])
        references = [target_sentence]
        score = sacrebleu.corpus_bleu([candidate], [[ref] for ref in references])
        bleu_scores.append(score.score / 100)  # 转换成小数制

    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
    return bleu_scores, avg_bleu_score



batch_size = 32
running_model = 'layer2_PreVec_100k'
writer = SummaryWriter(log_dir=f'log/{running_model}')
test_pairs = readLangs('eng', 'zh', True, 'test')[2]
input_lang, output_lang, train_dataloader = get_dataloader(batch_size, '100k')

encoder = EncoderRNN(input_lang.n_words, input_embeddings_dim, input_lang_embeddings_matrix).to(device)
decoder = AttnDecoderRNN(output_embeddings_dim, output_lang.n_words, output_lang_embeddings_matrix).to(device)

# 加载模型
load_model = 'layer2_PreVec_100k'
encoder = torch.load(f'saved_models/{layer2_PreVec_100k}/encoder_model.pth')
decoder = torch.load(f'saved_models/{layer2_PreVec_100}/decoder_model.pth')

encoder.eval()
decoder.eval()
evaluateRandomly(encoder, decoder)

# 读取语言对文件并创建语言对象
test_pairs = readLangs('eng', 'zh', True, 'valid')[2]
# 计算BLEU分数
bleu_scores, avg_bleu_score = evaluate_bleu_sacrebleu(encoder, decoder, test_pairs, input_lang, output_lang)