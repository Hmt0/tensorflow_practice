from hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
import codecs # 编码
import regex # 正则表达式

def load_de_vocab():
    # 发生大于min_cnt的单词加入vocab
    vocab = [line.split()[0] for line in codecs.open('data/de.vocab.tsv', 'r', 'utf-8').read().splitlines()
             if int(line.split()[1]) >= hp.min_cnt]
    # 对单词编号,顺序排列
    word2idx = {word:idx for idx,word in enumerate(vocab)}
    idx2word = {idx:word for idx,word in enumerate(vocab)}

    return word2idx,idx2word

def load_en_vocab():
    # 英文
    vocab = [line.split()[0] for line in codecs.open('data/en.vocab.tsv','r','utf-8').read().splitlines()
             if int(line.split()[1])>=hp.min_cnt]

    word2idx = {word:idx for idx,word in enumerate(vocab)}
    idx2word = {idx:word for idx,word in enumerate(vocab)}
    return word2idx,idx2word



def create_data(source_sents,target_sents):
    de2idx,idx2de = load_de_vocab() #读取德文
    en2idx,idx2en = load_en_vocab() #读取英文

    x_list ,y_list,Sources,Targets = [],[],[],[]
    for source_sent,target_sent in zip(source_sents,target_sents):
        # 对句子进行编码
        x = [de2idx.get(word,1) for word in (source_sent+u" </S>").split()] # 1: OOV, </S>: End of Text
        y = [en2idx.get(word,1) for word in (target_sent+u" </S>").split()]

        if max(len(x),len(y)) <= hp.maxlen: #句子中单词的最大数限制
            x_list.append(np.array(x))
            y_list.append(np.array(y))
            Sources.append(source_sent)
            Targets.append(target_sent)

    #Pad
    X = np.zeros([len(x_list),hp.maxlen],np.int32) # 第一维：数据量；第二维：句子长度
    Y = np.zeros([len(y_list),hp.maxlen],np.int32)

    for i,(x,y) in enumerate(zip(x_list,y_list)): # (x, y) = （源句， 目标句）
        X[i] = np.lib.pad(x,[0,hp.maxlen-len(x)],'constant',constant_values=(0,0)) # 填充
        Y[i] = np.lib.pad(y,[0,hp.maxlen-len(y)],'constant',constant_values=(0,0))
    return X,Y,Sources,Targets



def load_train_data():
    def _refine(line):
        line = regex.sub("[^\s\p{Latin}']", "", line) # 替换
        return line.strip()

    de_sents = [_refine(line) for line in codecs.open(hp.source_train, 'r', 'utf-8').read().split('\n') if
                line and line[0] != "<"]
    en_sents = [_refine(line) for line in codecs.open(hp.target_train, 'r', 'utf-8').read().split('\n') if
                line and line[0] != '<']

    X, Y, Sources, Targets = create_data(de_sents, en_sents)
    # X, Y为编码后的句子
    return X, Y


def load_test_data():
    def _refine(line):
        line = regex.sub("<[^>]+>", "", line)
        line = regex.sub("[^\s\p{Latin}']", "", line)
        return line.strip() #去除头尾指定字符

    de_sents = [_refine(line) for line in codecs.open(hp.source_test,'r','utf-8').read().split('\n') if line and line[:4] == "<seg"]
    # 测试数据最后一列为seg
    en_sents = [_refine(line) for line in codecs.open(hp.target_test,'r','utf-8').read().split('\n') if line and line[:4] == '<seg']

    X,Y,Sources,Targets = create_data(de_sents,en_sents)
    return X,Sources,Targets



def get_batch_data():
    X, Y = load_train_data()

    num_batch = len(X) // hp.batch_size

    #print("train_X:\n", X[:10])
    #print("train_Y:\n", Y[:10])
    X = tf.convert_to_tensor(X,tf.int32)
    Y = tf.convert_to_tensor(Y,tf.int32)
    # print(X)
    # print(Y)
    '''
    input_queues = tf.train.slice_input_producer([X,Y])
    '''
    input_queues = tf.data.Dataset.from_tensor_slices((X,Y))

    # print(list(input_queues.as_numpy_iterator()))
    # print("shape of input_queues:", input_queues)
    '''
    i = 0
    for element in input_queues.as_numpy_iterator():
        print(i, element)
        i += 1
        if i == 10:
            break
    '''
    '''
    x,y = tf.train.shuffle_batch(input_queues,
                                 num_threads=8,
                                 batch_size=hp.batch_size,
                                 capacity = hp.batch_size*64,
                                 min_after_dequeue=hp.batch_size * 32,
                                 allow_smaller_final_batch=False)
    '''
    dataset = tf.data.Dataset.shuffle(input_queues, buffer_size=hp.batch_size * 64).batch(batch_size=hp.batch_size)
    # dataset = input_queues.shuffle(hp.batch_size * 64).batch(hp.batch_size)

    # print(tf.compat.v1.data.get_output_shapes(input_queues))
    # print(tf.compat.v1.data.get_output_types(input_queues))

    iterator = tf.compat.v1.data.make_one_shot_iterator(input_queues)
    x, y = iterator.get_next()
    # print(x, y)
    return x, y, num_batch

def main():
    x, y, num_batch = get_batch_data()

if __name__ == '__main__':
    main()
