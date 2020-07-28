#/usr/bin/python
#-*- coding=utf-8 -*-
'''
CONNECTIONIST TEMPORAL CLASSIFICATION with Constrained Decoding

He Na 2018-8-22
'''


import math
import numpy as np
import pickle
import logging


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

def readTxt(txtpath):
    filelist = []
    with open(txtpath, 'r') as f:
        for line in f.readlines():
            filelist.append(line.strip())
    return filelist

# ==================== absolute requirements ====================
'''
Way1: setting the probability of all sequences that fail to meet the dictionary words to 0;
Way2: setting the probability of all word from sequences that fail to meet the dictionary abc to 0;
'''
def tok(w, s, t):
    pass

def log(val):
    '''return -inf for log(0) instead of throwing error like python implementation does it'''
    if val > 0:
        return math.log(val)
    return float('-inf')

def LMMat2p(w1, w2, WordsDict, LMMat):
    '''
    p(w2|w1)
    '''
    if w2 == ' ':
        return 0.5
    return LMMat[WordsDict.index(w1)][WordsDict.index(w2)]

tok = {}
ans_tok = {}

def s2NetIndex(w, s, classes):
    '''
    segment to net index
    classes: classes of the Net
    '''
    if w[s-1] == ' ':
        return -1
    result = classes.index(w[s-1])
    return result
    
def addBlank(w):
    astr = ' '
    for i in w:
        astr += i + ' '
    return astr

def getMaxSocreTok(TokList):
    socreList = [item[0] for item in TokList]
    return TokList[socreList.index(max(socreList))]

def Net4Top5_bk(net_ans, blackIdx):
    top5_score_list = net_ans[: 5]
    top5_list = [0, 1, 2, 3, 4]
    top5_score = min(top5_score_list)
    top5_index = top5_score_list.index(top5_score)
    
    for i, score in enumerate(net_ans[5: -1]):
        if score > top5_score:
            top5_score_list.remove(top5_score)
            top5_list.remove(top5_list[top5_index])

            pass

def get_WordsDict_classes_index(WordsDict, classes):
    return [classes.index(word) for word in WordsDict]



def Net4Top5(net_ans, blackIdx, word2classes):
    top1, top2, top3, top4, top5 = 0, 0, 0, 0, 0
    top1_index, top2_index, top3_index, top4_index, top5_index = blackIdx, blackIdx, blackIdx, blackIdx, blackIdx
    threshold = 0.001
    for i, item in enumerate(net_ans[: -1]):
        if i not in word2classes:
            continue
        if item < threshold:
            continue
        if item > top1:
            top1 = item
            top1_index = i
            continue
        if item > top2:
            top2 = item
            top2_index = i
            continue
        if item > top3:
            top3 = item
            top3_index = i
            continue
        if item > top4:
            top4 = item
            top4_index = i
            continue
        if item > top5:
            top5 = item
            top5_index = i
            continue
        continue
    ans_top5_score, ans_top5_index = [], []
    tmp = [top1_index, top2_index, top3_index, top4_index, top5_index]
    for i, item  in enumerate([top1, top2, top3, top4, top5]):
        if item == 0:
            break
        ans_top5_score.append(item)
        ans_top5_index.append(tmp[i])
    ans_top5_score.append(net_ans[-1])
    ans_top5_index.append(blackIdx)

    return ans_top5_score, ans_top5_index
    
def get_transitionVec(old_state, new_state, WordsDict, classes, LMMat):
    '''
    得到状态转移向量
    old_state: 1684
    new_state: [1301, 6937, 849, 7039, 2747, 7117]
    '''
    # 找到old_state所对应的中文
    word = classes[old_state]
    # 找LM中word对应的index
    assert(word in WordsDict), word + ' not in WordsDict!!!'
    old_lm_index = WordsDict.index(word)
    new_lm_indexs = [WordsDict.index(classes[i]) for i in new_state]
    
    return [LMMat[old_lm_index][i] for i in new_lm_indexs]

def SoftMax(net_ans):
    tmp_net = [math.exp(i) for i in net_ans]
    sum_exp = sum(tmp_net)
    return [i/sum_exp for i in tmp_net]


def pretreatmentNetMat(NetMat, WordsDict, classes):
    '''
    对网络输出结果先进行预处理，如果网络置信度比较高则不需要进行lm修正
    '''
    T, blackIdx = NetMat.shape
    blackIdx -= 1
    ans_vec = []
    ans_score = []
    threshold = 0.9

    word2classes = get_WordsDict_classes_index(WordsDict, classes)
    
    for t in range(T):
        net_ans = NetMat[t]
        net_ans = SoftMax(net_ans)
        top5_score, top5_index = Net4Top5(net_ans, blackIdx, word2classes)

        if top5_score[-1] > threshold or len(top5_score)== 1:
            if len(ans_vec) == 0 or ans_vec[-1] == blackIdx:
                continue
            ans_vec.append(blackIdx)
            ans_score.append(1)
            continue
        if top5_score[0] > threshold:
            if len(ans_vec) != 0 and ans_vec[-1] == top5_index[0]:
                continue
            ans_vec.append(top5_index[0])
            ans_score.append(1)
            # max_index = top5_index[0]
            continue
        ans_vec.append(top5_index)
        ans_score.append(top5_score)
    
    return ans_vec, ans_score

def get_words(confused_path, confused_prob):
    '''
    获取困惑结构中所有可能的词对
    '''
    if len(confused_path) == 1:
        return confused_path[0], confused_prob[0]
    
    now_state = confused_path[0]
    now_score = confused_prob[0]
    next_state = confused_path[1]
    next_score = confused_prob[1]

    tmp_state = []
    tmp_score = []

    for i, state in enumerate(now_state):
        if str(state).isdigit():
            state = [state]
        for j, nstate in enumerate(next_state):
            if state[-1] != nstate:
                tmp_state.append(state + [nstate])
            else:
                tmp_state.append(state)
            
            tmp_score.append(now_score[i]*next_score[j])
    if len(confused_path) < 3:
        confused_path = [tmp_state]
        confused_prob = [tmp_score]

    confused_path = [tmp_state] + confused_path[2:]
    confused_prob = [tmp_score] + confused_prob[2:]
    return get_words(confused_path, confused_prob)

def del_black(words, probs, blackIdx):
    rwords = []
    rprob = []
    for i, word in enumerate(words):
        if blackIdx not in word:
            rwords.append(word)
            rprob.append(probs[i])
            continue
        rword = []
        for j in word:
            if j == blackIdx:
                continue
            rword.append(j)
        if rword not in rwords:
            rwords.append(rword)
            rprob.append(probs[i])
        else:
            prob_idx = rwords.index(rword)
            rprob[prob_idx] += probs[i]
    return rwords, rprob

def get_lm_prob(word, WordsDict, LMMat, classes):
    if len(word) == 1:
        return 0.5
    if len(word) == 2:
        return LMMat2p(classes[word[0]], classes[word[1]], WordsDict, LMMat)/2
    p = LMMat2p(classes[word[0]], classes[word[1]], WordsDict, LMMat)/2
    return p*get_lm_prob(word[1:], WordsDict, LMMat, classes)
        


def get_best_path(confused_path, confused_prob, WordsDict, classes, LMMat, blackIdx):
    '''
    从一个困惑结构中获取最优路径，困惑结构指的是中间路径都为置信度不高的路径
    不使用viterbi算法，因为项目不确定，并不是词与词之间的转移矩阵
    return 困惑结构的最优，不包含首尾
    '''

    words, prob = get_words(confused_path, confused_prob)
    words, prob = del_black(words, prob, blackIdx)
    
    lm_prob = [get_lm_prob(i, WordsDict, LMMat, classes) for i in words]

    add_lm_prob = list(np.array(prob)*np.array(lm_prob))

    if len(words) == 1:
        return words[0][1:]
    word = words[add_lm_prob.index(max(add_lm_prob))]
    return word[1:]







def CTCPasssingSimpleVersion(NetMat, WordsDict, classes, LMMat):
    '''
    简易版本的lm修正，只有在模型困惑的时候才进行模型修正
    '''
    ans_vec, ans_score = pretreatmentNetMat(NetMat, WordsDict, classes)
    blackIdx = NetMat.shape[1] - 1

    path = []
    i = 0
    while i < len(ans_vec):
        if ans_score[i] == 1:
            if ans_vec[i] == blackIdx:
                i += 1
                continue
            if len(path) != 0 and ans_vec[i] == path[-1]:
                i += 1
                continue
            path.append(ans_vec[i])
            i += 1
            continue
        confused_path = []
        confused_prob = []
        if len(path) != 0:
            confused_path.append([path[-1]])
            confused_prob.append([1])
        confused_path.append(ans_vec[i])
        confused_prob.append(ans_score[i])
        i += 1
        while ans_score[i] != 1 and i < len(ans_vec):
            confused_path.append(ans_vec[i])
            confused_prob.append(ans_score[i])
            i += 1
        if ans_vec[i] == blackIdx and len(confused_path) == 1 and False:
            best_path = confused_path[0][confused_prob[0].index(max(confused_prob[0]))]
            path += [best_path]
            continue
        
        confused_path.append([ans_vec[i]])
        confused_prob.append([ans_score[i]])

        # 如果下一个状态确信且不为空,那么一个困惑结构完成
        if ans_vec[i] != blackIdx:
            best_path = get_best_path(confused_path, confused_prob, WordsDict, classes, LMMat, blackIdx)
            path += best_path
            continue
        else:
            i += 1
            while ans_score[i] != 1 and i < len(ans_vec):
                confused_path.append(ans_vec[i])
                confused_prob.append(ans_score[i])
                i += 1

    if path == []:
        return [blackIdx]
    return str('').join(str(classes[i]) for i in path)
        




def CTCViterbiPassing(WordsDict, NetMat, classes, LMMat, top=5):
    '''
    利用Viterbi算法获取time-step后的最优路径，转移矩阵只考虑bigram

    top: 取网络输出的top几作为计算，其中top数量不包括'空格';
    --> [7117]
    下面以lstm chn网络为例
    time-step = 216
    '''
    # vecBt 记录对于每一个项而言，导致到达该项最大分数的前一个结点的index
    # vecV 到达每一项的最大分数
    threshold = 0.7
    vecV, vecBt = [], []
    T, blackIdx = NetMat.shape
    blackIdx -= 1
    top5_index_map = [] # 为了verterbi回溯的时候可以找到对应网络里的index -> 216*6 最后一列为空白index
    ans_vec = []
    max_index = blackIdx

    if T == 1:
        return [NetMat[0].argsort()[-1]]
    for t in range(T):
        net_ans = NetMat[t]
        net_ans = SoftMax(net_ans)
        # 从网络中获取top5的index & score
        top5_score, top5_index = Net4Top5(net_ans, blackIdx)
        top5_index_map.append(top5_index)

        if top5_score[-1] > threshold:
            ans_vec.append(blackIdx)
            continue
        if top5_score[0] > threshold:
            ans_vec.append(top5_index[0])
            max_index = top5_index[0]
            continue
        
        # 初始化
        if t == 0:
            vecBt.append([0]*(top+1))
            vecV = top5_score
            continue
        

        vecV = top5_score
        
        # 状态转移矩阵
        transitionVec = get_transitionVec(max_index, top5_index[:-1], WordsDict, classes, LMMat)
        
        




def CTCTokenPassing_AG(WordsDict, NetMat, classes, LMMat, usingbigrams=False):
    '''
    implements CTC Token Passing Algorithm as shown by Alex Graves (Dissertation, p67)
    # tok[(w, 0, t)]: 在t时刻到达词w的最高分数的token
    # tok[(w, -1, t)]: 在t时刻离开词w的最高分数的token
    tok[(w, s, t)]
    '''
    T, blankIdx = NetMat.shape 

    # special s index for beginning and end of word
    beg, end = 0, -1

    # Initialisation: 1-9
    logging.info("==================== Initialisation ====================")

    for w in WordsDict:
        tok[(w, 1, 1)] = [log(NetMat[0][-1]), [w]]
        tok[(w, 2, 1)] = [log(NetMat[0][s2NetIndex(w, 1, classes)]), [w]]
        tok[(w, -1, 1)] = tok[(w, 2, 1)] if len(w) == 1 else [float('-inf'), []]
    
    logging.info("--> Initialisation done.")
    # Algorithm: 11-31:
    t = 2 
    while t <= T:
        # 13-16

        # 17-24
        max_s = float('-inf')
        
        for index_w, w in enumerate(WordsDict):
            if usingbigrams:
                tmp_score_list = [tok[(i, -1, t-1)][0] + log(LMMat2p(i, w, WordsDict, LMMat)) for i in WordsDict]
            else:
                tmp_score_list = [tok[(i, -1, t-1)][0] for i in WordsDict]
        
            w_star = WordsDict[tmp_score_list.index(max(tmp_score_list))]
            tok[(w, beg, t)] = [max(tmp_score_list), tok[(w_star, -1, t-1)][1] + [w]]
            # 25-31
            # w'
            primeW = addBlank(w)

            for s in range(len(primeW)):
                s += 1
                if t == 2:
                    if (w, s-1, 1) not in tok:
                        tok[(w, s-1, 1)] = [float('-inf'), []]
                    if (w, s, 1) not in tok:
                        tok[(w, s, 1)] = [float('-inf'), []]
                P = [tok[(w, s, t-1)], tok[(w, s-1, t-1)]]
                if primeW[s-1] != '_' and s > 2 and primeW[s-3] != primeW[s-1]:
                    P.append(tok[(w, s-2, t-1)])
                tok[(w, s, t)] = getMaxSocreTok(P)
                tok[(w, s, t)] = [tok[(w, s, t)][0] + log(NetMat[t-1][s2NetIndex(primeW, s, classes)]), tok[(w, s, t)][1]]
            
            tok[(w, -1, t)] = getMaxSocreTok([tok[(w, len(primeW), t)], tok[(w, len(primeW)-1, t)]])
            ans_tok[(w, -1, t)] = tok[(w, -1, t)]
            if tok[(w, -1, t)][0] > max_s:
                max_s = tok[(w, -1, t)][0]
                print_info = tok[(w, -1, t)]
        logging.info("[%s] %s" %(t, str(print_info)))
        t += 1
    
    # Termination: 34-35
    tmp_score_list = [tok[(i, -1, T)][0] for i in WordsDict]
    w_star = WordsDict[tmp_score_list.index(max(tmp_score_list))]
    return str('').join(i for i in tok[(w_star, -1, T)][1])

def get_ctc_decoder(alist, classes, black_id=7117):
    rlist = [alist[0]]
    for i in range(len(alist)-1):
        if alist[i+1] == alist[i]:
            continue
        rlist.append(alist[i+1])
    rrlist = []
    for i in rlist:
        if i == black_id:
            continue
        rrlist.append(i)
    return str('').join(str(classes[i]) for i in rrlist)

def testTokenPassing():
    fw = open('/work/hena/scripts/ocr/analysis/ctc_lm_ans.txt','a')

    fr = open('/work/hena/scripts/ocr/analysis/fs_lm_info.pkl', 'rb')
    chn_tab_fr = open('/work/hena/scripts/ocr/analysis/chn_tab.pkl', 'rb')
    chn_tab = pickle.load(chn_tab_fr)
    WordsDict, _, LMMat = pickle.load(fr)
    prob = np.load('/work/hena/scripts/ocr/analysis/prob.npy')
    index_prob = [prob[i].argsort()[-1] for i in range(len(prob))]
    print("lstm: ", get_ctc_decoder(index_prob, chn_tab))
    # act_label, index_prob, prob = pickle.load(lstm_info)
    # print(prob[0])
    # actual = CTCTokenPassing_AG(WordsDict, prob, chn_tab, LMMat, True)
    # CTCViterbiPassing(WordsDict, prob, chn_tab, LMMat, top=5)
    # print(actual)
    # print(prob[1][0])

    # ans_vec, ans_score = pretreatmentNetMat(prob, WordsDict, chn_tab)
    ans = CTCPasssingSimpleVersion(prob, WordsDict, chn_tab, LMMat)
    print("add lm:", ans)
    fw.write(ans + '\n')
    fw.close()


    # print(ans_vec)
    # print(ans_score)

if __name__ == '__main__':
    testTokenPassing()

# print(ans_tok)








# ============================================================