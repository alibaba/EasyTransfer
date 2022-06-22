import numpy as np


def sound_code_similarity(char1, char2, alpha, sound_code_dict):
    if char1 == char2:
        return 1
    if char1 not in sound_code_dict:
        return 0
    if char2 not in sound_code_dict:
        return 0
    sc1 = sound_code_dict[char1]
    sc2 = sound_code_dict[char2]
    wights = [0.4, 0.4, 0.1, 0.1]
    multiplier = []
    for i in range(4):
        if sc1[i] == sc2[i]:
            multiplier.append(1)
        else:
            multiplier.append(0)
    sc_sim = 0
    for i in range(4):
        sc_sim += wights[i] * multiplier[i]
    return sc_sim * alpha


def load_sound_code_dict(path):
    outcome = dict()
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            items = line.split("\t")
            char = items[0]
            code = items[1]
            outcome[char] = code
    return outcome


def trunc(values, decs=0):
    return np.trunc(values * 10 ** decs) / (10 ** decs)


def main():
    # http://atp-modelzoo.oss-cn-hangzhou.aliyuncs.com/asr_robust_bert/zh_sound_code.txt
    
    # consider top-k similar words
    top_k_value = 20

    # weight for similar word
    alpha = 0.2

    sound_code_dict = load_sound_code_dict("zh_sound_code.txt")
    vocab_list = list()
    with open("vocab.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            vocab_list.append(line.strip())

    sim_list = list()
    for i, char1 in enumerate(vocab_list):
        temp_word_sim_list = list()
        for j, char2 in enumerate(vocab_list):
            sim = sound_code_similarity(char1, char2, alpha, sound_code_dict)
            temp_word_sim_list.append(sim)
        word_sim_list = list()
        for k, ele in enumerate(temp_word_sim_list):
            word_sim_list.append(ele)
        sim_list.append(word_sim_list)

    sim_array = np.array(sim_list)
    ret_indices = np.zeros(shape=(len(sim_array), top_k_value))
    ret_values = np.zeros(shape=(len(sim_array), top_k_value))
    for idx in range(len(sim_array)):
        sim_vec = sim_array[idx]
        xx = np.nonzero(sim_vec > 0.0001)[0]
        yy = sim_vec[xx]
        zz = zip(xx, yy)
        sorted_zz = sorted(zz, key=lambda x: x[1], reverse=True)
        indices = [0] * top_k_value
        values = [0.0] * top_k_value
        for i in range(len(sorted_zz)):
            if i == top_k_value:
                break
            indices[i] = sorted_zz[i][0]
            values[i] = sorted_zz[i][1]
        values = values / np.sum(values)
        ret_indices[idx] = indices
        ret_values[idx] = values

    ret_indices.astype(int)
    ret_values = trunc(ret_values, decs=5)
    np.save('ret_indices.npy', ret_indices)
    np.save('ret_values.npy', ret_values)


if __name__ == '__main__':
    main()