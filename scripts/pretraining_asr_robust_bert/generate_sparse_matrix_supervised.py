import numpy as np


def trunc(values, decs=0):
    return np.trunc(values * 10 ** decs) / (10 ** decs)


def main():
    # init
    vocab_size = 0
    top_k_value = 0
    
    ret_indices = np.zeros(shape=(vocab_size, top_k_value))
    ret_values = np.zeros(shape=(vocab_size, top_k_value))

    # compute ...

    # save to disk
    ret_indices.astype(int)
    ret_values = trunc(ret_values, decs=5)
    np.save('ret_indices_supervised.npy', ret_indices)
    np.save('ret_values_supervised.npy', ret_values)


if __name__ == '__main__':
    main()