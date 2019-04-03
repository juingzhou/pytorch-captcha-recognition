import numpy as np
import captcha_generate

def encode(text):
    vector = np.zeros((captcha_generate.MAX_CAPTCHA, captcha_generate.ALL_CHAR_SET_LEN), dtype=np.float)
    for i, ch in enumerate(text):
        # print(c)
        vector[i, :] = 0
        vector[i, captcha_generate.ALL_CHAR_SET.find(ch)] = 1
    return vector


def decode(vec):
    vec = np.argmax(np.array(vec), axis=1)
    return ''.join([captcha_generate.ALL_CHAR_SET[x] for x in vec])



if __name__ == '__main__':
    e = encode("BK7H")
    print(e)
    print(decode(e))
