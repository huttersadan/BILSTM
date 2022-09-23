#rand(7) -> rand(10)

import random

dict_ = {11:1,12:2,13:3,14:4,15:5,16:6,17:7,21:8,22:9,23:10}
def get_rand10():
    a = random.randint(1,7)
    b = random.randint(1,7)
    flag = 1
    result = 0
    while(flag):
        if a >= 3:
            a = random.randint(1, 7)
            b = random.randint(1, 7)
            continue
        else:
            if (a == 2 and b >= 4):
                a = random.randint(1, 7)
                b = random.randint(1, 7)
                continue
            else:
                #只剩下11 ,12,13,14,15,16,17, 21,22,23 十个数字了，这十个数字可以出现的概率是一样的，
                num = a*10+b
                result = dict_[num]
                break
    return result


for i in range(100):
    print(get_rand10(8))