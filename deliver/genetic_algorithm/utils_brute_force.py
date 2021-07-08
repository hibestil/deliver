from itertools import permutations, product


def permutate(array):
    return list(permutations(array, len(array)))


def product_perms(data):
    perms = [list(map(list, permutations(subl))) for subl in data]
    result = []
    for data in product(*perms):
        result.append(list(data))
    return result


def mapdistr(K, N):
    result = []
    for x in range(N ** len(K)):
        t = x
        l = [[] for _ in range(N)]
        for i in K:
            id = t % N
            t = t // N  # integer division
            l[id].append(i)

        # result.append(l)
        # if l:
        for p in product_perms(l):
            # print("l:",p)
            result.append(p)
    return result
