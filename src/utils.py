def cud(cuda, *a):
    if cuda:
        res = [i.cuda() for i in a]
    else:
        res = a
    if len(res) == 1:
        return res[0]
    return res
