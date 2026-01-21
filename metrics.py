def accuracy(preds, labels):
    return sum([pred==l for pred, l in zip(preds, labels)])
def range(preds, labels):
    return sum([m[0]<=pred<=m[1] for pred, m in zip(preds, labels)])