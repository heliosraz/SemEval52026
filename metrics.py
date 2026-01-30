def accuracy(preds, labels):
    return sum([pred == l for pred, l in zip(preds, labels)])


def range(preds, labels):
    correct = 0
    for bottom, top, pred in zip(labels[0], labels[1], preds):
        correct += bottom.item() <= pred <= top.item()
    return correct
