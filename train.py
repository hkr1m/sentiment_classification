import torch
from math import copysign

def train_loop(dataloader, model, loss_fn, optimizer, scheduler):
    model.train()
    num_batches = len(dataloader)
    train_loss = 0.
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(model.config.device), y.to(model.config.device)
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        loss = loss.item()
        train_loss += loss
        if batch % 100 == 0:
            print(f"Loss: {loss:>7f}  [{(batch+1) * len(X):>5d}/{size:>5d}]")
    scheduler.step()
    return train_loss / num_batches

def _div(numerator, denominator):
    try:
        return numerator / denominator
    except ZeroDivisionError:
        return copysign(float("inf"), numerator)

def test_loop(dataloader, model, loss_fn):
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0.
    FP, TP, FN, TN, P, N = 0, 0, 0, 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(model.config.device), y.to(model.config.device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            tf = pred.argmax(1) == y
            for i in range(tf.size(0)):
                if y[i] == 0:
                    P = P + 1
                    if tf[i]: TP = TP + 1
                    else: FN = FN + 1
                else:
                    N = N + 1
                    if tf[i]: TN = TN + 1
                    else: FP = FP + 1
    test_loss /= num_batches
    precision = _div(TP, TP+FP)
    recall = _div(TP, P)
    accuracy = _div(TP+TN, P+N)
    F_measure = _div(2, _div(1, precision) + _div(1, recall))
    return test_loss, precision, recall, accuracy, F_measure