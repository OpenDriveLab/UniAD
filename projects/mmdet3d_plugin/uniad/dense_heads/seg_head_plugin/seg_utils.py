

def IOU(intputs, targets):
    numerator = (intputs * targets).sum(dim=1)
    denominator = intputs.sum(dim=1) + targets.sum(dim=1) - numerator
    loss = numerator / (denominator + 0.0000000000001)
    return loss.cpu(), numerator.cpu(), denominator.cpu()
