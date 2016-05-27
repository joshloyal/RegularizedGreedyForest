from rgforest import rgf

def test_train():
    est = rgf.RegularizedGreedyForest()
    est.train()
