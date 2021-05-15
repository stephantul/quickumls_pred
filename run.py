from collections import Counter
from itertools import chain

from umlspred.data import read_pubtator
from umlspred.quick import QuickUMLSClassifier
from umlspred.evaluation import evaluate_all


if __name__ == "__main__":

    test_ids = {x.strip() for x in open("data/corpus_pubtator_pmids_test.txt").readlines()}
    data = read_pubtator("data/corpus_pubtator.txt", mode="semtype")

    test, train = [], []
    for x in data:
        if x[0] in test_ids:
            test.append(x)
        else:
            train.append(x)

    smoothing = 0
    _, classes = zip(*list(chain(*[x[2] for x in train])))
    classes = Counter(classes)
    total = sum(classes.values()) + (len(classes) * smoothing)
    priors = {k: (v + smoothing) / total for k, v in classes.items()}

    path = None

    if path is None:
        raise ValueError("Please specify the path to the quickumls installation.")

    q = QuickUMLSClassifier.load(
        path,
        threshold=0.99,
        similarity_name="jaccard",
        priors=priors,
        pooling="max",
        spacy_string="en_core_sci_sm",
        n_workers=20,
    )
    print("Loaded QUMLS")
    pred = q.predict([x[1] for x in test])
    res = evaluate_all(test, pred)
