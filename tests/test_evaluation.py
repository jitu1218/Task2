import json
import unittest
import numpy as np
from sklearn.datasets import load_breast_cancer
from train import metrics, split_data, ROOT


class EvaluationTests(unittest.TestCase):
    def test_real_class_counts(self):
        d = load_breast_cancer()
        self.assertEqual(d.data.shape, (569, 30))
        self.assertEqual(int((d.target == 0).sum()), 212)

    def test_disjoint_splits(self):
        y = (load_breast_cancer().target == 0).astype(int)
        split = split_data(y)
        merged = np.concatenate(list(split.values()))
        self.assertEqual(len(set(merged)), 569)
        self.assertEqual([len(v) for v in split.values()], [341, 114, 114])

    def test_metrics_known_counts(self):
        r = metrics([0, 0, 1, 1], [.1, .8, .2, .9], .5)
        self.assertEqual([r[k] for k in ['tn','fp','fn','tp']], [1,1,1,1])
        self.assertEqual(r['precision'], .5)
        self.assertEqual(r['cost'], 6)

    def test_no_positive_predictions(self):
        r = metrics([0, 1], [.2, .8], 1)
        self.assertEqual(r['precision'], 0)
        self.assertEqual(r['recall'], 0)

    def test_artifact_metrics(self):
        d = json.loads((ROOT / 'results/evaluation.json').read_text())
        for m in d['models']:
            for s in ['validation', 'test']:
                computed = metrics(d['splits'][s]['y'], m['results'][s]['p'], m['threshold'])
                for key, value in computed.items():
                    self.assertAlmostEqual(value, m['results'][s]['metrics'][key])

    def test_validation_selection(self):
        d = json.loads((ROOT / 'results/evaluation.json').read_text())
        best = min(d['models'], key=lambda m: (m['results']['validation']['metrics']['cost'],
                                                -m['results']['validation']['metrics']['f1']))
        self.assertEqual(d['selected'], best['id'])
        for m in d['models']:
            y, p = d['splits']['validation']['y'], m['results']['validation']['p']
            choices = [(float(t), metrics(y, p, t)) for t in np.round(np.arange(.05,.951,.01),2)]
            threshold, _ = min(choices,key=lambda v:(v[1]['cost'],-v[1]['f1'],v[0]))
            self.assertEqual(m['threshold'], threshold)


if __name__ == '__main__':
    unittest.main()
