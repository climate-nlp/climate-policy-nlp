import unittest
from typing import Dict, List
import sklearn.metrics as sk_metrics
import src.evaluate_f1


def eval_f1(pred_jds: List[Dict], gold_jds: List[Dict], relieve_stance: bool = False) -> Dict:
    if relieve_stance:
        gold_jds = src.evaluate_f1.relieve_stance(jds=gold_jds)
        pred_jds = src.evaluate_f1.relieve_stance(jds=pred_jds)

    result = {
        'document': src.evaluate_f1.evaluate_document_f1(gold_jds=gold_jds, pred_jds=pred_jds),
        'overlap': src.evaluate_f1.evaluate_overlap_f1(gold_jds=gold_jds, pred_jds=pred_jds),
        'strict': src.evaluate_f1.evaluate_strict_f1(gold_jds=gold_jds, pred_jds=pred_jds),
    }
    return result


class TestEvaluateF1(unittest.TestCase):
    """
    Test class of src/evaluator_f1.py
    """

    def test_exact_match(self):
        system = [
            {
                'document_id': "001.pdf",
                'evidences': [
                    {'query': 'renewable_energy', 'stance': 'supporting', 'page_indices': [0]},
                    {'query': 'ghg_emission_regulation', 'stance': 'opposing', 'page_indices': [1]},
                ],
            },
            {
                'document_id': "002.pdf",
                'evidences': [
                    {'query': 'communication_of_climate_science', 'stance': 'no_position_or_mixed_position',
                     'page_indices': [0, 1]},
                ],
            }
        ]
        gold = system
        res = eval_f1(system, gold)

        y_true = ['[0]', '[1]', '[0, 1]']
        y_pred = ['[0]', '[1]', '[0, 1]']
        p = sk_metrics.precision_score(y_true=y_true, y_pred=y_pred, average='micro') * 100
        r = sk_metrics.recall_score(y_true=y_true, y_pred=y_pred, average='micro') * 100
        f = sk_metrics.f1_score(y_true=y_true, y_pred=y_pred, average='micro') * 100
        self.assertEqual(p, res['strict']['page']['p'])
        self.assertEqual(r, res['strict']['page']['r'])
        self.assertEqual(f, res['strict']['page']['f'])
        self.assertEqual(f, 100)

        y_true = ['renewable_energy,[0]', 'ghg_emission_regulation,[1]', 'communication_of_climate_science,[0, 1]']
        y_pred = ['renewable_energy,[0]', 'ghg_emission_regulation,[1]', 'communication_of_climate_science,[0, 1]']
        p = sk_metrics.precision_score(y_true=y_true, y_pred=y_pred, average='micro') * 100
        r = sk_metrics.recall_score(y_true=y_true, y_pred=y_pred, average='micro') * 100
        f = sk_metrics.f1_score(y_true=y_true, y_pred=y_pred, average='micro') * 100
        self.assertEqual(p, res['strict']['query']['p'])
        self.assertEqual(r, res['strict']['query']['r'])
        self.assertEqual(f, res['strict']['query']['f'])
        self.assertEqual(f, 100)

        y_true = ['supporting,[0]', 'opposing,[1]', 'no_position_or_mixed_position,[0, 1]']
        y_pred = ['supporting,[0]', 'opposing,[1]', 'no_position_or_mixed_position,[0, 1]']
        p = sk_metrics.precision_score(y_true=y_true, y_pred=y_pred, average='micro') * 100
        r = sk_metrics.recall_score(y_true=y_true, y_pred=y_pred, average='micro') * 100
        f = sk_metrics.f1_score(y_true=y_true, y_pred=y_pred, average='micro') * 100
        self.assertEqual(p, res['strict']['stance']['p'])
        self.assertEqual(r, res['strict']['stance']['r'])
        self.assertEqual(f, res['strict']['stance']['f'])
        self.assertEqual(f, 100)

        self.assertEqual(100, res['document']['page']['p'])
        self.assertEqual(100, res['document']['page']['r'])
        self.assertEqual(100, res['document']['page']['f'])

        self.assertEqual(100, res['document']['query']['p'])
        self.assertEqual(100, res['document']['query']['r'])
        self.assertEqual(100, res['document']['query']['f'])

        self.assertEqual(100, res['document']['stance']['p'])
        self.assertEqual(100, res['document']['stance']['r'])
        self.assertEqual(100, res['document']['stance']['f'])

        self.assertEqual(100, res['overlap']['page']['p'])
        self.assertEqual(100, res['overlap']['page']['r'])
        self.assertEqual(100, res['overlap']['page']['f'])

        self.assertEqual(100, res['overlap']['query']['p'])
        self.assertEqual(100, res['overlap']['query']['r'])
        self.assertEqual(100, res['overlap']['query']['f'])

        self.assertEqual(100, res['overlap']['stance']['p'])
        self.assertEqual(100, res['overlap']['stance']['r'])
        self.assertEqual(100, res['overlap']['stance']['f'])

    def test_partial_match_1(self):
        system = [
            {
                'document_id': "001.pdf",
                'evidences': [
                    {'query': 'renewable_energy', 'stance': 'supporting', 'page_indices': [0]},
                    {'query': 'ghg_emission_regulation', 'stance': 'opposing', 'page_indices': [1]},
                ],
            },
        ]
        gold = [
            {
                'document_id': "001.pdf",
                'evidences': [
                    {'query': 'renewable_energy', 'stance': 'supporting', 'page_indices': [0]},
                    {'query': 'ghg_emission_regulation', 'stance': 'opposing', 'page_indices': [1]},
                ],
            },
            {
                'document_id': "002.pdf",
                'evidences': [
                    {'query': 'communication_of_climate_science', 'stance': 'no_position_or_mixed_position',
                     'page_indices': [0, 1]},
                ],
            }
        ]
        res = eval_f1(system, gold)

        y_true = ['[0]', '[1]', '[0, 1]']
        y_pred = ['[0]', '[1]', '-']
        labels = sorted(list(set(y_true + y_pred) - {'-'}))
        p = sk_metrics.precision_score(y_true=y_true, y_pred=y_pred, labels=labels, average='micro') * 100
        r = sk_metrics.recall_score(y_true=y_true, y_pred=y_pred, labels=labels, average='micro') * 100
        f = sk_metrics.f1_score(y_true=y_true, y_pred=y_pred, labels=labels, average='micro') * 100
        self.assertAlmostEqual(p, res['strict']['page']['p'], delta=0.1)
        self.assertAlmostEqual(r, res['strict']['page']['r'], delta=0.1)
        self.assertAlmostEqual(f, res['strict']['page']['f'], delta=0.1)

        y_true = ['renewable_energy,[0]', 'ghg_emission_regulation,[1]', 'communication_of_climate_science,[0, 1]']
        y_pred = ['renewable_energy,[0]', 'ghg_emission_regulation,[1]', '-']
        labels = sorted(list(set(y_true + y_pred) - {'-'}))
        p = sk_metrics.precision_score(y_true=y_true, y_pred=y_pred, labels=labels, average='micro') * 100
        r = sk_metrics.recall_score(y_true=y_true, y_pred=y_pred, labels=labels, average='micro') * 100
        f = sk_metrics.f1_score(y_true=y_true, y_pred=y_pred, labels=labels, average='micro') * 100
        self.assertAlmostEqual(p, res['strict']['query']['p'], delta=0.1)
        self.assertAlmostEqual(r, res['strict']['query']['r'], delta=0.1)
        self.assertAlmostEqual(f, res['strict']['query']['f'], delta=0.1)

        y_true = ['supporting,[0]', 'opposing,[1]', 'no_position_or_mixed_position,[0, 1]']
        y_pred = ['supporting,[0]', 'opposing,[1]', '-']
        labels = sorted(list(set(y_true + y_pred) - {'-'}))
        p = sk_metrics.precision_score(y_true=y_true, y_pred=y_pred, labels=labels, average='micro') * 100
        r = sk_metrics.recall_score(y_true=y_true, y_pred=y_pred, labels=labels, average='micro') * 100
        f = sk_metrics.f1_score(y_true=y_true, y_pred=y_pred, labels=labels, average='micro') * 100
        self.assertAlmostEqual(p, res['strict']['stance']['p'], delta=0.1)
        self.assertAlmostEqual(r, res['strict']['stance']['r'], delta=0.1)
        self.assertAlmostEqual(f, res['strict']['stance']['f'], delta=0.1)

        p = 100 * 2 / 2
        r = 100 * 2 / 4
        f = 2 * p * r / (p + r)
        self.assertAlmostEqual(p, res['document']['page']['p'], delta=0.1)
        self.assertAlmostEqual(r, res['document']['page']['r'], delta=0.1)
        self.assertAlmostEqual(f, res['document']['page']['f'], delta=0.1)

        p = 100 * 2 / 2
        r = 100 * 2 / 3
        f = 2 * p * r / (p + r)
        self.assertAlmostEqual(p, res['document']['query']['p'], delta=0.1)
        self.assertAlmostEqual(r, res['document']['query']['r'], delta=0.1)
        self.assertAlmostEqual(f, res['document']['query']['f'], delta=0.1)

        p = 100 * 2 / 2
        r = 100 * 2 / 3
        f = 2 * p * r / (p + r)
        self.assertAlmostEqual(p, res['document']['stance']['p'], delta=0.1)
        self.assertAlmostEqual(r, res['document']['stance']['r'], delta=0.1)
        self.assertAlmostEqual(f, res['document']['stance']['f'], delta=0.1)

        tp = 2
        tpw = 2
        fp = 0
        fn = 1
        p = 100 * tpw / (tp + fp)
        r = 100 * tpw / (tp + fn)
        f = 2 * p * r / (p + r)
        self.assertAlmostEqual(p, res['overlap']['page']['p'], delta=0.1)
        self.assertAlmostEqual(r, res['overlap']['page']['r'], delta=0.1)
        self.assertAlmostEqual(f, res['overlap']['page']['f'], delta=0.1)

        self.assertAlmostEqual(p, res['overlap']['query']['p'], delta=0.1)
        self.assertAlmostEqual(r, res['overlap']['query']['r'], delta=0.1)
        self.assertAlmostEqual(f, res['overlap']['query']['f'], delta=0.1)

        self.assertAlmostEqual(p, res['overlap']['stance']['p'], delta=0.1)
        self.assertAlmostEqual(r, res['overlap']['stance']['r'], delta=0.1)
        self.assertAlmostEqual(f, res['overlap']['stance']['f'], delta=0.1)

    def test_partial_match_2(self):
        system = [
            {
                'document_id': "001.pdf",
                'evidences': [
                    {'query': 'renewable_energy', 'stance': 'not_supporting', 'page_indices': [1]},
                    {'query': 'ghg_emission_regulation', 'stance': 'supporting', 'page_indices': [2]},
                ],
            },
        ]
        gold = [
            {
                'document_id': "001.pdf",
                'evidences': [
                    {'query': 'renewable_energy', 'stance': 'supporting', 'page_indices': [1]},
                    {'query': 'ghg_emission_regulation', 'stance': 'opposing', 'page_indices': [0, 1]},
                ],
            },
        ]
        res = eval_f1(system, gold)

        y_true = ['[1]', '[2]']
        y_pred = ['[1]', '[0,1]']
        labels = sorted(list(set(y_true + y_pred) - {'-'}))
        p = sk_metrics.precision_score(y_true=y_true, y_pred=y_pred, labels=labels, average='micro') * 100
        r = sk_metrics.recall_score(y_true=y_true, y_pred=y_pred, labels=labels, average='micro') * 100
        f = sk_metrics.f1_score(y_true=y_true, y_pred=y_pred, labels=labels, average='micro') * 100
        self.assertAlmostEqual(p, res['strict']['page']['p'], delta=0.1)
        self.assertAlmostEqual(r, res['strict']['page']['r'], delta=0.1)
        self.assertAlmostEqual(f, res['strict']['page']['f'], delta=0.1)

        y_true = ['renewable_energy,[1]', 'ghg_emission_regulation,[2]']
        y_pred = ['renewable_energy,[1]', 'ghg_emission_regulation,[0,1]']
        labels = sorted(list(set(y_true + y_pred) - {'-'}))
        p = sk_metrics.precision_score(y_true=y_true, y_pred=y_pred, labels=labels, average='micro') * 100
        r = sk_metrics.recall_score(y_true=y_true, y_pred=y_pred, labels=labels, average='micro') * 100
        f = sk_metrics.f1_score(y_true=y_true, y_pred=y_pred, labels=labels, average='micro') * 100
        self.assertAlmostEqual(p, res['strict']['query']['p'], delta=0.1)
        self.assertAlmostEqual(r, res['strict']['query']['r'], delta=0.1)
        self.assertAlmostEqual(f, res['strict']['query']['f'], delta=0.1)

        y_true = ['not_supporting,[1]', 'supporting,[2]']
        y_pred = ['supporting,[1]', 'opposing,[0,1]']
        labels = sorted(list(set(y_true + y_pred) - {'-'}))
        p = sk_metrics.precision_score(y_true=y_true, y_pred=y_pred, labels=labels, average='micro') * 100
        r = sk_metrics.recall_score(y_true=y_true, y_pred=y_pred, labels=labels, average='micro') * 100
        f = sk_metrics.f1_score(y_true=y_true, y_pred=y_pred, labels=labels, average='micro') * 100
        self.assertAlmostEqual(p, res['strict']['stance']['p'], delta=0.1)
        self.assertAlmostEqual(r, res['strict']['stance']['r'], delta=0.1)
        self.assertAlmostEqual(f, res['strict']['stance']['f'], delta=0.1)
        self.assertAlmostEqual(f, 0.)

        p = 100 * 1 / 2
        r = 100 * 1 / 2
        f = 2 * p * r / (p + r)
        self.assertAlmostEqual(p, res['document']['page']['p'], delta=0.1)
        self.assertAlmostEqual(r, res['document']['page']['r'], delta=0.1)
        self.assertAlmostEqual(f, res['document']['page']['f'], delta=0.1)

        p = 100 * 2 / 2
        r = 100 * 2 / 2
        f = 2 * p * r / (p + r)
        self.assertAlmostEqual(p, res['document']['query']['p'], delta=0.1)
        self.assertAlmostEqual(r, res['document']['query']['r'], delta=0.1)
        self.assertAlmostEqual(f, res['document']['query']['f'], delta=0.1)

        p = 100 * 1 / 2
        r = 100 * 1 / 2
        f = 2 * p * r / (p + r)
        self.assertAlmostEqual(p, res['document']['stance']['p'], delta=0.1)
        self.assertAlmostEqual(r, res['document']['stance']['r'], delta=0.1)
        self.assertAlmostEqual(f, res['document']['stance']['f'], delta=0.1)

        tp_prec = 1
        tpw_prec = 1
        fp_prec = 1
        tp_rec = 2
        tpw_rec = 1 + 1 / 2
        fn_rec = 0
        p = 100 * tpw_prec / (tp_prec + fp_prec)
        r = 100 * tpw_rec / (tp_rec + fn_rec)
        f = 2 * p * r / (p + r)
        self.assertAlmostEqual(p, res['overlap']['page']['p'], delta=0.1)
        self.assertAlmostEqual(r, res['overlap']['page']['r'], delta=0.1)
        self.assertAlmostEqual(f, res['overlap']['page']['f'], delta=0.1)

        tp_prec = 1
        tpw_prec = 1
        fp_prec = 1
        tp_rec = 1
        tpw_rec = 1
        fn_rec = 1
        p = 100 * tpw_prec / (tp_prec + fp_prec)
        r = 100 * tpw_rec / (tp_rec + fn_rec)
        f = 2 * p * r / (p + r)
        self.assertAlmostEqual(p, res['overlap']['query']['p'], delta=0.1)
        self.assertAlmostEqual(r, res['overlap']['query']['r'], delta=0.1)
        self.assertAlmostEqual(f, res['overlap']['query']['f'], delta=0.1)

        tp_prec = 0
        tpw_prec = 0
        fp_prec = 2
        tp_rec = 0
        tpw_rec = 0
        fn_rec = 2
        p = 100 * tpw_prec / (tp_prec + fp_prec)
        r = 100 * tpw_rec / (tp_rec + fn_rec)
        f = 0.
        self.assertAlmostEqual(p, res['overlap']['stance']['p'], delta=0.1)
        self.assertAlmostEqual(r, res['overlap']['stance']['r'], delta=0.1)
        self.assertAlmostEqual(f, res['overlap']['stance']['f'], delta=0.1)

    def test_partial_match_3(self):
        system = [
            {
                'document_id': "001.pdf",
                'evidences': [
                    # Duplication
                    {'query': 'renewable_energy', 'stance': 'not_supporting', 'page_indices': [1]},
                    {'query': 'renewable_energy', 'stance': 'not_supporting', 'page_indices': [1]},
                    {'query': 'renewable_energy', 'stance': 'not_supporting', 'page_indices': [1]},
                    {'query': 'renewable_energy', 'stance': 'not_supporting', 'page_indices': [1]},
                    {'query': 'ghg_emission_regulation', 'stance': 'supporting', 'page_indices': [2]},
                    {'query': 'ghg_emission_regulation', 'stance': 'supporting', 'page_indices': [2]},
                    {'query': 'ghg_emission_regulation', 'stance': 'supporting', 'page_indices': [2]},
                    {'query': 'ghg_emission_regulation', 'stance': 'supporting', 'page_indices': [2]},
                ],
            },
        ]
        gold = [
            {
                'document_id': "001.pdf",
                'evidences': [
                    {'query': 'renewable_energy', 'stance': 'supporting', 'page_indices': [1]},
                    {'query': 'ghg_emission_regulation', 'stance': 'opposing', 'page_indices': [0, 1]},
                ],
            },
        ]
        res = eval_f1(system, gold)

        y_true = ['[1]', '[2]']
        y_pred = ['[1]', '[0,1]']
        labels = sorted(list(set(y_true + y_pred) - {'-'}))
        p = sk_metrics.precision_score(y_true=y_true, y_pred=y_pred, labels=labels, average='micro') * 100
        r = sk_metrics.recall_score(y_true=y_true, y_pred=y_pred, labels=labels, average='micro') * 100
        f = sk_metrics.f1_score(y_true=y_true, y_pred=y_pred, labels=labels, average='micro') * 100
        self.assertAlmostEqual(p, res['strict']['page']['p'], delta=0.1)
        self.assertAlmostEqual(r, res['strict']['page']['r'], delta=0.1)
        self.assertAlmostEqual(f, res['strict']['page']['f'], delta=0.1)

        y_true = ['renewable_energy,[1]', 'ghg_emission_regulation,[2]']
        y_pred = ['renewable_energy,[1]', 'ghg_emission_regulation,[0,1]']
        labels = sorted(list(set(y_true + y_pred) - {'-'}))
        p = sk_metrics.precision_score(y_true=y_true, y_pred=y_pred, labels=labels, average='micro') * 100
        r = sk_metrics.recall_score(y_true=y_true, y_pred=y_pred, labels=labels, average='micro') * 100
        f = sk_metrics.f1_score(y_true=y_true, y_pred=y_pred, labels=labels, average='micro') * 100
        self.assertAlmostEqual(p, res['strict']['query']['p'], delta=0.1)
        self.assertAlmostEqual(r, res['strict']['query']['r'], delta=0.1)
        self.assertAlmostEqual(f, res['strict']['query']['f'], delta=0.1)

        y_true = ['not_supporting,[1]', 'supporting,[2]']
        y_pred = ['supporting,[1]', 'opposing,[0,1]']
        labels = sorted(list(set(y_true + y_pred) - {'-'}))
        p = sk_metrics.precision_score(y_true=y_true, y_pred=y_pred, labels=labels, average='micro') * 100
        r = sk_metrics.recall_score(y_true=y_true, y_pred=y_pred, labels=labels, average='micro') * 100
        f = sk_metrics.f1_score(y_true=y_true, y_pred=y_pred, labels=labels, average='micro') * 100
        self.assertAlmostEqual(p, res['strict']['stance']['p'], delta=0.1)
        self.assertAlmostEqual(r, res['strict']['stance']['r'], delta=0.1)
        self.assertAlmostEqual(f, res['strict']['stance']['f'], delta=0.1)
        self.assertAlmostEqual(f, 0.)

        p = 100 * 1 / 2
        r = 100 * 1 / 2
        f = 2 * p * r / (p + r)
        self.assertAlmostEqual(p, res['document']['page']['p'], delta=0.1)
        self.assertAlmostEqual(r, res['document']['page']['r'], delta=0.1)
        self.assertAlmostEqual(f, res['document']['page']['f'], delta=0.1)

        p = 100 * 2 / 2
        r = 100 * 2 / 2
        f = 2 * p * r / (p + r)
        self.assertAlmostEqual(p, res['document']['query']['p'], delta=0.1)
        self.assertAlmostEqual(r, res['document']['query']['r'], delta=0.1)
        self.assertAlmostEqual(f, res['document']['query']['f'], delta=0.1)

        p = 100 * 1 / 2
        r = 100 * 1 / 2
        f = 2 * p * r / (p + r)
        self.assertAlmostEqual(p, res['document']['stance']['p'], delta=0.1)
        self.assertAlmostEqual(r, res['document']['stance']['r'], delta=0.1)
        self.assertAlmostEqual(f, res['document']['stance']['f'], delta=0.1)

        tp_prec = 1
        tpw_prec = 1
        fp_prec = 1
        tp_rec = 2
        tpw_rec = 1 + 1 / 2
        fn_rec = 0
        p = 100 * tpw_prec / (tp_prec + fp_prec)
        r = 100 * tpw_rec / (tp_rec + fn_rec)
        f = 2 * p * r / (p + r)
        self.assertAlmostEqual(p, res['overlap']['page']['p'], delta=0.1)
        self.assertAlmostEqual(r, res['overlap']['page']['r'], delta=0.1)
        self.assertAlmostEqual(f, res['overlap']['page']['f'], delta=0.1)

        tp_prec = 1
        tpw_prec = 1
        fp_prec = 1
        tp_rec = 1
        tpw_rec = 1
        fn_rec = 1
        p = 100 * tpw_prec / (tp_prec + fp_prec)
        r = 100 * tpw_rec / (tp_rec + fn_rec)
        f = 2 * p * r / (p + r)
        self.assertAlmostEqual(p, res['overlap']['query']['p'], delta=0.1)
        self.assertAlmostEqual(r, res['overlap']['query']['r'], delta=0.1)
        self.assertAlmostEqual(f, res['overlap']['query']['f'], delta=0.1)

        tp_prec = 0
        tpw_prec = 0
        fp_prec = 2
        tp_rec = 0
        tpw_rec = 0
        fn_rec = 2
        p = 100 * tpw_prec / (tp_prec + fp_prec)
        r = 100 * tpw_rec / (tp_rec + fn_rec)
        f = 0.
        self.assertAlmostEqual(p, res['overlap']['stance']['p'], delta=0.1)
        self.assertAlmostEqual(r, res['overlap']['stance']['r'], delta=0.1)
        self.assertAlmostEqual(f, res['overlap']['stance']['f'], delta=0.1)

    def test_partial_match_4(self):
        system = [
            {
                'document_id': "001.pdf",
                'evidences': [
                    {'query': 'renewable_energy', 'stance': 'opposing', 'page_indices': [0, 1]},
                ],
            },
        ]
        gold = [
            {
                'document_id': "001.pdf",
                'evidences': [
                    {'query': 'renewable_energy', 'stance': 'not_supporting', 'page_indices': [1, 2]},
                ],
            },
        ]
        res = eval_f1(system, gold)

        y_true = ['[0,1]']
        y_pred = ['[1,2]']
        labels = sorted(list(set(y_true + y_pred) - {'-'}))
        p = sk_metrics.precision_score(y_true=y_true, y_pred=y_pred, labels=labels, average='micro') * 100
        r = sk_metrics.recall_score(y_true=y_true, y_pred=y_pred, labels=labels, average='micro') * 100
        f = sk_metrics.f1_score(y_true=y_true, y_pred=y_pred, labels=labels, average='micro') * 100
        self.assertAlmostEqual(p, res['strict']['page']['p'], delta=0.1)
        self.assertAlmostEqual(r, res['strict']['page']['r'], delta=0.1)
        self.assertAlmostEqual(f, res['strict']['page']['f'], delta=0.1)

        y_true = ['renewable_energy,[0, 1]']
        y_pred = ['renewable_energy,[1, 2]']
        labels = sorted(list(set(y_true + y_pred) - {'-'}))
        p = sk_metrics.precision_score(y_true=y_true, y_pred=y_pred, labels=labels, average='micro') * 100
        r = sk_metrics.recall_score(y_true=y_true, y_pred=y_pred, labels=labels, average='micro') * 100
        f = sk_metrics.f1_score(y_true=y_true, y_pred=y_pred, labels=labels, average='micro') * 100
        self.assertAlmostEqual(p, res['strict']['query']['p'], delta=0.1)
        self.assertAlmostEqual(r, res['strict']['query']['r'], delta=0.1)
        self.assertAlmostEqual(f, res['strict']['query']['f'], delta=0.1)

        y_true = ['opposing,[0, 1]']
        y_pred = ['not_supporting,[1,2]']
        labels = sorted(list(set(y_true + y_pred) - {'-'}))
        p = sk_metrics.precision_score(y_true=y_true, y_pred=y_pred, labels=labels, average='micro') * 100
        r = sk_metrics.recall_score(y_true=y_true, y_pred=y_pred, labels=labels, average='micro') * 100
        f = sk_metrics.f1_score(y_true=y_true, y_pred=y_pred, labels=labels, average='micro') * 100
        self.assertAlmostEqual(p, res['strict']['stance']['p'], delta=0.1)
        self.assertAlmostEqual(r, res['strict']['stance']['r'], delta=0.1)
        self.assertAlmostEqual(f, res['strict']['stance']['f'], delta=0.1)

        p = 100 * 1 / 2
        r = 100 * 1 / 2
        f = 2 * p * r / (p + r)
        self.assertAlmostEqual(p, res['document']['page']['p'], delta=0.1)
        self.assertAlmostEqual(r, res['document']['page']['r'], delta=0.1)
        self.assertAlmostEqual(f, res['document']['page']['f'], delta=0.1)

        p = 100 * 1 / 1
        r = 100 * 1 / 1
        f = 2 * p * r / (p + r)
        self.assertAlmostEqual(p, res['document']['query']['p'], delta=0.1)
        self.assertAlmostEqual(r, res['document']['query']['r'], delta=0.1)
        self.assertAlmostEqual(f, res['document']['query']['f'], delta=0.1)

        p = 0
        r = 0
        f = 0
        self.assertAlmostEqual(p, res['document']['stance']['p'], delta=0.1)
        self.assertAlmostEqual(r, res['document']['stance']['r'], delta=0.1)
        self.assertAlmostEqual(f, res['document']['stance']['f'], delta=0.1)

        tp = 1
        tpw_prec = 1 / 2
        tpw_rec = 1 / 2
        fp = 0
        fn = 0
        p = 100 * tpw_prec / (tp + fp)
        r = 100 * tpw_rec / (tp + fn)
        f = 2 * p * r / (p + r)
        self.assertAlmostEqual(p, res['overlap']['page']['p'], delta=0.1)
        self.assertAlmostEqual(r, res['overlap']['page']['r'], delta=0.1)
        self.assertAlmostEqual(f, res['overlap']['page']['f'], delta=0.1)

        tp = 1
        tpw_prec = 1 / 2
        tpw_rec = 1 / 2
        fp = 0
        fn = 0
        p = 100 * tpw_prec / (tp + fp)
        r = 100 * tpw_rec / (tp + fn)
        f = 2 * p * r / (p + r)
        self.assertAlmostEqual(p, res['overlap']['query']['p'], delta=0.1)
        self.assertAlmostEqual(r, res['overlap']['query']['r'], delta=0.1)
        self.assertAlmostEqual(f, res['overlap']['query']['f'], delta=0.1)

        p = 0
        r = 0
        f = 0
        self.assertAlmostEqual(p, res['overlap']['stance']['p'], delta=0.1)
        self.assertAlmostEqual(r, res['overlap']['stance']['r'], delta=0.1)
        self.assertAlmostEqual(f, res['overlap']['stance']['f'], delta=0.1)

    def test_zero_1(self):
        system = [
            {
                'document_id': "001.pdf",
                'evidences': [
                ],
            },
        ]
        gold = [
            {
                'document_id': "001.pdf",
                'evidences': [
                    {'query': 'renewable_energy', 'stance': 'supporting', 'page_indices': [1]},
                    {'query': 'ghg_emission_regulation', 'stance': 'opposing', 'page_indices': [0, 1]},
                ],
            },
        ]
        res = eval_f1(system, gold)

        self.assertEqual(0, res['strict']['page']['p'])
        self.assertEqual(0, res['strict']['page']['r'])
        self.assertEqual(0, res['strict']['page']['f'])

        self.assertEqual(0, res['strict']['query']['p'])
        self.assertEqual(0, res['strict']['query']['r'])
        self.assertEqual(0, res['strict']['query']['f'])

        self.assertEqual(0, res['strict']['stance']['p'])
        self.assertEqual(0, res['strict']['stance']['r'])
        self.assertEqual(0, res['strict']['stance']['f'])

        self.assertEqual(0, res['document']['page']['p'])
        self.assertEqual(0, res['document']['page']['r'])
        self.assertEqual(0, res['document']['page']['f'])

        self.assertEqual(0, res['document']['query']['p'])
        self.assertEqual(0, res['document']['query']['r'])
        self.assertEqual(0, res['document']['query']['f'])

        self.assertEqual(0, res['document']['stance']['p'])
        self.assertEqual(0, res['document']['stance']['r'])
        self.assertEqual(0, res['document']['stance']['f'])

        self.assertEqual(0, res['overlap']['page']['p'])
        self.assertEqual(0, res['overlap']['page']['r'])
        self.assertEqual(0, res['overlap']['page']['f'])

        self.assertEqual(0, res['overlap']['query']['p'])
        self.assertEqual(0, res['overlap']['query']['r'])
        self.assertEqual(0, res['overlap']['query']['f'])

        self.assertEqual(0, res['overlap']['stance']['p'])
        self.assertEqual(0, res['overlap']['stance']['r'])
        self.assertEqual(0, res['overlap']['stance']['f'])

    def test_zero_2(self):
        system = [
            {
                'document_id': "001.pdf",
                'evidences': [
                    {'query': 'renewable_energy', 'stance': 'not_supporting', 'page_indices': [1]},
                    {'query': 'ghg_emission_regulation', 'stance': 'supporting', 'page_indices': [2]},
                ],
            },
        ]
        gold = [
            {
                'document_id': "001.pdf",
                'evidences': [
                ],
            },
        ]
        res = eval_f1(system, gold)

        self.assertEqual(0, res['strict']['page']['p'])
        self.assertEqual(0, res['strict']['page']['r'])
        self.assertEqual(0, res['strict']['page']['f'])

        self.assertEqual(0, res['strict']['query']['p'])
        self.assertEqual(0, res['strict']['query']['r'])
        self.assertEqual(0, res['strict']['query']['f'])

        self.assertEqual(0, res['strict']['stance']['p'])
        self.assertEqual(0, res['strict']['stance']['r'])
        self.assertEqual(0, res['strict']['stance']['f'])

        self.assertEqual(0, res['document']['page']['p'])
        self.assertEqual(0, res['document']['page']['r'])
        self.assertEqual(0, res['document']['page']['f'])

        self.assertEqual(0, res['document']['query']['p'])
        self.assertEqual(0, res['document']['query']['r'])
        self.assertEqual(0, res['document']['query']['f'])

        self.assertEqual(0, res['document']['stance']['p'])
        self.assertEqual(0, res['document']['stance']['r'])
        self.assertEqual(0, res['document']['stance']['f'])

        self.assertEqual(0, res['overlap']['page']['p'])
        self.assertEqual(0, res['overlap']['page']['r'])
        self.assertEqual(0, res['overlap']['page']['f'])

        self.assertEqual(0, res['overlap']['query']['p'])
        self.assertEqual(0, res['overlap']['query']['r'])
        self.assertEqual(0, res['overlap']['query']['f'])

        self.assertEqual(0, res['overlap']['stance']['p'])
        self.assertEqual(0, res['overlap']['stance']['r'])
        self.assertEqual(0, res['overlap']['stance']['f'])
