"""
This code is obtained and modified from:
https://github.com/jerbarnes/semeval22_structured_sentiment/
"""

from distutils.util import strtobool
import copy
import json
import math
import argparse
from typing import Dict, List, Tuple, Set


def process_score(n: float):
    assert 0 <= n <= 1
    decimal_places = 1
    score = math.floor(n * 10 ** (2 + decimal_places)) / (10 ** decimal_places)
    assert 0 <= score <= 100
    return score


def convert_evidence_to_tuple(instance: Dict, keep_query: bool = True, keep_stance: bool = True):
    evidences = instance["evidences"]

    stance_tuples = []
    for evidence in evidences:
        query = evidence["query"]
        if not keep_query:
            query = '_'

        stance = evidence["stance"]
        if not keep_stance:
            stance = '_'

        evidence_idxs = frozenset(evidence["page_indices"])
        assert len(evidence_idxs) > 0, f'At least 1 evidence idx required, where output id={instance["document_id"]}'
        if (query, stance, evidence_idxs) in stance_tuples:
            continue
        stance_tuples.append((query, stance, evidence_idxs))

    return stance_tuples


def sent_tuples_in_list(
        sent_tuple1: Tuple,
        list_of_sent_tuples: List[Tuple],
        keep_query: bool = True,
        keep_stance: bool = True
):
    query1, stance1, evidence1 = sent_tuple1

    for query2, stance2, evidence2 in list_of_sent_tuples:
        if evidence1 & evidence2:
            if keep_query and (query1 != query2):
                continue
            if keep_stance and (stance1 != stance2):
                continue
            return True

    return False


def weighted_score(sent_tuple1, list_of_sent_tuples):
    best_overlap = 0
    query1, stance1, evidence1 = sent_tuple1

    for query2, stance2, evidence2 in list_of_sent_tuples:
        if evidence1 & evidence2:
            overlap = len(evidence1 & evidence2) / len(evidence1)
            if overlap > best_overlap:
                best_overlap = overlap

    return best_overlap


def tuple_precision(
    gold: Dict[str, List[Tuple]],
    pred: Dict[str, List[Tuple]],
    keep_query: bool = True,
    keep_stance: bool = True,
    weighted: bool = True
):
    """
    Weighted true positives / (true positives + false positives)
    """
    weighted_tp = []
    tp = []
    fp = []

    for pred_id, ptuples in pred.items():
        gtuples = gold[pred_id] if pred_id in gold else []

        for stuple in ptuples:
            if sent_tuples_in_list(
                    sent_tuple1=stuple,
                    list_of_sent_tuples=gtuples,
                    keep_query=keep_query,
                    keep_stance=keep_stance
            ):
                if weighted:
                    weighted_tp.append(weighted_score(stuple, gtuples))
                    tp.append(1)
                else:
                    weighted_tp.append(1)
                    tp.append(1)
            else:
                fp.append(1)
    #print("weighted tp: {}".format(sum(weighted_tp)))
    #print("tp: {}".format(sum(tp)))
    #print("fp: {}".format(sum(fp)))
    return sum(weighted_tp) / (sum(tp) + sum(fp) + 0.0000000000000001)


def tuple_recall(
    gold: Dict[str, List[Tuple]],
    pred: Dict[str, List[Tuple]],
    keep_query: bool = True,
    keep_stance: bool = True,
    weighted: bool = True
):
    """
    Weighted true positives / (true positives + false negatives)
    """
    weighted_tp = []
    tp = []
    fn = []

    for gold_id, gtuples in gold.items():
        ptuples = pred[gold_id] if gold_id in pred else []

        for stuple in gtuples:
            if sent_tuples_in_list(
                    sent_tuple1=stuple,
                    list_of_sent_tuples=ptuples,
                    keep_query=keep_query,
                    keep_stance=keep_stance
            ):
                if weighted:
                    weighted_tp.append(weighted_score(stuple, ptuples))
                    tp.append(1)
                else:
                    weighted_tp.append(1)
                    tp.append(1)
            else:
                fn.append(1)
    return sum(weighted_tp) / (sum(tp) + sum(fn) + 0.0000000000000001)


def tuple_f1(
        gold_jds: List[Dict],
        pred_jds: List[Dict],
        keep_query: bool = True,
        keep_stance: bool = True,
        weighted: bool = True
):
    gold: Dict[str, List[Tuple]] = dict(
        [(s["document_id"], convert_evidence_to_tuple(s, keep_query=keep_query, keep_stance=keep_stance))
         for s in gold_jds]
    )
    pred: Dict[str, List[Tuple]] = dict(
        [(s["document_id"], convert_evidence_to_tuple(s, keep_query=keep_query, keep_stance=keep_stance))
         for s in pred_jds]
    )

    prec = tuple_precision(gold=gold, pred=pred, keep_query=keep_query, keep_stance=keep_stance, weighted=weighted)
    rec = tuple_recall(gold=gold, pred=pred, keep_query=keep_query, keep_stance=keep_stance, weighted=weighted)
    f1 = (2 * (prec * rec) / (prec + rec)) if prec + rec > 0 else 0.

    return {
        'p': process_score(prec),
        'r': process_score(rec),
        'f': process_score(f1)
    }


def evaluate_overlap_f1(gold_jds: List[Dict], pred_jds: List[Dict]) -> Dict:
    result = {
        'page': tuple_f1(gold_jds=gold_jds, pred_jds=pred_jds, keep_query=False, keep_stance=False),
        'query': tuple_f1(gold_jds=gold_jds, pred_jds=pred_jds, keep_query=True, keep_stance=False),
        'stance': tuple_f1(gold_jds=gold_jds, pred_jds=pred_jds, keep_query=False, keep_stance=True),
    }
    return result


def document_level_f1(
        gold_jds: List[Dict],
        pred_jds: List[Dict],
        keep_page: bool = True,
        keep_query: bool = True,
        keep_stance: bool = True
) -> Dict:

    def _get_tuples(_jds: List[Dict]) -> Set[Tuple]:
        _tpls = set()
        for _jd in _jds:
            for _e in _jd['evidences']:
                for _page_idx in _e['page_indices']:
                    tpl = (_jd['document_id'],)
                    if keep_page:
                        tpl = tpl + (_page_idx,)
                    if keep_query:
                        tpl = tpl + (_e['query'],)
                    if keep_stance:
                        tpl = tpl + (_e['stance'],)
                    _tpls.add(tpl)
        return _tpls

    def _metric(_g_tpls, _s_tpls) -> Dict[str, float]:
        _prec = (len(_g_tpls & _s_tpls) / len(_s_tpls)) if _s_tpls else 0.
        _rec = (len(_g_tpls & _s_tpls) / len(_g_tpls)) if _g_tpls else 0.
        _f1 = ((2 * _prec * _rec) / (_prec + _rec)) if (_prec + _rec) > 0 else 0.
        assert 0 <= _prec <= 1
        assert 0 <= _rec <= 1
        assert 0 <= _f1 <= 1
        assert _f1 <= ((_prec + _rec) / 2) + 1e-7
        return {
            'g': len(_g_tpls),
            's': len(_s_tpls),
            'c': len(_g_tpls & _s_tpls),
            'p': process_score(_prec),
            'r': process_score(_rec),
            'f': process_score(_f1)
        }

    g_tpls = _get_tuples(_jds=gold_jds)
    s_tpls = _get_tuples(_jds=pred_jds)
    return _metric(_g_tpls=g_tpls, _s_tpls=s_tpls)


def strict_f1(
        gold_jds: List[Dict],
        pred_jds: List[Dict],
        keep_page: bool = True,
        keep_query: bool = True,
        keep_stance: bool = True
) -> Dict:

    def _get_tuples(_jds: List[Dict]) -> Set[Tuple]:
        _tpls = set()
        for _jd in _jds:
            for _e in _jd['evidences']:
                tpl = (_jd['document_id'],)
                if keep_page:
                    tpl = tpl + (tuple(sorted(_e['page_indices'])),)
                if keep_query:
                    tpl = tpl + (_e['query'],)
                if keep_stance:
                    tpl = tpl + (_e['stance'],)
                _tpls.add(tpl)
        return _tpls

    def _metric(_g_tpls, _s_tpls) -> Dict[str, float]:
        _prec = (len(_g_tpls & _s_tpls) / len(_s_tpls)) if _s_tpls else 0.
        _rec = (len(_g_tpls & _s_tpls) / len(_g_tpls)) if _g_tpls else 0.
        _f1 = ((2 * _prec * _rec) / (_prec + _rec)) if (_prec + _rec) > 0 else 0.
        assert 0 <= _prec <= 1
        assert 0 <= _rec <= 1
        assert 0 <= _f1 <= 1
        assert _f1 <= ((_prec + _rec) / 2) + 1e-7
        return {
            'g': len(_g_tpls),
            's': len(_s_tpls),
            'c': len(_g_tpls & _s_tpls),
            'p': process_score(_prec),
            'r': process_score(_rec),
            'f': process_score(_f1)
        }

    g_tpls = _get_tuples(_jds=gold_jds)
    s_tpls = _get_tuples(_jds=pred_jds)
    return _metric(_g_tpls=g_tpls, _s_tpls=s_tpls)


def evaluate_document_f1(gold_jds: List[Dict], pred_jds: List[Dict]) -> Dict:
    result = {
        'page': document_level_f1(
            gold_jds=gold_jds, pred_jds=pred_jds, keep_page=True, keep_query=False, keep_stance=False),
        'query': document_level_f1(
            gold_jds=gold_jds, pred_jds=pred_jds, keep_page=False, keep_query=True, keep_stance=False),
        'stance': document_level_f1(
            gold_jds=gold_jds, pred_jds=pred_jds, keep_page=False, keep_query=False, keep_stance=True),
    }
    return result


def evaluate_strict_f1(gold_jds: List[Dict], pred_jds: List[Dict]) -> Dict:
    result = {
        'all': strict_f1(
            gold_jds=gold_jds, pred_jds=pred_jds, keep_page=True, keep_query=True, keep_stance=True),
        'page': strict_f1(
            gold_jds=gold_jds, pred_jds=pred_jds, keep_page=True, keep_query=False, keep_stance=False),
        'query': strict_f1(
            gold_jds=gold_jds, pred_jds=pred_jds, keep_page=True, keep_query=True, keep_stance=False),
        'stance': strict_f1(
            gold_jds=gold_jds, pred_jds=pred_jds, keep_page=True, keep_query=False, keep_stance=True),
    }
    return result


def relieve_stance(jds: List[Dict]) -> List[Dict]:
    jds = copy.deepcopy(jds)
    label_map = {
        'strongly_supporting': 'supporting',
        'supporting': 'supporting',
        'no_position_or_mixed_position': 'no_position',
        'not_supporting': 'not_supporting',
        'opposing': 'not_supporting',
    }
    for jd in jds:
        for evidence in jd['evidences']:
            evidence['stance'] = label_map[evidence['stance']]
    return jds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g",
        type=str,
        help="The gold jsonline file",
    )
    parser.add_argument(
        "-s",
        type=str,
        help="The system prediction jsonline file",
    )
    parser.add_argument(
        "--relieve_stance",
        type=strtobool,
        default=False,
        help="Whether to convert 5-class stance labels into 3-class labels",
    )
    args = parser.parse_args()

    with open(args.g) as o:
        gold_jds = [json.loads(l) for l in o.readlines() if l.strip()]

    with open(args.s) as o:
        pred_jds = [json.loads(l) for l in o.readlines() if l.strip()]

    if args.relieve_stance:
        gold_jds = relieve_stance(jds=gold_jds)
        pred_jds = relieve_stance(jds=pred_jds)

    result = {
        'document': evaluate_document_f1(gold_jds=gold_jds, pred_jds=pred_jds),
        'overlap': evaluate_overlap_f1(gold_jds=gold_jds, pred_jds=pred_jds),
        'strict': evaluate_strict_f1(gold_jds=gold_jds, pred_jds=pred_jds),
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
