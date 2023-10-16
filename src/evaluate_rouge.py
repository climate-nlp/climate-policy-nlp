import evaluate
import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g",
        type=str,
        help="The gold json file",
    )
    parser.add_argument(
        "-s",
        type=str,
        help="The system prediction json file",
    )
    args = parser.parse_args()

    with open(args.g, 'r') as o:
        gold_jds = json.loads(o.read())
    with open(args.s, 'r') as o:
        pred_jds = json.loads(o.read())

    g_key2out = dict()
    for g in gold_jds:
        key = (g['document_id'], g['query'], g['stance'], tuple(g['page_indices']))
        g_key2out[key] = g['output']

    s_key2out = dict()
    for s in pred_jds:
        key = (s['document_id'], s['query'], s['stance'], tuple(s['page_indices']))
        s_key2out[key] = s['output']

    g_keys = set(g_key2out.keys())
    s_keys = set(s_key2out.keys())

    assert g_keys
    assert s_keys
    assert len(g_keys & s_keys) == len(g_keys) == len(s_keys)

    pred_str, label_str = [], []
    for key in g_keys:
        pred_str.append(s_key2out[key])
        label_str.append(g_key2out[key])

    assert len(pred_str) == len(label_str)

    rouge = evaluate.load('rouge')
    eval_scores = rouge.compute(
        predictions=pred_str,
        references=label_str,
    )
    print(json.dumps(eval_scores))


if __name__ == "__main__":
    main()
