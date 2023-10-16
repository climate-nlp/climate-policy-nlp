import json
from dataclasses import dataclass, field
from transformers import HfArgumentParser


@dataclass
class Arguments:
    """
    Arguments
    """
    input_file: str = field(
        default='./log/longformer/3_classify_stance/predict_results_None.jsonl',
        metadata={'help': 'The prediction data (jsonline) file path'},
    )
    output_file: str = field(
        default=None,
        metadata={'help': 'The output (jsonline) file path'},
    )


def main(conf: Arguments):
    if conf.input_file.endswith('.jsonl'):
        with open(conf.input_file, 'r') as f:
            jds = [json.loads(l) for l in f.readlines() if l.strip()]
    else:
        with open(conf.input_file, 'r') as f:
            jds = json.loads(f.read())

    for jd in jds:

        added_labels = set()
        new_evidences = []

        for evidence in jd['evidences']:
            label = (evidence['query'], evidence['stance'])

            if label in added_labels:
                added_evidence = [e for e in new_evidences if (e['query'], e['stance']) == label]
                assert len(added_evidence) == 1
                added_evidence = added_evidence[0]

                added_evidence['page_indices'] += evidence['page_indices']
                added_evidence['page_indices'] = sorted(list(set(added_evidence['page_indices'])))

                if 'comment' in added_evidence and evidence['comment'] not in added_evidence['comment']:
                    added_evidence['comment'] += '\n' + evidence['comment']
            else:
                new_evidences.append(evidence)

            added_labels.add(label)

        jd['evidences'] = new_evidences

    with open(conf.output_file, 'w') as f:
        for jd in jds:
            f.write(f'{json.dumps(jd, ensure_ascii=False)}\n')
    return


if __name__ == '__main__':
    parser = HfArgumentParser(Arguments)
    conf = parser.parse_args_into_dataclasses()[0]
    main(conf)
