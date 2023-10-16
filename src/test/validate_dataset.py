import unittest
import json
import hashlib
import collections
from typing import Dict, List

score2label = {
    2: 'strongly_supporting',
    1: 'supporting',
    0: 'no_position_or_mixed_position',
    -1: 'not_supporting',
    -2: 'opposing'
}

query_label_list = [
    "communication_of_climate_science",
    "alignment_with_ipcc_on_climate_action",
    "supporting_the_need_for_regulations",
    "support_of_un_climate_process",
    "transparency_on_legislation",
    "carbon_tax",
    "emissions_trading",
    "energy_and_resource_efficiency",
    "renewable_energy",
    "energy_transition_&_zero_carbon_technologies",
    "ghg_emission_regulation",
    "disclosure_on_relationships",
    "land_use",
]

data_source_list = [
    'Media Reports',
    'Main Web Site',
    'CEO Messaging',
    'Direct Consultation with Governments',
    'Social Media',
    'CDP Responses',
    'Financial Disclosures'
]


def get_hash(d: List or Dict or str or int):
    dhash = hashlib.md5()
    dhash.update(json.dumps(d, sort_keys=True).encode())
    return dhash.hexdigest()


def read_json(data_path: str) -> List[Dict] or Dict:
    try:
        with open(data_path, 'r') as f:
            result = [json.loads(l) for l in f.readlines() if l.strip()]
    except:
        with open(data_path, 'r') as f:
            result = json.loads(f.read())

    return result


class ValidateDataset(unittest.TestCase):
    TRAIN_DATA_PATH = './data/lobbymap_dataset/train.jsonl'
    VALID_DATA_PATH = './data/lobbymap_dataset/valid.jsonl'
    TEST_DATA_PATH = './data/lobbymap_dataset/test.jsonl'

    TRAIN_COMMENT_DATA_PATH = './data/lobbymap_dataset/train.comment.json'
    VALID_COMMENT_DATA_PATH = './data/lobbymap_dataset/valid.comment.json'
    TEST_COMMENT_DATA_PATH = './data/lobbymap_dataset/test.comment.json'

    def test_content_duplication(self):

        for data_path in [self.TRAIN_DATA_PATH, self.VALID_DATA_PATH, self.TEST_DATA_PATH]:
            jds = read_json(data_path=data_path)
            document_ids = [jd['document_id'] for jd in jds]
            self.assertEqual(len(document_ids), len(set(document_ids)))
            input_hashes = [get_hash(jd['sentences']) for jd in jds]
            self.assertAlmostEqual(len(input_hashes), len(set(input_hashes)), delta=5)

            if len(input_hashes) != len(set(input_hashes)):
                hash_2_doc_id = {
                    get_hash(jd['sentences']): jd['document_id']
                    for jd in jds
                }
                duplicated_hashes = [h for h, cnt in collections.Counter(input_hashes).items() if cnt > 1]
                print(f'{data_path} content duplication:')
                print(f'ids:', [hash_2_doc_id[h] for h in duplicated_hashes])

    def test_evidence_duplication(self):

        for data_path in [self.TRAIN_DATA_PATH, self.VALID_DATA_PATH, self.TEST_DATA_PATH]:
            jds = read_json(data_path=data_path)

            for jd in jds:
                evidence_hashes = [get_hash(e) for e in sum(jd['meta']['evidences'], [])]
                self.assertEqual(len(evidence_hashes), len(set(evidence_hashes)))

    def test_evidence_meta_data(self):

        for data_path in [self.TRAIN_DATA_PATH, self.VALID_DATA_PATH, self.TEST_DATA_PATH]:
            jds = read_json(data_path=data_path)

            for jd in jds:
                sentence_ids = [s['sentence_id'] for s in jd['sentences']]
                self.assertEqual(len(sentence_ids), len(set(sentence_ids)))
                page_indices = sorted(list(set([s['page_idx'] for s in jd['sentences']])))

                self.assertTrue(len(sentence_ids) >= len(page_indices) > 0)

                self.assertEqual(len(jd['meta']['evidences']), len(jd['evidences']))

                for orig_evidence in sum(jd['meta']['evidences'], []):
                    self.assertEqual(len(orig_evidence['evidence_pdf_urls']), 1)
                    self.assertEqual(len(orig_evidence['evidence_pdf_filenames']), 1)
                    self.assertTrue(
                        # This is derived from a typo in origin LobbyMap data (maybe typo of 2015)
                        (orig_evidence['evidence_year'] == 105) or
                        (2005 <= orig_evidence['evidence_year'] <= 2023)
                    )
                    self.assertTrue(set(orig_evidence['sentence_ids']) & set(sentence_ids))
                    self.assertEqual(
                        len(set(orig_evidence['sentence_ids'])), len(orig_evidence['sentence_ids'])
                    )
                    self.assertTrue(max(orig_evidence['sentence_ids']) <= max(sentence_ids))
                    self.assertTrue(min(orig_evidence['sentence_ids']) >= min(sentence_ids))
                    self.assertIn(orig_evidence['evidence_data_source'], data_source_list)

                for orig_evidences, evidence in zip(jd['meta']['evidences'], jd['evidences']):
                    queries = [
                        e['evidence_query'].lower().replace(' ', '_') for e in orig_evidences
                    ]
                    self.assertTrue(len(set(queries)) == 1)
                    self.assertEqual(queries[0], evidence['query'])

                    stances = [
                        score2label[e['evidence_score_for_this_evidence_item']] for e in orig_evidences
                    ]
                    self.assertTrue(len(set(stances)) == 1)
                    self.assertEqual(stances[0], evidence['stance'])

                    self.assertIn(evidence['query'], query_label_list)
                    self.assertIn(evidence['stance'], score2label.values())

                    self.assertEqual(len(set(evidence['page_indices'])), len(evidence['page_indices']))
                    self.assertTrue(set(evidence['page_indices']) & set(page_indices))
                    self.assertTrue(max(evidence['page_indices']) <= max(page_indices))
                    self.assertTrue(min(evidence['page_indices']) >= min(page_indices))

    def test_train_test_split(self):

        jds = read_json(data_path=self.TEST_DATA_PATH)
        for jd in jds:
            years = [e['evidence_year'] for e in sum(jd['meta']['evidences'], [])]
            self.assertTrue(max(years) >= 2022)

        for data_path in [self.TRAIN_DATA_PATH, self.VALID_DATA_PATH]:
            jds = read_json(data_path=data_path)
            for jd in jds:
                years = [e['evidence_year'] for e in sum(jd['meta']['evidences'], [])]
                self.assertTrue(min(years) < 2022)

    def test_split_duplication(self):

        test_jds = read_json(data_path=self.TEST_DATA_PATH)
        test_document_ids = set([test_jd['document_id'] for test_jd in test_jds])
        test_doc_hashes = {
            get_hash(test_jd['sentences'])
            for test_jd in test_jds
        }

        for data_path in [self.TRAIN_DATA_PATH, self.VALID_DATA_PATH]:
            train_jds = read_json(data_path=data_path)

            train_document_ids = set([train_jd['document_id'] for train_jd in train_jds])
            self.assertFalse(train_document_ids & test_document_ids)

            train_doc_hashes = {
                get_hash(train_jd['sentences'])
                for train_jd in train_jds
            }
            self.assertFalse(train_doc_hashes & test_doc_hashes)

    def test_comment_content_duplication(self):

        for data_path in [
            self.TRAIN_COMMENT_DATA_PATH, self.VALID_COMMENT_DATA_PATH, self.TEST_COMMENT_DATA_PATH
        ]:
            jds = read_json(data_path=data_path)
            instances = [(jd['document_id'], jd['query'], jd['stance'], tuple(jd['page_indices'])) for jd in jds]
            self.assertEqual(len(instances), len(set(instances)))

    def test_comment_split_duplication(self):

        test_jds = read_json(data_path=self.TEST_COMMENT_DATA_PATH)
        test_document_ids = set([test_jd['document_id'] for test_jd in test_jds])
        test_doc_hashes = {
            get_hash(test_jd['input'])
            for test_jd in test_jds
        }
        test_hash_2_doc_id = {
            get_hash(test_jd['input']): test_jd['document_id']
            for test_jd in test_jds
        }

        for data_path in [self.TRAIN_COMMENT_DATA_PATH, self.VALID_COMMENT_DATA_PATH]:
            train_jds = read_json(data_path=data_path)

            train_document_ids = set([train_jd['document_id'] for train_jd in train_jds])
            self.assertFalse(train_document_ids & test_document_ids)

            train_doc_hashes = {
                get_hash(train_jd['input'])
                for train_jd in train_jds
            }
            train_hash_2_doc_id = {
                get_hash(train_jd['input']): train_jd['document_id']
                for train_jd in train_jds
            }
            # Ugly workaround..
            self.assertAlmostEqual(len(train_doc_hashes & test_doc_hashes), 0, delta=5)
            if len(train_doc_hashes & test_doc_hashes):
                print(f'Test and {data_path} content duplication:')
                duplicated_hashes = list(train_doc_hashes & test_doc_hashes)
                print(f'Test ids:', [test_hash_2_doc_id[h] for h in duplicated_hashes])
                print(f'Train ids:', [train_hash_2_doc_id[h] for h in duplicated_hashes])

        for comment_data_path, origin_data_path in zip(
                [self.TRAIN_DATA_PATH, self.VALID_DATA_PATH, self.TEST_DATA_PATH],
                [self.TRAIN_COMMENT_DATA_PATH, self.VALID_COMMENT_DATA_PATH, self.TEST_COMMENT_DATA_PATH]
        ):
            comment_jds = read_json(data_path=comment_data_path)
            origin_jds = read_json(data_path=origin_data_path)

            comment_document_ids = set([comment_jd['document_id'] for comment_jd in comment_jds])
            origin_document_ids = set([origin_jd['document_id'] for origin_jd in origin_jds])
            self.assertEqual(len(set(comment_document_ids) & set(origin_document_ids)), len(comment_document_ids))

