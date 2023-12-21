import os
import argparse
import re
import tempfile
import copy
import datetime
from typing import List, Dict
import json

import numpy as np
import datasets
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
import torch
import nltk
import fitz


def read_pdf_by_doctr(pdf_file_path: str, ocr_model=None, max_pages: int = 100, **kwargs) -> List[Dict]:
    """
    Reads a PDF file and performs OCR on it using Doctr OCR model. Returns a list of dictionaries where each dictionary
    represents a block of text in the PDF.

    Parameters
    ----------
    pdf_file_path : str
        The path to the PDF file to be read and OCR performed on.
    ocr_model : Doctr model
        The Doctr OCR model to be used.
    max_pages : int, optional
        The maximum number of pages to OCR, defaults to 100.

    Returns
    -------
    List[Dict]
        A list of dictionaries where each dictionary represents a block of text in the PDF. Each dictionary has the
        following keys:
        - 'page_idx': the page index number where the block appears.
        - 'data': a dictionary representing the text block as returned by the OCR model.
        - 'sentences': a list of sentence texts extracted from the text block. Sentences are tokenized by NLTK.

    Notes
    -----
    1. The function will throw an exception where the PDF file is not valid or cannot be opened
    by doctr.io.DocumentFile.from_pdf(). The function handles this exception by returning an empty list.
    """
    import doctr

    assert os.path.exists(pdf_file_path)
    assert max_pages > 0
    assert ocr_model is not None

    try:
        doc = doctr.io.DocumentFile.from_pdf(pdf_file_path)
    except Exception as e:
        print(e)
        return []

    if len(doc) > max_pages:
        doc = doc[:max_pages]

    result = ocr_model(doc)

    text_blocks = []
    for page in result.pages:
        for block in page.blocks:
            block_text = ''
            for line in block.lines:
                for word in line.words:
                    block_text += word.value + ' '

            text_blocks.append({
                'page_idx': page.page_idx,
                'data': block.export(),
                'sentences': nltk.sent_tokenize(block_text)
            })

    return text_blocks


def read_pdf_by_fitz(pdf_file_path: str, max_pages: int = 100, **kwargs) -> List[Dict]:
    """
    Reads a PDF file and extracts text from it using PyMuPDF's fitz package. Returns a list of dictionaries where each
    dictionary represents a block of text in the PDF.

    Parameters
    ----------
    pdf_file_path : str
        The path to the PDF file to be read and text extracted from.
    max_pages : int, optional
        The maximum number of pages to extract text from, defaults to 100.

    Returns
    -------
    List[Dict]
        A list of dictionaries where each dictionary represents a block of text in the PDF. Each dictionary has the
        following keys:
        - 'page_idx': the page index number where the block appears.
        - 'data': a dictionary representing the text block as returned by fitz.
        - 'sentences': a list of sentence texts extracted from the text block. Sentences are tokenized by NLTK.

    Notes
    -----
    1. The function will throw an exception where the PDF file is not valid or cannot be opened by fitz.open().
    The function handles this exception by returning an empty list.
    2. The function will throw an exception if the text extraction fails for any reason.
    The function handles this exception by returning an empty list.
    """
    assert os.path.exists(pdf_file_path)
    assert max_pages > 0

    text_blocks = []

    try:
        doc = fitz.open(pdf_file_path)
    except:
        return []

    try:
        for i_page, page in enumerate(doc):
            if i_page >= max_pages:
                break

            blocks = page.get_text('dict')["blocks"]
            # blocks.sort(key=lambda x: x["bbox"][1])
            for block in blocks:
                if 'lines' not in block:
                    continue

                block_text = ''
                for line in block["lines"]:
                    for span in line["spans"]:
                        block_text += span["text"] + ' '

                text_blocks.append({
                    'page_idx': i_page,
                    'data': block,
                    'sentences': nltk.sent_tokenize(block_text)
                })
        doc.close()
    except:
        doc.close()
        return []

    return text_blocks


def read_pdf_by_tesseract(pdf_file_path: str, max_pages: int = 100, **kwargs) -> List[Dict]:
    """
    Reads a PDF file and extracts text from it using Tesseract OCR. Returns a list of dictionaries where each dictionary
    represents a block of text in the PDF.

    Parameters
    ----------
    pdf_file_path : str
        The path to the PDF file to be read and text extracted from.
    max_pages : int, optional
        The maximum number of pages to extract text from, defaults to 100.

    Returns
    -------
    List[Dict]
        A list of dictionaries where each dictionary represents a block of text in the PDF. Each dictionary has the
        following keys:
        - 'page_idx': the page index number where the block appears.
        - 'data': a dictionary representing the text block as a bounding box and rotation angle.
        - 'sentences': a list of sentence texts extracted from the text block. Sentences are tokenized by NLTK.

    Notes
    -----
    1. The function will throw an exception where the PDF file is not valid or cannot be opened by fitz.open().
    The function handles this exception by returning an empty list.
    2. The function will throw an exception if the OCR processing fails for any reason.
    The function handles this exception by skipping the current page.
    3. The function assumes that each text block has a unique block number, and merges all blocks with the same block
    number into a single block. However, this assumption may not always hold, which could result in duplicated or
    missing text blocks.
    """
    import pytesseract
    import imutils
    from PIL import Image

    assert os.path.exists(pdf_file_path)
    assert max_pages > 0

    text_blocks = []

    try:
        doc = fitz.open(pdf_file_path)
    except Exception as e:
        print(e)
        return []

    try:
        for i_page, page in enumerate(doc):
            if i_page >= max_pages:
                break

            # Convert page to an image, rotate if necessary, and run OCR.
            try:
                pix = page.get_pixmap(dpi=72 * 3)
                with tempfile.TemporaryDirectory() as temp_dir:
                    image_input = os.path.join(temp_dir, 'temp.png')
                    pix.save(image_input)

                    image = Image.open(image_input).convert('RGB')
                    rotate_info = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
                    open_cv_image = np.array(image)
                    open_cv_image = open_cv_image[:, :, ::-1].copy()
                    rotated_image = imutils.rotate_bound(open_cv_image, angle=rotate_info["rotate"])

                    ocr_result = pytesseract.image_to_data(rotated_image, output_type=pytesseract.Output.DICT)
            except Exception as e:
                print(e)
                continue

            # Merge OCR results into text blocks
            blocks = []
            prev_block_num = -1
            for block_num, x, y, w, h, text in zip(
                    ocr_result['block_num'], ocr_result['left'], ocr_result['top'],
                    ocr_result['width'], ocr_result['height'],
                    ocr_result['text'],
            ):
                if prev_block_num != block_num:
                    assert prev_block_num < block_num
                    blocks.append({
                        'page_idx': i_page,
                        'bbox': [x, y, x + w, y + h],
                        'rotate': rotate_info["rotate"],
                        'text': text,
                    })
                    prev_block_num = block_num
                else:
                    blocks[-1]['bbox'][0] = min(blocks[-1]['bbox'][0], x)
                    blocks[-1]['bbox'][1] = min(blocks[-1]['bbox'][1], y)
                    blocks[-1]['bbox'][2] = max(blocks[-1]['bbox'][2], x + w)
                    blocks[-1]['bbox'][3] = max(blocks[-1]['bbox'][3], y + h)
                    blocks[-1]['text'] += ' ' + text

            # Remove empty blocks and tokenize text into sentences
            blocks = [b for b in blocks if b['text'].strip()]

            for block in blocks:
                text = block['text']
                text = re.sub(r'\s+', ' ', text)
                text = text.strip()

                block['sentences'] = nltk.sent_tokenize(text)
                block['data'] = {'bbox': block['bbox'], 'rotate': block['rotate']}

                block.pop('text')
                block.pop('bbox')

            text_blocks += blocks
    except Exception as e:
        print(e)
        pass

    try:
        doc.close()
    except Exception as e:
        print(e)
        pass

    return text_blocks


def convert_to_sentences(docs: List[Dict]) -> List[Dict]:
    input_sentences = []
    for i_block, doc in enumerate(docs):
        for i_sentence, sentence in enumerate(doc['sentences']):
            if sentence:
                input_sentences.append({
                    'sentence_id': len(input_sentences),
                    'page_idx': doc['page_idx'],
                    'block_idx': i_block,
                    'block_sentence_idx': i_sentence,
                    'text': sentence,
                })
    return input_sentences


def read_pdf(pdf_path: str, max_pages: int) -> List[Dict]:
    # Fitz
    print('Attempt (1) extracting text by fitz')
    docs = read_pdf_by_fitz(pdf_file_path=pdf_path, max_pages=max_pages)
    sentences = convert_to_sentences(docs)
    pdf_parser = 'fitz'

    # Tesseract
    if not sentences:
        try:
            print('Attempt (2) extracting text by tesseract')
            docs = read_pdf_by_tesseract(pdf_file_path=pdf_path, max_pages=max_pages)
            sentences = convert_to_sentences(docs)
            pdf_parser = 'tesseract'
        except Exception as e:
            print(e)

    # DocTr
    if not sentences:
        try:
            print('Attempt (3) extracting text by doctr')
            import doctr
            doctr_model = doctr.models.ocr_predictor(pretrained=True)
            if torch.cuda.is_available():
                doctr_model = doctr_model.cuda()
            docs = read_pdf_by_doctr(pdf_file_path=pdf_path, ocr_model=doctr_model, max_pages=max_pages)
            sentences = convert_to_sentences(docs)
            pdf_parser = 'doctr'
        except Exception as e:
            print(e)

    if not sentences:
        print('Failed to extract text from the pdf file')
        exit()

    print(f'Extracted sentences: {len(sentences)}')

    data_points = [{
        'document_id': pdf_path,
        'sentences': sentences,
        'evidences': [],
        'meta': {
            'parser': pdf_parser,
        },
    }]
    return data_points


def detect_evidence(data_points: List[Dict], pipe, batch_size: int) -> List[Dict]:
    doc_id_2_sentences = {d['document_id']: d['sentences'] for d in data_points}

    prepro_data_points = []
    for data_point in data_points:
        page_indices = sorted(list(set([s['page_idx'] for s in data_point['sentences']])))

        for page_idx in page_indices:
            page_sentences = [s for s in data_point['sentences'] if s['page_idx'] == page_idx]

            page_text = f' {pipe.tokenizer.sep_token} '.join(
                [s['text'] for s in page_sentences]
            )

            prepro_data_points.append({
                'document_id': data_point['document_id'],
                'page_idx': page_idx,
                'text': page_text,
            })

    predict_dataset = datasets.Dataset.from_list(prepro_data_points)

    scores = []
    for out in pipe(
            KeyDataset(predict_dataset, "text"),
            batch_size=batch_size,
            truncation="only_first",
            top_k=None,
    ):
        score = next(v for v in out if v['label'] == 'evidence')['score']
        scores.append(score)

    assert len(scores) == len(predict_dataset)
    predict_dataset = predict_dataset.add_column('score', scores)

    document_id_2_candidates = dict()
    for predict_data in predict_dataset:
        document_id = predict_data['document_id']

        if document_id not in document_id_2_candidates:
            document_id_2_candidates[document_id] = []
        document_id_2_candidates[document_id].append({
            "query": 'energy_transition_&_zero_carbon_technologies',  # Dummy!
            "stance": 'strongly_supporting',  # Dummy!
            "page_indices": [predict_data['page_idx']],
            "evidence_prob": predict_data['score']
        })

    predicted_data_points = []
    for document_id, output_candidates in document_id_2_candidates.items():
        outputs = [o for o in output_candidates if o['evidence_prob'] >= 0.5]

        # Workaround for the lobbymap data
        # If we do not find any evidence pages, we find the highest scored page as the evidence page
        min_prediction_per_doc = 1
        if min_prediction_per_doc > 0 and len(outputs) < min_prediction_per_doc:
            output_candidates = sorted(
                output_candidates,
                key=lambda x: x['evidence_prob'],
                reverse=True
            )
            outputs = output_candidates[:min_prediction_per_doc]

        outputs = sorted(outputs, key=lambda x: x['page_indices'][0])

        jd = {
            'document_id': document_id,
            'sentences': doc_id_2_sentences[document_id],
            'evidences': outputs,
            'meta': [{
                'predictor': os.path.basename(__file__),
                'predicted_at': str(datetime.datetime.now())
            }]
        }
        predicted_data_points.append(jd)

    return predicted_data_points


def classify_query(data_points: List[Dict], pipe, batch_size: int) -> List[Dict]:
    doc_id_2_sentences = {d['document_id']: d['sentences'] for d in data_points}

    prepro_data_points = []
    for data_point in data_points:
        for evidence in data_point['evidences']:
            assert len(evidence['page_indices']) == 1
            page_idx = evidence['page_indices'][0]

            page_sentences = [s for s in data_point['sentences'] if s['page_idx'] == page_idx]

            page_text = f' {pipe.tokenizer.sep_token} '.join(
                [s['text'] for s in page_sentences]
            )

            prepro_data_points.append({
                'document_id': data_point['document_id'],
                'page_idx': page_idx,
                'text': page_text,
                'evidence_prob': evidence['evidence_prob'],
            })

    predict_dataset = datasets.Dataset.from_list(prepro_data_points)

    labels_list = []
    for out in pipe(
            KeyDataset(predict_dataset, "text"),
            batch_size=batch_size,
            truncation="only_first",
            top_k=None,
    ):
        labels = [o['label'] for o in out if o['score'] >= 0.5]
        if labels:
            labels_list.append(labels)
        else:
            labels = [max(out, key=lambda o: o['score'])['label']]
            labels_list.append(labels)

    assert len(predict_dataset) == len(labels_list)

    document_id2output = dict()
    for index, query_labels in enumerate(labels_list):

        document_id = predict_dataset[index]['document_id']
        page_idx = predict_dataset[index]['page_idx']
        evidence_prob = predict_dataset[index]['evidence_prob']

        if document_id not in document_id2output:
            document_id2output[document_id] = []

        for query_label in query_labels:
            document_id2output[document_id].append({
                "query": query_label,
                "page_indices": [page_idx],
                "evidence_prob": evidence_prob,
            })

    predicted_data_points = []
    for document_id, outputs in document_id2output.items():
        jd = {
            'document_id': document_id,
            'sentences': doc_id_2_sentences[document_id],
            'evidences': outputs,
            'meta': [{
                'predictor': os.path.basename(__file__),
                'predicted_at': str(datetime.datetime.now())
            }]
        }
        predicted_data_points.append(jd)

    return predicted_data_points


def classify_stance(data_points: List[Dict], pipe, batch_size: int ) -> List[Dict]:
    doc_id_2_sentences = {d['document_id']: d['sentences'] for d in data_points}

    prepro_data_points = []
    for data_point in data_points:
        for evidence in data_point['evidences']:
            assert len(evidence['page_indices']) == 1
            page_idx = evidence['page_indices'][0]

            page_sentences = [s for s in data_point['sentences'] if s['page_idx'] == page_idx]
            page_text = f' {pipe.tokenizer.sep_token} '.join(
                [s['text'] for s in page_sentences]
            )
            query = evidence['query']

            prepro_data_points.append({
                'document_id': data_point['document_id'],
                'page_idx': page_idx,
                'query': query,
                'text': f"{query} {pipe.tokenizer.sep_token} {page_text}",
                'evidence_prob': evidence['evidence_prob']
            })

    predict_dataset = datasets.Dataset.from_list(prepro_data_points)

    labels = []
    for out in pipe(
            KeyDataset(predict_dataset, "text"),
            batch_size=batch_size,
            truncation="only_first",
    ):
        labels.append(out['label'])

    assert len(labels) == len(predict_dataset)
    predict_dataset = predict_dataset.add_column('label', labels)

    document_id2output = dict()
    for predict_data in predict_dataset:
        document_id = predict_data['document_id']

        if document_id not in document_id2output:
            document_id2output[document_id] = []

        document_id2output[document_id].append({
            "query": predict_data['query'],
            "stance": predict_data['label'],
            "page_indices": [predict_data['page_idx']],
            #"evidence_prob": predict_data['evidence_prob'],
        })

    predicted_data_points = []
    for document_id, outputs in document_id2output.items():
        jd = {
            'document_id': document_id,
            'sentences': doc_id_2_sentences[document_id],
            'evidences': outputs,
            'meta': [{
                'predictor': os.path.basename(__file__),
                'predicted_at': str(datetime.datetime.now())
            }]
        }
        predicted_data_points.append(jd)

    return predicted_data_points


def post_process(data_points: List[Dict]) -> List[Dict]:
    data_points = copy.deepcopy(data_points)

    for data_point in data_points:

        added_labels = set()
        new_evidences = []

        for evidence in data_point['evidences']:
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

        data_point['evidences'] = new_evidences

    return data_points


def main(args):
    print(f'Your setting: {args}')

    device = 0 if torch.cuda.is_available() else -1
    print(f'Using device "{device}"')

    # Load input pdf file
    data_points = read_pdf(pdf_path=args.pdf, max_pages=args.max_pages)

    # 1. Detect evidence
    print('Detecting evidence')
    data_points = detect_evidence(
        data_points=data_points,
        pipe=pipeline("text-classification", model=args.model_name_detect_evidence, device=device),
        batch_size=args.batch_size
    )

    # 2. Classify query
    print('Classifying query')
    data_points = classify_query(
        data_points=data_points,
        pipe=pipeline("text-classification", model=args.model_name_classify_query, device=device),
        batch_size=args.batch_size
    )

    # 3. Classify stance
    print('Classifying stance')
    data_points = classify_stance(
        data_points=data_points,
        pipe=pipeline("text-classification", model=args.model_name_classify_stance, device=device),
        batch_size=args.batch_size
    )

    # Post-process
    data_points = post_process(data_points=data_points)

    # Save
    output_file = args.pdf + '.jsonl'
    with open(output_file, 'w') as f:
        for data_point in data_points:
            f.write(f'{json.dumps(data_point, ensure_ascii=False)}\n')
    print(f'Result saved at {output_file}')
    print(f'The contained information:')
    print(f'\tdocument_id: the filename of the pdf file')
    print(f'\tsentences: extracted sentences of the pdf file')
    print(f'\tevidences: the output labels')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdf",
        type=str,
        help="The input pdf file",
    )
    parser.add_argument(
        "--max_pages",
        type=int,
        default=100,
        help="The maximum pages to read",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="The batch size",
    )
    parser.add_argument(
        "--model_name_detect_evidence",
        type=str,
        default='climate-nlp/longformer-large-4096-1-detect-evidence',
        help="The model name for the evidence detection",
    )
    parser.add_argument(
        "--model_name_classify_query",
        type=str,
        default='climate-nlp/longformer-large-4096-2-classify-query',
        help="The model name for the query classification",
    )
    parser.add_argument(
        "--model_name_classify_stance",
        type=str,
        default='climate-nlp/longformer-large-4096-3-classify-stance',
        help="The model name for the stance classification",
    )
    args = parser.parse_args()
    main(args)
