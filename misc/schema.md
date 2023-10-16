# JSON schema of the dataset

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Generated schema for Root",
  "type": "object",
  "properties": {
    "document_id": {
      "type": "string"
    },
    "sentences": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "sentence_id": {
            "type": "number"
          },
          "page_idx": {
            "type": "number"
          },
          "block_idx": {
            "type": "number"
          },
          "block_sentence_idx": {
            "type": "number"
          },
          "text": {
            "type": "string"
          }
        },
        "required": [
          "sentence_id",
          "page_idx",
          "block_idx",
          "block_sentence_idx",
          "text"
        ]
      }
    },
    "evidences": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "query": {
            "type": "string"
          },
          "stance": {
            "type": "string"
          },
          "page_indices": {
            "type": "array",
            "items": {
              "type": "number"
            }
          },
          "comment": {
            "type": "string"
          }
        },
        "required": [
          "query",
          "stance",
          "page_indices",
          "comment"
        ]
      }
    },
    "meta": {
      "type": "object",
      "properties": {
        "parser": {
          "type": "string"
        },
        "evidences": {
          "type": "array",
          "items": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "company_name": {
                  "type": "string"
                },
                "company_influencemap_url": {
                  "type": "string"
                },
                "evidence_url": {
                  "type": "string"
                },
                "evidence_query": {
                  "type": "string"
                },
                "evidence_data_source": {
                  "type": "string"
                },
                "evidence_region": {
                  "type": "string"
                },
                "evidence_year": {
                  "type": "number"
                },
                "evidence_score_for_this_evidence_item": {
                  "type": "number"
                },
                "evidence_title": {
                  "type": "string"
                },
                "evidence_influencemap_comment": {
                  "type": "string"
                },
                "evidence_extract_from_source": {
                  "type": "string"
                },
                "evidence_pdf_urls": {
                  "type": "array",
                  "items": {
                    "type": "string"
                  }
                },
                "evidence_pdf_filenames": {
                  "type": "array",
                  "items": {
                    "type": "string"
                  }
                },
                "evidence_external_link": {
                  "type": "string"
                },
                "evidence_timestamps": {
                  "type": "string"
                },
                "sentence_ids": {
                  "type": "array",
                  "items": {
                    "type": "number"
                  }
                }
              },
              "required": [
                "company_name",
                "company_influencemap_url",
                "evidence_url",
                "evidence_query",
                "evidence_data_source",
                "evidence_region",
                "evidence_year",
                "evidence_score_for_this_evidence_item",
                "evidence_title",
                "evidence_influencemap_comment",
                "evidence_extract_from_source",
                "evidence_pdf_urls",
                "evidence_pdf_filenames",
                "evidence_external_link",
                "evidence_timestamps",
                "sentence_ids"
              ]
            }
          }
        }
      },
      "required": [
        "parser",
        "evidences"
      ]
    }
  },
  "required": [
    "document_id",
    "sentences",
    "evidences",
    "meta"
  ]
}
```