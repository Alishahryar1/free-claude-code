# cl100k_base

The tokenizer data and encoding definition are copied from OpenAI tiktoken
0.14.0 under the accompanying MIT license.

- Definition: https://github.com/openai/tiktoken/blob/0.14.0/tiktoken_ext/openai_public.py
- Data: https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken
- SHA256: 223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7
- Encoding-data license clarification: https://github.com/openai/tiktoken/issues/92#issuecomment-1497875652

FCC loads the packaged file locally, including when tiktoken's cache is empty.
When updating this definition or data, verify exact encoding parity.
