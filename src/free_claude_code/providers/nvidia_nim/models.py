"""Maintained exclusions from NVIDIA's hosted chat model discovery.

Recheck these exact IDs when updating the list. Observed 404s describe the
tested hosted route and account, not permanent worldwide retirement. Remove
an exclusion when hosted chat is verified to work again. New IDs remain visible.
"""

HOSTED_MODEL_EXCLUSIONS = frozenset(
    {
        # Model-specific 404 on the tested hosted routes, checked 2026-10-13.
        "01-ai/yi-large",
        "adept/fuyu-8b",
        "ai21labs/jamba-1.5-large-instruct",
        "aisingapore/sea-lion-7b-instruct",
        "bigcode/starcoder2-15b",
        "databricks/dbrx-instruct",
        "deepseek-ai/deepseek-coder-6.7b-instruct",
        "google/codegemma-1.1-7b",
        "google/codegemma-7b",
        "google/deplot",
        "google/gemma-2b",
        "google/gemma-3-12b-it",
        "google/gemma-3-4b-it",
        "google/recurrentgemma-2b",
        "ibm/granite-3.0-3b-a800m-instruct",
        "ibm/granite-3.0-8b-instruct",
        "ibm/granite-34b-code-instruct",
        "ibm/granite-8b-code-instruct",
        "meta/codellama-70b",
        "meta/llama2-70b",
        "meta/muse-glimmer-30b",
        "microsoft/kosmos-2",
        "microsoft/phi-3-vision-128k-instruct",
        "microsoft/phi-3.5-moe-instruct",
        "mistralai/codestral-22b-instruct-v0.1",
        "mistralai/mistral-7b-instruct-v0.3",
        "mistralai/mistral-large",
        "mistralai/mistral-large-2-instruct",
        "mistralai/mixtral-8x22b-v0.1",
        "moonshotai/kimi-k2.6",
        "nv-mistralai/mistral-nemo-12b-instruct",
        "nvidia/cosmos-reason2-8b",
        "nvidia/embed-qa-4",
        "nvidia/llama-3.1-nemotron-51b-instruct",
        "nvidia/llama-3.1-nemotron-70b-instruct",
        "nvidia/llama-3.1-nemotron-ultra-253b-v1",
        "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1",
        "nvidia/llama-3.2-nv-embedqa-1b-v1",
        "nvidia/llama3-chatqa-1.5-70b",
        "nvidia/mistral-nemo-minitron-8b-8k-instruct",
        "nvidia/nemotron-4-340b-instruct",
        "nvidia/nemotron-4-340b-reward",
        "nvidia/nemotron-nano-3-30b-a3b",
        "nvidia/neva-22b",
        "nvidia/nv-embedqa-mistral-7b-v2",
        "nvidia/nvclip",
        "nvidia/riva-translate-4b-instruct",
        "nvidia/vila",
        "snowflake/arctic-embed-l",
        "writer/palmyra-creative-122b",
        "writer/palmyra-fin-70b-32k",
        "writer/palmyra-med-70b",
        "writer/palmyra-med-70b-32k",
        "zyphra/zamba2-7b-instruct",
        # Explicit retirement 410, checked 2026-10-13.
        "nvidia/ising-calibration-1-35b-a3b",
        # Non-chat services, checked 2026-10-13.
        "nvidia/ai-synthetic-video-detector",
        "nvidia/llama-nemotron-embed-vl-1b-v2",
        "nvidia/nemotron-3-embed-1b",
    }
)
