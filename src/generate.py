"""
Step 9 - Generate a grounded German answer with citations.

The LLM gets the question plus the top passages from retrieval, numbered [1] ... [k].
The prompt enforces three rules:
  1. answer ONLY from the passages (no outside knowledge)  -> groundedness
  2. cite every statement with the passage number           -> traceability
  3. say a fixed sentence if the passages don't contain it  -> no hallucinated answers
Afterwards we map [n] back to "document + page" for the user.
"""
import re

NO_ANSWER = "Dazu finde ich in den Dokumenten keine Information."

SYSTEM_PROMPT = f"""Du bist der IT-Wissensassistent des Klinikums Musterstadt.
Beantworte die Frage ausschließlich mit Informationen aus den nummerierten Auszügen.

Regeln:
1. Antworte auf Deutsch, sachlich und knapp, in höchstens vier Sätzen.
2. Belege jede Aussage mit der Nummer des Auszugs in eckigen Klammern, zum Beispiel [2].
3. Verwende kein Wissen außerhalb der Auszüge und erfinde nichts.
4. Wenn die Auszüge die Frage nicht beantworten, antworte genau mit: {NO_ANSWER}"""


def build_messages(question, passages):
    """passages: list of chunk dicts (doc, page, title, text), best first."""
    blocks = []
    for n, c in enumerate(passages, start=1):
        doc = c["doc"].removesuffix(".pdf").replace("_", " ")
        blocks.append(f"[{n}] {doc}, Seite {c['page']} – {c['title']}:\n{c['text']}")
    user = "Auszüge:\n\n" + "\n\n".join(blocks) + f"\n\nFrage: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def parse_citations(answer, n_passages):
    """Find [1], [2, 3], [1][4] ... Returns (valid passage numbers in order, invalid numbers)."""
    nums = []
    for group in re.findall(r"\[([\d,\s]+)\]", answer):
        nums += [int(x) for x in re.findall(r"\d+", group)]
    valid = list(dict.fromkeys(n for n in nums if 1 <= n <= n_passages))
    invalid = sorted({n for n in nums if not 1 <= n <= n_passages})
    return valid, invalid


def is_abstention(answer):
    """Did the model say it can't answer? Tolerant to small wording changes."""
    a = answer.lower()
    return "keine information" in a or "finde ich in den dokumenten nicht" in a


class Generator:
    """A local Hugging Face instruct model. Greedy decoding: same input -> same answer."""

    def __init__(self, model_id, load_4bit=False, device_map="auto", dtype=None):
        """device_map="auto" spreads the model over all visible GPUs (Kaggle: 2x T4).
        device_map=None loads it normally, so the caller can move it with .to("cuda") (ZeroGPU)."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id)
        kwargs = {"device_map": device_map} if device_map else {}
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
            kwargs["dtype"] = torch.float16
            # Loading briefly holds 16-bit copies of weights before quantizing them, so leave
            # headroom on every GPU instead of letting device_map="auto" fill one GPU completely.
            kwargs["max_memory"] = {i: "9GiB" for i in range(torch.cuda.device_count())}
        else:
            if dtype is None:
                # bfloat16 needs GPU compute capability >= 8 (A100, H100 ...). The T4 is 7.5 -> float16.
                major = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
                dtype = torch.bfloat16 if major >= 8 else torch.float16
            kwargs["dtype"] = dtype
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.model.eval()

    def generate(self, messages, max_new_tokens=300):
        inputs = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                              return_dict=True, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                  pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        return self.tok.decode(new_tokens, skip_special_tokens=True).strip()
