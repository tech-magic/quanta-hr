import json
from modules.llm.invoke_llm import retrieve_llm_answer
from tqdm import tqdm

def make_prompt(mode, chunk, heading):
    if mode=="summary":
        instruction = f"Summarize the following {heading} concisely in 2-4 sentences. If there are procedures or steps, list them."
        input = f"{heading}:\n{chunk}\n\n"
        prompt = f"{instruction}\n\n{input}Summary:"
    elif mode=="qa":
        # create a generic Q + answer prompt; you can later create multiple question variants with templates
        instruction = f"Given the following {heading}, generate a concise answer to the user's question: 'What does this {heading} say?'"
        input = f"{heading}:\n{chunk}\n\n"
        prompt = f"{instruction}\n\n{input}Answer:"
    else:
        raise ValueError("mode must be summary|qa")
    
    return instruction, input, prompt

def distill(preprocessed_jsonl, output_alpaca_jsonl, provider="aws", execution_profile="default", model_name="anthropic.claude-3-haiku-20240307-v1:0", mode="summary"):
    out_f = open(output_alpaca_jsonl, "w", encoding="utf-8")
    with open(preprocessed_jsonl, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            rec = json.loads(line)
            txt = rec["text"]
            heading = rec.get("heading", "Section")
            instruction, input, prompt = make_prompt(mode, txt, heading)
            g = retrieve_llm_answer(prompt=prompt, provider=provider, execution_profile=execution_profile, model_id=model_name)

            try:
                response_text = g.text
            except AttributeError:
                response_text = str(g)
            # keep only the generated part after prompt if needed
            # Here we take full generated_text and strip the prompt
            if response_text.startswith(prompt):
                output = response_text[len(prompt):].strip()
            else:
                output = response_text.strip()

            alpaca = {
                "instruction": instruction,
                "input": input,
                "output": output,
                "meta": {"doc_id": rec["doc_id"], "section_id": rec["section_id"], "heading": rec["heading"]}
            }

            print(f"Generated Alpaca Instruction -> {alpaca}")

            out_f.write(json.dumps(alpaca, ensure_ascii=False) + "\n")
    out_f.close()
    print("Saved distilled data to", output_alpaca_jsonl)
