from llama_index.llms.bedrock_converse import BedrockConverse

def retrieve_llm_answer(prompt, provider="aws", execution_profile="default", model_id="anthropic.claude-3-haiku-20240307-v1:0"):
    if provider == "aws":
        resp = BedrockConverse(
            model=model_id,
            profile_name=execution_profile,
        ).complete(prompt)

        print(resp)
    else:
        # TODO: use other implementations in llama_index.llms to support multiple platforms
        raise ValueError(f"LLM Driven Distillation is not implemented for provider -> {provider}")

    return resp