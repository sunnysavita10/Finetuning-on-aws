from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch, json

def model_fn(model_dir):
    # Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True
    )

    text_gen = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.8
    )
    return text_gen


def predict_fn(input_data, model):
    # Normalize payload
    if isinstance(input_data, dict) and "inputs" in input_data:
        prompt = input_data["inputs"]
    else:
        prompt = str(input_data)

    # Inference
    result = model(prompt)[0]["generated_text"]
    return {"generated_text": result}


def output_fn(prediction, accept):
    return json.dumps(prediction)
