from fastapi import FastAPI, HTTPException
# from langchain.prompt import ChatPrompt
# from langserve import add_routes
import uvicorn
import os

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, PegasusForConditionalGeneration, PegasusTokenizer
from peft import PeftModel
from pydantic import BaseModel

app = FastAPI(
    title="IMDB Server",
    version="1.0",
    description="Simple IMDB Review Summarizer and Sentiment Analysis"
)

# add_routes(
#     app,

# )

device = 'cuda' if torch.cuda.is_available() else 'cpu'

## Summarize
summarize_model = 'tuner007/pegasus_summarizer'
pegasus_tokenizer = PegasusTokenizer.from_pretrained(summarize_model)
pegasus_model = PegasusForConditionalGeneration.from_pretrained(summarize_model).to(device)

class SummarizationRequest(BaseModel):
    text: str

def get_summarize(input_text):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch = pegasus_tokenizer([input_text], truncation=True, padding='longest', max_length=1024, return_tensors="pt").to(device)
    gen_out = pegasus_model.generate(
        **batch,
        max_length=128,
        num_beams=3,
        temperature=0.7,        # Lower temperature to reduce randomness
        top_k=50,               # Use top_k sampling
        top_p=0.95,             # Use top_p sampling
        repetition_penalty=2.0  # Penalize repetition
    )
    output_text = pegasus_tokenizer.batch_decode(gen_out, skip_special_tokens=True)
    return output_text[0]

@app.post("/summarize/")
async def summarize(request: SummarizationRequest):
    try:
        summary = get_summarize(request.text)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


## Sentiment

# sentiment_base_model = AutoModelForSeq2SeqLM.from_pretrained(
#     'google/flan-t5-base', 
#     torch_dtype=torch.bfloat16,
#     trust_remote_code=True
# )

# sentiment_model = PeftModel.from_pretrained(sentiment_base_model, '../peft-dialogue-summary-checkpoint-local/')
# sentiment_model = sentiment_model.merge_and_unload().to(device)
# tokenizer_peft = AutoTokenizer.from_pretrained("./peft-dialogue-summary-checkpoint-local/", trust_remote_code=True)
# sentiment_model.pad_token_id = tokenizer_peft.pad_token_id
# sentiment_model.config.pad_token_id = tokenizer_peft.pad_token_id


# generation_config = GenerationConfig(
#     max_new_tokens=1,
#     num_beams=2,      # Increase the number of beams for better exploration but more computation need
#     temperature=0.0,  # Set to 0 to eliminate randomness
#     top_k=1,          # Setting top_k to 1 ensures only the most probable token is considered
#     top_p=1.0         # Setting top_p to 1.0 effectively disables nucleus sampling
# )

# def get_sentiment(text):
#     prompt = f"""
#     classify sentiment: 
#     {text}
    
#     Sentiment:
#     """

#     # Tokenize the input and move to the same device
#     input_ids = tokenizer_peft(prompt, 
#                                return_tensors='pt',
#                                max_length=512,
#                                truncation=True,
#                                padding='max_length'
#     ).input_ids.to(device)

#     # Generate output with the PEFT model
#     with torch.no_grad():
#         peft_output = sentiment_model.generate(input_ids=input_ids, generation_config=generation_config)
    
#     sentiment_model_output = tokenizer_peft.decode(peft_output[0], skip_special_tokens=True)

#     return(sentiment_model_output)

if __name__=="__main__":
   uvicorn.run(app, host="localhost", port=8000)
