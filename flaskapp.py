from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
import torch
from flask import Flask, render_template, request


tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token 
model = GPT2LMHeadModel.from_pretrained("distilgpt2", pad_token_id=tokenizer.eos_token_id)

dm_tokenizer = GPT2Tokenizer.from_pretrained("./model_save")
dm_tokenizer.pad_token = dm_tokenizer.eos_token 
dm_model = GPT2LMHeadModel.from_pretrained("./model_save", pad_token_id=dm_tokenizer.eos_token_id)



# create the Flask app
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/answer", methods=["POST"])
def answer():
    # get the question from the web form
    question = request.form["question"]
    q = question + " is"
    
    # use the ChatGPT-2 model to generate an answer
    input_ids = tokenizer.encode(q, return_tensors="pt")
    output = model.generate(input_ids)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    # Domain knowledge trained model
    dm_input_ids = dm_tokenizer.encode(q, return_tensors="pt")
    dm_output = dm_model.generate(dm_input_ids )
    dm_answer = dm_tokenizer.decode(dm_output[0], skip_special_tokens=True)

    
    # return the answer to the web page
    return render_template("answer.html", question=question, answer=answer, dm_answer = dm_answer)


if __name__ == "__main__":
    app.run(debug=True)

