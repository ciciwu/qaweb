from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from torch.optim import Adam
import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

TEST_Q = "Retinol"
DEVICE= torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE=2

class DomainData(Dataset):
    def __init__(self, tokenizer):
        self.data =  [
        'Retinol is a fat-soluble vitamin in the vitamin A family that is found in food and used as a dietary supplement. Retinol or other forms of vitamin A are needed for vision, cellular development, maintenance of skin and mucous membranes, immune function and reproductive development.',
        'Retinol helps neutralize free radicals in the middle layer of your skin. This can help reduce the appearance of wrinkles and enlarged pores. It may also reduce symptoms in people with some skin conditions.',
        'Salicylic acid (SA) is a beta-hydroxy acid (BHA) that helps promote the skin\'s natural exfoliation process. Originally derived from the bark of certain plants—such as white willow and wintergreen leaf—salicylic acid is most often created in a lab today.'
        ]

        #Preprocess with termation token 
        self.X = []
        for question in enumerate(self.data):
            self.X.append(  f"{question} {tokenizer.eos_token}")

        self.X_encoded = tokenizer(self.X,max_length=40, add_special_tokens=False,padding=True, return_tensors="pt")
        self.input_ids = self.X_encoded['input_ids']
        self.attention_mask = self.X_encoded['attention_mask']

    def __len__(self):
            return len(self.X)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_mask[idx])


tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token 

model = GPT2LMHeadModel.from_pretrained("distilgpt2", pad_token_id=tokenizer.pad_token_id)

data = DomainData(tokenizer)
train_dataloader = DataLoader(
            data,  
            batch_size = BATCH_SIZE
        )

optimizer = AdamW(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

model.to(DEVICE)

def generate_response(question):
    # Encode the question using the tokenizer
    question = f"{question}"
    input_ids = tokenizer.encode(question  , return_tensors="pt").to(DEVICE)
    # Generate the answer using the model
    sample_outputs = model.generate(input_ids, do_sample=True, max_length=150, top_k=10, top_p=0.99,num_return_sequences=5)
    # Decode the generated answer using the tokenizer
    answer = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
    return answer

model.train()

for epoch in range(2):
    print(f"training epoch {epoch}")
    print(generate_response(TEST_Q))
    for X, attnt in train_dataloader:
        X = X.to(DEVICE)
        attnt = attnt.to(DEVICE)
        optimizer.zero_grad()
        loss = model(X, labels=X, attention_mask=attnt).loss
        print(loss)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
model.eval()

response = generate_response(TEST_Q)
print("===================")
print(f"{response}")