import torch
import torch.nn.init as init
# train_mohawk_stage1.py
import sys
import pathlib
import matplotlib.pyplot as plt 
from collections import defaultdict
import numpy as np 
import os 

import shutil
# ----------------------------------------------------------------------
# Ensure the project root (one level up from `scripts`) is on sys.path
# ----------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[1]   # ../ from this file
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from transformers import AutoTokenizer

from modules.lm_head import LMHeadModel
from modules.modeling_phi import PhiForCausalLM
from utils.config import Config

device = "cuda"

teacher_model = PhiForCausalLM.from_pretrained(
    "microsoft/phi-1_5", attn_implementation="eager"
).to(device)
teacher_model.eval()
teacher_model.requires_grad_(False)

model_config = Config.from_json("assets/sample_config.json")
student_model = LMHeadModel(model_config).to(device)

dataset = load_dataset("stas/openwebtext-10k", trust_remote_code=True)["train"]

# TODO : This should to be used instead of single inputs
dataloader = DataLoader(dataset, batch_size=4)

# Adding an Optimizer : 
optimizer = Adam(student_model.parameters(),lr=2E-5)

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")
tokenizer.pad_token = tokenizer.eos_token  # Set padding token

# TODO : Pretokenize the dataset 

# Stage 1 skeleton
student_model.requires_grad_(True)
mean_iteration_losses = []
token_counts = []
total_tokens = 0 
for idx, batch in enumerate(dataloader):
    input_ids = (
        tokenizer(batch["text"], return_tensors="pt", truncation=True, padding=True)
        .to(device)
        .input_ids
    )

    # Count tokens in this batch (excluding padding tokens)
    batch_tokens = (input_ids != tokenizer.pad_token_id).sum().item()
    total_tokens += batch_tokens

    # Zero your gradients for every batch!
    optimizer.zero_grad()

    _, seq_len = input_ids.size()
    with torch.no_grad(): 
        teacher_outputs = teacher_model(
            input_ids=input_ids,
            output_hidden_states=True,
            output_attention_results=True,
            output_attentions=True,
            use_cache=False,
        )

    mean_layer_loss = []
    for layer_idx, student_layer in enumerate(student_model.backbone.layers):
        
        student_input = teacher_outputs.all_hidden_states[layer_idx]

        # Forward pass
        student_output = student_layer(
            hidden_states=student_input,
            run_mlp_component=False,
            return_mixer_matrix=True,
        )
        transfer_matrix = student_output["transfer_matrix"]
        attn_matrix = teacher_outputs.all_attn_matrices[layer_idx]

        assert transfer_matrix.size() == attn_matrix.size()

        loss = torch.linalg.matrix_norm(
            transfer_matrix - attn_matrix, ord="fro"
        ).mean()

        loss.backward()

        # Adjust learning weights
        optimizer.step()
        mean_layer_loss.append(loss.item())
        print(f"Iter {idx}, Layer {layer_idx}, Loss: {loss.item()}")

    token_counts.append(total_tokens)
    mean_iteration_losses.append(np.mean(mean_layer_loss))
    # Plot the results

plt.figure(figsize=(10, 6))
plt.plot(token_counts, mean_iteration_losses, 'b-', linewidth=2, label='Mean Attention L2 Loss')
plt.xlabel('Number of Tokens')
plt.ylabel('Attention L2 Loss')
plt.title('Training Progress: Mean Attention Loss vs Tokens Processed')
plt.grid(True, alpha=0.3)
plt.legend()

# Format x-axis to show tokens in K/M
def format_tokens(x, pos):
    if x >= 1_000_000:
        return f'{x/1_000_000:.1f}M'
    elif x >= 1_000:
        return f'{x/1_000:.1f}K'
    else:
        return f'{int(x)}'

from matplotlib.ticker import FuncFormatter
plt.gca().xaxis.set_major_formatter(FuncFormatter(format_tokens))


# Save the plot
plt.savefig('training_loss_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# Also save the data for later analysis
np.save('loss_data.npy', {'losses': mean_iteration_losses, 'token_counts': token_counts})

# Create a directory for the saved model
save_dir = "saved_models/student_mohawk_stage1"
os.makedirs(save_dir, exist_ok=True)

# Save the model state dict
torch.save(student_model.state_dict(), f"{save_dir}/model_state_dict.pth")

# Copy the original config file
shutil.copy("assets/sample_config.json", f"{save_dir}/config.json")

# Optional: Save the entire model (larger file size)
torch.save(student_model, f"{save_dir}/complete_model.pth")

print(f"Model saved to {save_dir}")

print("DONE")