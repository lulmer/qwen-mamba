import sys
import pathlib
import shutil

# ----------------------------------------------------------------------
# Ensure the project root (one level up from `scripts`) is on sys.path
# ----------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[1]   # ../ from this file
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.init as init
from torch.optim import Adam

from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from modules.lm_head import LMHeadModel
from modules.modeling_phi import PhiForCausalLM
from utils.config import Config
from utils.plotting import plot_loss
import numpy as np 

device = "cuda"
experiment_name = "stage_2"

teacher_model = PhiForCausalLM.from_pretrained(
    "microsoft/phi-1_5", attn_implementation="eager"
).to(device)
teacher_model.eval()
teacher_model.requires_grad_(False)

model_config = Config.from_json("assets/sample_config.json")
student_model = LMHeadModel(model_config).to(device)

dataset = load_dataset("stas/openwebtext-10k", revision="refs/convert/parquet")["train"].select(range(8))
dataloader = DataLoader(dataset, batch_size=4)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")
tokenizer.pad_token = tokenizer.eos_token  # Set padding token

# Adding an Optimizer : 
optimizer = Adam(student_model.parameters(),lr=2E-5)


# Stage 2 skeleton
student_model.requires_grad_(True)
for name, param in student_model.named_parameters():
    if any(
        [
            n in name
            for n in [
                "mlp",
                "input_layernorm",
                "embedding",
                "final_layernorm",
                "lm_head",
            ]
        ]
    ):
        param.requires_grad_(False)
    else:
        param.requires_grad_(True)

freeze_mlp = True  # up to training scheme

mean_iteration_losses = []
token_counts = []
total_tokens = 0 

for idx, batch in enumerate(dataloader):
    input_ids = (
        tokenizer(batch["text"], return_tensors="pt", truncation=True, padding=True)
        .to(device)
        .input_ids
    )

    teacher_outputs = teacher_model(
        input_ids=input_ids,
        output_hidden_states=True,
        use_cache=False,
        output_attention_results=freeze_mlp,
    )

    # Count tokens in this batch (excluding padding tokens)
    batch_tokens = (input_ids != tokenizer.pad_token_id).sum().item()
    total_tokens += batch_tokens


    mean_layer_loss = []
    for layer_idx, student_layer in enumerate(student_model.backbone.layers):
        student_input = teacher_outputs.all_hidden_states[layer_idx].float()

        # Forward pass
        student_output = student_layer(
            hidden_states=student_input,
            run_mlp_component=not freeze_mlp,
            return_hidden_states=not freeze_mlp,
        )
        teacher_hstate = (
            teacher_outputs.all_attn_outputs[layer_idx]
            if freeze_mlp
            else teacher_outputs.all_hidden_states[layer_idx + 1]
        )

        assert student_output["hidden_states"].size() == teacher_hstate.size()

        loss = torch.norm(
            student_output["hidden_states"] - teacher_hstate, p=2, dim=(-1,)
        ).mean()

        loss.backward()
        optimizer.step()
        mean_layer_loss.append(loss.item())
        print(f"Iter {idx}, Layer {layer_idx}, Loss: {loss.item()}")

    token_counts.append(total_tokens)
    mean_iteration_losses.append(np.mean(mean_layer_loss))

# Plot the loss 
plot_loss(token_counts, mean_iteration_losses, experiment_name)

print("DONE")