# %% [markdown]
# # Fine-tune a model with GRPO using Unsloth (cell notebook style)
# This script adapts the Hugging Face GRPO fine-tuning exercise to use Unsloth for faster training.

# %% [markdown]
# ## 1. Install dependencies (run in terminal if not already installed)
# !pip install -qqq datasets==3.2.0 transformers==4.47.1 trl==0.14.0 accelerate==1.2.1 bitsandbytes==0.45.2 wandb==0.19.7 --progress-bar off
# !pip install -qqq unsloth --progress-bar off
# !pip install -qqq flash-attn --no-build-isolation --progress-bar off

# %%
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
import wandb
from unsloth import FastLanguageModel

# %% [markdown]
# ## 2. Log in to Weights & Biases (optional)
wandb.login()

# %% [markdown]
# ## 3. Load the dataset

dataset = load_dataset("mlabonne/smoltldr")
print(dataset)

# %% [markdown]
# ## 4. Load model with Unsloth

model_id = "HuggingFaceTB/SmolLM-135M-Instruct"

# Using Unsloth's FastLanguageModel instead of AutoModelForCausalLM
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_id,
    max_seq_length=2048,
    dtype=torch.bfloat16,
    load_in_4bit=True,
    device_map="auto",
)

# %% [markdown]
# ## 5. Apply LoRA using Unsloth's optimized method

# Unsloth has its own LoRA implementation which is more efficient
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=42,
)

# print(f"Trainable parameters: {(model.num_parameters(train_only=True) / 1e6):.2f}M")
# print(f"Total parameters: {(model.num_parameters() / 1e6):.2f}M")

# %% [markdown]
# ## 6. Define the reward function

ideal_length = 50

def reward_len(completions, **kwargs):
    return [-abs(ideal_length - len(completion)) for completion in completions]

# %% [markdown]
# ## 7. Define the training arguments

training_args = GRPOConfig(
    output_dir="GRPO_Unsloth",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    max_prompt_length=512,
    max_completion_length=96,
    num_generations=8,
    optim="adamw_8bit",
    num_train_epochs=1,
    bf16=True,
    report_to=["wandb"],
    remove_unused_columns=False,
    logging_steps=1,
)

# %% [markdown]
# ## 8. Initialize trainer and train

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_len],
    args=training_args,
    train_dataset=dataset["train"],
)

wandb.init(project="GRPO_Unsloth")
trainer.train()

# %% [markdown]
# ## 9. Save and publish the model

# Save the fine-tuned model
model.save_pretrained("./SmolGRPO-135M-Unsloth")
tokenizer.save_pretrained("./SmolGRPO-135M-Unsloth")

# Optionally push to Hugging Face
# from huggingface_hub import HfApi
# api = HfApi()
# api.upload_folder(
#     folder_path="./SmolGRPO-135M-Unsloth",
#     repo_id="YourUsername/SmolGRPO-135M-Unsloth",
#     repo_type="model",
# )

# %% [markdown]
# ## 10. Generate text

from transformers import pipeline
from unsloth import FastLanguageModel

prompt = """
# A long document about the Cat

The cat (Felis catus), also referred to as the domestic cat or house cat, is a small 
domesticated carnivorous mammal. It is the only domesticated species of the family Felidae.
Advances in archaeology and genetics have shown that the domestication of the cat occurred
in the Near East around 7500 BC. It is commonly kept as a pet and farm cat, but also ranges
freely as a feral cat avoiding human contact. It is valued by humans for companionship and
its ability to kill vermin. Its retractable claws are adapted to killing small prey species
such as mice and rats. It has a strong, flexible body, quick reflexes, and sharp teeth,
and its night vision and sense of smell are well developed. It is a social species,
but a solitary hunter and a crepuscular predator. Cat communication includes
vocalizations—including meowing, purring, trilling, hissing, growling, and grunting—as
well as body language. It can hear sounds too faint or too high in frequency for human ears,
such as those made by small mammals. It secretes and perceives pheromones.
"""

messages = [
    {"role": "user", "content": prompt},
]

# Load the saved model and tokenizer for inference using Unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./SmolGRPO-135M-Unsloth", # Load from the saved directory
    max_seq_length=2048, # Use the same max_seq_length as training
    dtype=torch.bfloat16, # Use the same dtype as training
    load_in_4bit=True, # Load in 4bit for efficiency
    device_map="auto",
)

# Prepare model for faster inference
FastLanguageModel.for_inference(model)

generate_kwargs = {
    "max_new_tokens": 256,
    "do_sample": True,
    "temperature": 0.5,
    "min_p": 0.1,
}

# Encode the input messages
# The apply_chat_template is needed if you used a chat template during training
# If not, you can use tokenizer(prompt, return_tensors="pt").to("cuda")
input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
input_ids = input_ids.to("cuda")

# Generate text directly using the model's generate method
outputs = model.generate(input_ids, **generate_kwargs)

# Decode the generated output
generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

for text in generated_text:
    print(text)
# %%
