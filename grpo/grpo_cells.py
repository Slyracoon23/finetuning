# %% [markdown]
# # Fine-tune a model with GRPO (cell notebook style)
# This script follows the Hugging Face GRPO fine-tuning exercise, organized as Python cells.

# %% [markdown]
# ## 1. Install dependencies (run in terminal if not already installed)
# !pip install -qqq datasets==3.2.0 transformers==4.47.1 trl==0.14.0 peft==0.14.0 accelerate==1.2.1 bitsandbytes==0.45.2 wandb==0.19.7 --progress-bar off
# !pip install -qqq flash-attn --no-build-isolation --progress-bar off

# %%
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
import wandb

# %% [markdown]
# ## 2. Log in to Weights & Biases (optional)
wandb.login()

# %% [markdown]
# ## 3. Load the dataset

dataset = load_dataset("mlabonne/smoltldr")
print(dataset)

# %% [markdown]
# ## 4. Load model

model_id = "HuggingFaceTB/SmolLM-135M-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# %% [markdown]
# ## 5. Load LoRA

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
)
model = get_peft_model(model, lora_config)
print(model.print_trainable_parameters())

# %% [markdown]
# ## 6. Define the reward function

ideal_length = 50

def reward_len(completions, **kwargs):
    return [-abs(ideal_length - len(completion)) for completion in completions]

# %% [markdown]
# ## 7. Define the training arguments

training_args = GRPOConfig(
    output_dir="GRPO",
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

wandb.init(project="GRPO")
trainer.train()

# %% [markdown]
# ## 9. Save and publish the model

# merged_model = trainer.model.merge_and_unload()
# merged_model.push_to_hub(
#     "SmolGRPO-135M", private=False, tags=["GRPO", "Reasoning-Course"]
# )

# %% [markdown]
# ## 10. Generate text

from transformers import pipeline

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

generator = pipeline("text-generation", model="SmolGRPO-135M")
# Or use the model and tokenizer we defined earlier
# generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

generate_kwargs = {
    "max_new_tokens": 256,
    "do_sample": True,
    "temperature": 0.5,
    "min_p": 0.1,
}

generated_text = generator(messages, generate_kwargs=generate_kwargs)
print(generated_text) 