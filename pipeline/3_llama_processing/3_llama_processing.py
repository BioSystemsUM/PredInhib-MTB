import time
import pickle
import pandas as pd
import numpy as np
import os
import random
import torch
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import Union

import os

# --- Config ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
data_path = os.path.join(REPO_ROOT, "data")

descriptions_path = os.path.join(data_path, "llama", "unique_assay_descriptions.csv")
prompt_template_path = os.path.join(data_path, "llama", "prompt_template.txt")
save_descriptions_path = os.path.join(data_path, "llama", "processed_assay_descriptions.pkl")
save_description_csv_path = os.path.join(data_path, "llama", "processed_assay_descriptions.csv")


# --- Seeds ---
seed = 1234
random.seed(seed)
torch.manual_seed(seed)

# --- Load Data ---
descriptions_df = pd.read_csv(descriptions_path)
descriptions = descriptions_df["assay_description"].tolist()

# --- LLM Client (Single model using all GPUs) ---
model = OllamaLLM(model="llama3.3", base_url="http://localhost:11434")

# --- Schema ---
class Answers(BaseModel):
    mtb_strain: Union[str, bool] = Field(description="strain of Mycobacterium tuberculosis")
    mentions_resistance: Union[str, bool] = Field(description="level of drug resistance (R, mdr, xdr, mdr/xdr, or false)")
    resistant_to: Union[str, bool] = Field(description="drugs to which there is resistance.")
    mutant: Union[str, bool] = Field(description="if the strain is a mutant.")
    mutant_type: Union[str, bool] = Field(description="type of mutation.")
    checkerboard: Union[str, bool] = Field(description="checkerboard assay.")

# --- Prompt Setup ---
parser = JsonOutputParser(pydantic_object=Answers)
with open(prompt_template_path, "r", encoding="utf-8") as f:
    response_template = f.read()

prompt = PromptTemplate(
    template=response_template,
    input_variables=["assay_description"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# --- Chain ---
chain = prompt | model | parser

# --- Load Progress ---
if os.path.exists(save_descriptions_path):
    with open(save_descriptions_path, "rb") as f:
        processed_descriptions = pickle.load(f)
else:
    processed_descriptions = {}

remaining_descriptions = sorted(set(descriptions) - set(processed_descriptions.keys()))


if len(remaining_descriptions) == 0:
    print("All descriptions have already been processed.")
    import sys
    sys.exit()
    
# --- Processing Loop ---
for index, description in enumerate(remaining_descriptions):
    if description in processed_descriptions:
        continue

    print(f"\nProcessing {index + 1}/{len(remaining_descriptions)}")
    start_time = time.time()

    try:
        result = chain.invoke(description)
    except Exception as e:
        print(f"  ❌ Error: {e}")
        result = {key: 'error' for key in Answers.model_fields.keys()}

    elapsed = time.time() - start_time
    print(f"⏱️ Time for description: {elapsed:.2f}s")

    processed_descriptions[description] = result

    if index % 100 == 0:
        with open(save_descriptions_path, "wb") as f:
            pickle.dump(processed_descriptions, f)

print("\n✅ All done.")

# Save final state
with open(save_descriptions_path, "wb") as f:
    pickle.dump(processed_descriptions, f)

# --- Format and Export ---
def get_output_dict(description):
    if pd.isna(description):
        return {key: np.nan for key in Answers.model_fields.keys()}
    else:
        return processed_descriptions.get(description, {key: np.nan for key in Answers.model_fields.keys()})

output_df = descriptions_df["assay_description"].apply(get_output_dict).apply(pd.Series)
descriptions_df[list(Answers.model_fields.keys())] = output_df
descriptions_df.to_csv(save_description_csv_path, index=False, sep=",")
