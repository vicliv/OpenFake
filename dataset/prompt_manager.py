"""
Creates a streamer class to pass prompts to the models. Prompts come from filepath var
"""

import pandas as pd
import random

class CSVPromptStreamer:
    def __init__(self, filepath="data/prompts/unused_prompts2.csv"):
        print(f"Loading and shuffling prompts from {filepath}...")
        
        # Load the CSV
        df = pd.read_csv(filepath)
        
        # Grab the column that contains the prompt
        self.prompts = df["prompt"].dropna().tolist()
        
        # Shuffle immediately
        random.shuffle(self.prompts)
        
        self.index = 0
        self.total_prompts = len(self.prompts)
        print(f"Successfully loaded {self.total_prompts} prompts into memory.")

    def get_next_prompt(self):
        # Reshuffle if we hit the end of the list
        if self.index >= self.total_prompts:
            random.shuffle(self.prompts)
            self.index = 0
            
        prompt = self.prompts[self.index]
        self.index += 1
        
        return str(prompt)