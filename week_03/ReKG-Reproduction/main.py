import sqlite3
import json
import os
import time
import argparse
import re
import string
from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI
from dotenv import load_dotenv

# Load env vars
load_dotenv()

class DeepSeekBaselineExperiment:
    def __init__(self, db_path="experiment.db", model_name="deepseek-chat"):
        self.db_path = db_path
        self.model_name = model_name
        
        # Setup DeepSeek client (OpenAI-compatible)
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY missing in .env")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        self._init_db()

    def _init_db(self):
        """Setup SQLite tables."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        # Table 1: Raw Data
        c.execute('''CREATE TABLE IF NOT EXISTS webqsp_data
                     (id TEXT PRIMARY KEY, question TEXT, answers JSON)''')
        # Table 2: Results
        c.execute('''CREATE TABLE IF NOT EXISTS baseline_runs
                     (id TEXT PRIMARY KEY, 
                      question TEXT,
                      gold_answers JSON,
                      model_reasoning TEXT, 
                      predicted_answer TEXT, 
                      is_correct INTEGER,
                      model_name TEXT)''')
        conn.commit()
        conn.close()

    def load_dataset(self, limit=None):
        """Fetch dataset from HF and store in SQLite."""
        print(f"Loading dataset (Limit: {limit})...")
        dataset = load_dataset("stanfordnlp/web_questions", split="test")
        
        if limit:
            dataset = dataset.select(range(limit))
            
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        for item in tqdm(dataset, desc="Ingesting"):
            c.execute("INSERT OR IGNORE INTO webqsp_data (id, question, answers) VALUES (?, ?, ?)", 
                      (item['url'], item['question'], json.dumps(item['answers'])))
        
        conn.commit()
        conn.close()

    def _normalize(self, s):
        """Standardize answer text for eval."""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        def white_space_fix(text):
            return ' '.join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        return white_space_fix(remove_articles(remove_punc(s.lower())))

    def _exact_match(self, pred, gold):
        return self._normalize(gold) in self._normalize(pred)

    def run_inference(self, limit=None, retry=3):
        """Main CoT inference loop."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        # Fetch data with limit
        query = "SELECT id, question, answers FROM webqsp_data"
        if limit:
            query += f" LIMIT {limit}"
        
        rows = c.execute(query).fetchall()
        correct_cnt = 0
        
        print(f"Starting inference with {self.model_name}...")
        pbar = tqdm(rows, desc="Reasoning")
        
        for row in pbar:
            gold_answers = json.loads(row['answers'])
            
            # CoT Prompt
            prompt = f"""
            Question: {row['question']}
            Answer the question concisely.
            First, think step-by-step.
            Finally, output the answer strictly in this format:
            Answer: [Short Answer]
            """
            
            resp_text = ""
            for attempt in range(retry):
                try:
                    resp = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        stream=False
                    )
                    resp_text = resp.choices[0].message.content
                    break
                except Exception as e:
                    if attempt == retry - 1: resp_text = f"ERROR: {e}"
                    time.sleep(1)

            # Parse Answer
            pred = resp_text.split("Answer:")[-1].strip() if "Answer:" in resp_text else resp_text.split("\n")[-1].strip()
            
            # Evaluate
            is_correct = any(self._exact_match(pred, gold) for gold in gold_answers)
            if is_correct: correct_cnt += 1
            
            # Update Progress
            if pbar.n > 0:
                pbar.set_postfix({"Acc": f"{(correct_cnt / (pbar.n + 1)) * 100:.2f}%"})

            # Save Results
            c.execute('''INSERT OR REPLACE INTO baseline_runs 
                         (id, question, gold_answers, model_reasoning, predicted_answer, is_correct, model_name)
                         VALUES (?, ?, ?, ?, ?, ?, ?)''',
                      (row['id'], row['question'], row['answers'], resp_text, pred, 1 if is_correct else 0, self.model_name))
            conn.commit()
            
        if len(rows) > 0:
            print(f"\nFinal Accuracy: {(correct_cnt / len(rows)) * 100:.2f}%")
        conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10, help="Sample size")
    parser.add_argument("--model", type=str, default="deepseek-chat", help="Model name")
    args = parser.parse_args()
    
    # Init and Run
    exp = DeepSeekBaselineExperiment(model_name=args.model)
    exp.load_dataset(limit=args.limit)
    exp.run_inference(limit=args.limit)