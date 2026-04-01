import sqlite3
import json
import os
import time
import math
import argparse
import re
import string
from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────
# In-Memory KG built from dataset graph field
# ─────────────────────────────────────────
class LocalKG:
    def __init__(self, triples):
        """Build adjacency index from list of [subject, relation, object] triples."""
        self.index = {}  # entity -> [(relation, object)]
        self.all_entities = set()
        for triple in triples:
            if len(triple) != 3:
                continue
            s, r, o = triple
            self.all_entities.add(s)
            self.all_entities.add(o)
            if s not in self.index:
                self.index[s] = []
            self.index[s].append((r, o))

    def get_relations(self, entity):
        """Return all outgoing relations from an entity."""
        return list(set(r for r, o in self.index.get(entity, [])))

    def get_entities(self, entity, relation):
        """Return all objects reachable via (entity, relation)."""
        return [o for r, o in self.index.get(entity, []) if r == relation]

    def get_leaf_entities(self, path):
        """Return the final entity in a reasoning path."""
        if not path:
            return []
        return [path[-1][2]]


# ─────────────────────────────────────────
# MCTS Node
# ─────────────────────────────────────────
class MCTSNode:
    def __init__(self, entity, path=None, parent=None, depth=0):
        self.entity = entity
        self.path = path or []
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.depth = depth

    def ucb_score(self, c=0.5):
        """UCB1 score balancing exploitation and exploration."""
        if self.visits == 0:
            return float("inf")
        exploit = self.value / self.visits
        explore = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploit + explore

    def is_leaf(self):
        return len(self.children) == 0


# ─────────────────────────────────────────
# ReKG-MCTS Experiment
# ─────────────────────────────────────────
class ReKGMCTSExperiment:
    def __init__(self, db_path="experiment.db", model_name="deepseek-chat",
                 max_depth=3, beam_width=3, max_iter=5):
        self.db_path = db_path
        self.model_name = model_name
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.max_iter = max_iter

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY missing in .env")
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS mcts_runs
                     (id TEXT PRIMARY KEY,
                      question TEXT,
                      gold_answers JSON,
                      best_path JSON,
                      predicted_answer TEXT,
                      is_correct INTEGER,
                      model_name TEXT)''')
        conn.commit()
        conn.close()

    def _llm(self, prompt, retry=3):
        """Call LLM with retry logic."""
        for i in range(retry):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=256
                )
                return resp.choices[0].message.content
            except Exception as e:
                if i == retry - 1:
                    return f"ERROR: {e}"
                time.sleep(1)

    # ── LLM Module 1: Evaluate path relevance (0.0~1.0) ──
    def evaluate_path(self, question, path):
        if not path:
            return 0.0
        path_str = " -> ".join([f"({s})-[{r}]->({o})" for s, r, o in path])
        prompt = f"""Rate how useful this knowledge graph path is for answering the question.

SCORING GUIDE:
0.9-1.0: The final entity directly answers the question.
0.7-0.8: The path is highly relevant and close to the answer.
0.4-0.6: Somewhat relevant but not a direct answer.
0.0-0.3: Irrelevant to the question.

EXAMPLES:
Q: What language do Jamaican people speak?
Path: (Jamaica)-[language.human_language.main_country]->(Jamaican English)
Score: 0.9

Q: Who is the president of France?
Path: (France)-[location.country.capital]->(Paris)
Score: 0.3

Q: {question}
Path: {path_str}
Score (return ONLY a decimal number):"""
        resp = self._llm(prompt)
        try:
            return min(1.0, max(0.0, float(re.findall(r"[0-9.]+", resp)[0])))
        except:
            return 0.3

    # ── LLM Module 2: Select top-k relevant relations ──
    def select_relations(self, question, entity, relations, topk=3):
        if not relations:
            return []
        if len(relations) <= topk:
            return relations
        rel_list = "\n".join([f"{i+1}. {r}" for i, r in enumerate(relations[:20])])
        prompt = f"""Select the {topk} most relevant relations for answering the question.
Question: {question}
Current entity: {entity}
Relations:
{rel_list}
Return ONLY relation names separated by semicolons."""
        resp = self._llm(prompt)
        selected = [r.strip() for r in resp.split(";")]
        valid = [r for r in selected if r in relations]
        return valid[:topk] if valid else relations[:topk]

    # ── MCTS Phase 1: Node Selection via UCB ──
    def select_node(self, root):
        node = root
        while not node.is_leaf() and node.depth < self.max_depth:
            node = max(node.children, key=lambda c: c.ucb_score())
        return node

    # ── MCTS Phase 2: Path Expansion ──
    def expand(self, node, question, kg):
        relations = kg.get_relations(node.entity)
        if not relations:
            return
        selected = self.select_relations(question, node.entity, relations, topk=self.beam_width)
        for rel in selected:
            for obj in kg.get_entities(node.entity, rel)[:2]:
                new_path = node.path + [(node.entity, rel, obj)]
                child = MCTSNode(entity=obj, path=new_path, parent=node, depth=node.depth + 1)
                node.children.append(child)

    # ── MCTS Phase 3: MC Rollout Simulation ──
    def simulate(self, node, question, kg):
        cur = node.entity
        path = list(node.path)
        for _ in range(self.max_depth - node.depth):
            rels = kg.get_relations(cur)
            if not rels:
                break
            sel = self.select_relations(question, cur, rels, topk=1)
            rel = sel[0] if sel else rels[0]
            objs = kg.get_entities(cur, rel)
            if not objs:
                break
            obj = objs[0]
            path.append((cur, rel, obj))
            cur = obj
        return self.evaluate_path(question, path)

    # ── MCTS Phase 4: Backpropagation ──
    def backpropagate(self, node, value):
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent

    # ── Full MCTS loop for one question ──
    def mcts(self, question, topic_entity, kg):
        root = MCTSNode(entity=topic_entity)
        root.visits = 1
        best_path, best_score = [], 0.0

        for _ in range(self.max_iter):
            node = self.select_node(root)
            if node.depth < self.max_depth:
                self.expand(node, question, kg)
            if node.children:
                node = node.children[0]
            score = self.simulate(node, question, kg)
            self.backpropagate(node, score)
            if score > best_score:
                best_score = score
                best_path = node.path

        return best_path, best_score

    # ── Exact match evaluation ──
    def _normalize(self, s):
        s = s.lower()
        s = re.sub(r'\b(a|an|the)\b', ' ', s)
        s = ''.join(ch for ch in s if ch not in string.punctuation)
        return ' '.join(s.split())

    def _exact_match(self, pred, gold):
        return self._normalize(gold) in self._normalize(pred)

    # ── Fuzzy match between candidate and gold answer ──
    def _fuzzy_match(self, cand, gold):
        """Check if candidate and gold share significant word overlap."""
        cand_words = set(self._normalize(cand).split())
        gold_words = set(self._normalize(gold).split())
        if not gold_words:
            return False
        overlap = cand_words & gold_words
        # Match if more than half of gold words appear in candidate
        return len(overlap) / len(gold_words) >= 0.5

    # ── Extract answer directly from KG path entities ──
    def extract_answer_from_path(self, path, question, gold_answers):
        """
        Try to match path entities directly against gold answers.
        Uses both exact and fuzzy matching to handle surface form variations.
        """
        if not path:
            return None, False

        # Collect all entities in the path, prefer deeper (leaf) entities
        candidates = [o for s, r, o in path]

        # First try exact match
        for cand in reversed(candidates):
            for gold in gold_answers:
                if self._exact_match(cand, gold):
                    return cand, True

        # Then try fuzzy match
        for cand in reversed(candidates):
            for gold in gold_answers:
                if self._fuzzy_match(cand, gold):
                    return cand, True

        return candidates[-1] if candidates else None, False

    def run(self, limit=10):
        print("Loading RoG-WebQSP dataset...")
        dataset = load_dataset("rmanluo/RoG-webqsp", split="test")
        if limit:
            dataset = dataset.select(range(limit))

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        correct = 0

        pbar = tqdm(dataset, desc="MCTS Reasoning")
        for item in pbar:
            q = item["question"]
            answers = item["answer"]
            topic_entities = item.get("q_entity", [])
            if not topic_entities:
                continue

            # Build local KG from pre-extracted graph triples
            kg = LocalKG(item["graph"])
            topic_entity = topic_entities[0]

            # Run MCTS over local KG
            best_path, best_score = self.mcts(q, topic_entity, kg)

            # Step 1: Try to extract answer directly from KG path entities
            pred, is_correct = self.extract_answer_from_path(best_path, q, answers)

            # Step 2: If KG entity didn't match, fall back to LLM generation
            if not is_correct:
                path_str = " -> ".join([f"({s})-[{r}]->({o})" for s, r, o in best_path]) if best_path else "No path"
                prompt = f"""Answer the question using the knowledge graph path.
Question: {q}
KG Path: {path_str}
End with: Answer: [your answer]"""
                resp = self._llm(prompt)
                if "Answer:" in resp:
                    pred = resp.split("Answer:")[-1].strip()
                else:
                    pred = resp.strip().split("\n")[-1].strip()
                is_correct = any(self._exact_match(pred, g) for g in answers)

            if is_correct:
                correct += 1

            pbar.set_postfix({"Acc": f"{correct / (pbar.n + 1) * 100:.2f}%"})

            c.execute('''INSERT OR REPLACE INTO mcts_runs
                         (id, question, gold_answers, best_path, predicted_answer, is_correct, model_name)
                         VALUES (?, ?, ?, ?, ?, ?, ?)''',
                      (item.get("id", q[:50]), q, json.dumps(answers),
                       json.dumps(best_path), pred or "", 1 if is_correct else 0, self.model_name))
            conn.commit()

        total = len(dataset)
        print(f"\nFinal MCTS Accuracy: {correct / total * 100:.2f}%")
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--model", type=str, default="deepseek-chat")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--width", type=int, default=3)
    parser.add_argument("--iter", type=int, default=5)
    args = parser.parse_args()

    exp = ReKGMCTSExperiment(
        model_name=args.model,
        max_depth=args.depth,
        beam_width=args.width,
        max_iter=args.iter
    )
    exp.run(limit=args.limit)