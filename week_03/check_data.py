from datasets import load_dataset
ds = load_dataset("rmanluo/RoG-webqsp", split="test")
item = ds[0]
print("Keys:", item.keys())
print("Question:", item["question"])
print("Answer:", item["answer"])
print("Q_entity:", item["q_entity"])
print("Graph size:", len(item["graph"]))
print("Graph sample:", item["graph"][:3])