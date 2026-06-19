from datasets import load_dataset

ds = load_dataset("MizzenAI/HPDv3")["train"]

models = set()

for ex in ds:
    models.add(ex["model1"])
    models.add(ex["model2"])

print(sorted(models))