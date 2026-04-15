from datasets import load_dataset

# Stream the dataset to see what's inside without downloading 1M rows
ds = load_dataset("Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M", streaming=True)
sample = next(iter(ds['train']))

print(f"Title: {sample['title']}")
print(f"Text Snippet: {sample['text']}")