from endee_client import resolve_endee_client

client, _, _ = resolve_endee_client()
index = client.get_index(name="dbpedia_10k_benchmark")  # or your ENDEE_INDEX_NAME

# After get_index, metadata is already on the object:
print("M:", index.M, "ef_con:", index.ef_con)

# Or full dict (includes M and ef_con under stable keys):
print(index.describe())