import random

import msgspec


def rand(n: int, r: random.Random):
    return int(r.random() * n)


def transform(data, num_sample: int, r: random.Random):
    json_decode = msgspec.json.decode
    try:
        data = json_decode(data["clean_content"])
        return {"input": "", "output": data["content"]}
    except Exception as e:
        print(f"error: {e}")
        return None
