import json

# --- Writing to a JSON file ---
data = {
    "name": "your-name",
    "age": 22,
    "major": "Computer Engineering",
    "languages": ["Persian", "English", "Python"]
}

# Save dictionary as JSON file
with open("data.json", "w") as f:
    json.dump(data, f, indent=4)   # indent=4 makes it pretty-printed


# --- Reading from a JSON file ---
with open("data.json", "r") as f:
    loaded_data = json.load(f)

print("Loaded data:", loaded_data)
print("Name:", loaded_data["name"])
