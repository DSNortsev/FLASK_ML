"""Utilities Library"""
import json


def read_json(file):
    """Read JSON file"""
    with open(file, 'r') as f:
        return json.loads(f.read())


def save_json(data, file):
    """Save python dict to JSON file"""
    with open(file, 'w') as f:
        return json.dump(data, f, indent=4)
