import json
import os

def py_to_ipynb(py_filename, ipynb_filename):
    with open(py_filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cells = []
    current_cell = []
    current_type = 'code'

    def flush_cell():
        if not current_cell:
            return
        
        source = current_cell.copy()
        
        if current_type == 'markdown':
            # Remove the '# ' or '#' from markdown lines
            cleaned_source = []
            for line in source:
                if line.startswith('# '):
                    cleaned_source.append(line[2:])
                elif line.startswith('#'):
                    cleaned_source.append(line[1:])
                else:
                    cleaned_source.append(line)
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": cleaned_source
            })
        else:
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source
            })
        current_cell.clear()

    for line in lines:
        if line.startswith('# %%'):
            flush_cell()
            if 'markdown' in line.lower():
                current_type = 'markdown'
            else:
                current_type = 'code'
        else:
            current_cell.append(line)

    flush_cell()

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open(ipynb_filename, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)

    print(f"Created {ipynb_filename} from {py_filename}")

if __name__ == "__main__":
    py_to_ipynb("pointer_chasing_baselines.py", "pointer_chasing_baselines.ipynb")
