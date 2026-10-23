"""Execute the focused notebook in a fresh kernel and retain outputs on failure."""
import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('demo', 'smoke'), default='demo')
    parser.add_argument('--output', type=Path, required=True, help='New executed .ipynb path')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a fresh path')
    root = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(root / 'focused_comparison.ipynb', as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    os.environ['COMPARISON_MODE'] = args.mode
    args.output.parent.mkdir(parents=True, exist_ok=True)
    client = NotebookClient(notebook, timeout=3600, kernel_name='python3',
                            resources={'metadata': {'path': str(root)}})
    try:
        client.execute()
    finally:
        nbformat.write(notebook, args.output)
    print(f'Executed notebook: {args.output.resolve()}')


if __name__ == '__main__':
    main()
