import sys
import nbformat

def clear_outputs(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    dirty = False
    for cell in nb.cells:
        if cell.cell_type == 'code' and cell.outputs:
            cell.outputs = []
            dirty = True

    if dirty:
        with open(file_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)
    for path in sys.argv[1:]:
        clear_outputs(path)
