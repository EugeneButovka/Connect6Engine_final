import sys
import nbformat
from nbclient import NotebookClient

nb = nbformat.read(sys.argv[1], as_version=4)
client = NotebookClient(nb, timeout=3600, kernel_name="python3")
client.execute()
nbformat.write(nb, sys.argv[1])
print("executed + stored outputs:", sys.argv[1])
