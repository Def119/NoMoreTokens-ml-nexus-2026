"""Execute the generated audit notebook with this exact Python interpreter."""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
import nbformat
from nbclient import NotebookClient

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('notebook');args=p.parse_args()
    path=Path(args.notebook).resolve()
    root=Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix='nexus-kernel-') as temp:
        kernels=Path(temp)/'kernels'/'nexus-project';kernels.mkdir(parents=True)
        (kernels/'kernel.json').write_text(json.dumps(dict(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
                                                          display_name='Nexus project Python',language='python')),encoding='utf-8')
        os.environ['JUPYTER_PATH']=temp+os.pathsep+os.environ.get('JUPYTER_PATH','')
        notebook=nbformat.read(path,as_version=4)
        notebook=NotebookClient(notebook,timeout=180,kernel_name='nexus-project',resources={'metadata':{'path':str(root)}}).execute()
        nbformat.write(notebook,path)
        print(f'Executed {sum(c.cell_type=="code" for c in notebook.cells)} cells: {path}')
