import os
from pathlib import Path
p = Path('/data/ephemeral/home/jeonga/level2-objectdetection-cv-18/Co-DETR/work_dirs/test')
print(os.path.join(p.parent, f'_result.csv'))
# print(x for x in p.iterdir() if x.is_dir())
print(p.parent.resolve())
print(os.getcwd())
print(p.parent.joinpath('Co-DETR/work_dirs/test'))
# print(os.pardir())