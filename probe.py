from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

import bldc_loader.loader as L
print([x for x in dir(L) if x.startswith("load")])
