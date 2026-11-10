import os
import platform
from pathlib import Path

directory = Path(os.environ["GENESIS_RUN_DIRECTORY"])
result = directory / "execution-proof.txt"
result.write_text("Remote execution on " + platform.system() + "\n", encoding="ascii")
print(result.read_text(), end="", flush=True)
