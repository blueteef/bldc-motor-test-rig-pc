from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd


@dataclass
class Run:
    run_id: int
    csv_path: Path
    json_path: Optional[Path]
    df: pd.DataFrame
    meta: Dict[str, Any] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"RUN{self.run_id:04d}"


@dataclass
class RunSet:
    folder: Path
    runs: List[Run] = field(default_factory=list)
    summary: Optional[pd.DataFrame] = None
    summary_short: Optional[pd.DataFrame] = None
    issues: List[str] = field(default_factory=list)

    def get(self, run_id: int) -> Optional[Run]:
        for r in self.runs:
            if r.run_id == run_id:
                return r
        return None
