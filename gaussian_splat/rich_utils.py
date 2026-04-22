from collections import deque

from rich.console import Console, Group
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    track,
)
from rich.table import Table


console = Console(stderr=True)


class LossTableProgress(Progress):
    def __init__(self, *columns, loss_columns=None, table_max_rows: int = 10, **kwargs) -> None:
        self.loss_columns = list(loss_columns or [])
        self.results = deque(maxlen=table_max_rows)
        self.table = Table()
        self._rebuild_table()
        super().__init__(*columns, **kwargs)

    def _rebuild_table(self):
        table = Table()
        table.add_column("Iter", justify="right")
        table.add_column("Total", justify="right")
        for column in self.loss_columns:
            table.add_column(column, justify="right")

        for row in self.results:
            cells = [row.get("iter", ""), row.get("total", "")]
            cells.extend(row.get(column, "") for column in self.loss_columns)
            table.add_row(*cells)

        self.table = table

    def update_loss_table(self, iteration: int, total: float, loss_terms: dict[str, float]):
        if not self.loss_columns:
            self.loss_columns = list(loss_terms.keys())
        row = {"iter": str(iteration), "total": f"{total:.5f}"}
        row.update({name: f"{value:.5f}" for name, value in loss_terms.items()})
        self.results.append(row)
        self._rebuild_table()

    def get_renderable(self):
        return Group(self.table, *self.get_renderables())


def create_training_progress(loss_columns=None, table_max_rows: int = 10) -> Progress:
    return LossTableProgress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[status]}", justify="right"),
        loss_columns=loss_columns,
        table_max_rows=table_max_rows,
        console=console,
        transient=False,
    )


def rich_track(sequence, description: str):
    return track(sequence, description=description, console=console)
