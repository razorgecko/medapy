from typing import Any, Iterable

from pint.compat import Unpack
from pint.delegates.formatter._compound_unit_helpers import BabelKwds, SortFunc
from pint.delegates.formatter.plain import CompactFormatter
from pint.facets.plain import PlainUnit
from pint.delegates.formatter._spec_helpers import REGISTERED_FORMATTERS


class MeasurementSheetFormatter(CompactFormatter):
     def format_unit(self,
        unit: PlainUnit | Iterable[tuple[str, Any]],
        uspec: str = "",
        sort_func: SortFunc | None = None,
        **babel_kwds: Unpack[BabelKwds],
    ) -> str:
        """Format a unit (can be compound) into string
        given a string formatting specification and locale related arguments.
        """
        compact_str = super().format_unit(unit, uspec, sort_func, **babel_kwds)
        return compact_str.replace('**', '^')

REGISTERED_FORMATTERS['ms'] = MeasurementSheetFormatter()