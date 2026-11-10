"""v8 T13: Tool optimizer -> Tool and stations: Alternate (opt.tool_alternate)."""

import pytest
import viser

from tests.helpers import confirm_all


@pytest.mark.slow
def test_alternate_is_never_worse_than_fixed_stations():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8870, verbose=False)
    try:
        app = build_app(server)
        assert app["project"].load_example("rail_long_part")
        tool = app["tool_opt"]
        assert app["session"].tool.extension.type == "bent"  # the alternation applies to L / theta / d tools
        tool.shape_evals.value = 6
        cost = {}
        for loop in ("Stations fixed after length search", "Alternate"):
            tool.loop.value = loop
            confirm_all(tool)
            tool.start(block=True)
            assert tool.result.feasible
            cost[loop] = tool.result.solution.cost
        assert cost["Alternate"] <= cost["Stations fixed after length search"] + 1e-9
        tool.loop.value = "Stations fixed after length search"
    finally:
        server.stop()
