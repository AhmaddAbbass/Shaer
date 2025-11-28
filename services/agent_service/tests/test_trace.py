from services.agent_service.app.agents import TraceRecorder


def test_trace_recorder_increments_steps():
    trace = TraceRecorder()
    trace.add("Orchestrator", "parse_intent", "test")
    trace.add("Poetry", "build_spec", "spec")
    assert trace.steps[0].step == 1
    assert trace.steps[1].step == 2
    assert trace.current_step == 3
