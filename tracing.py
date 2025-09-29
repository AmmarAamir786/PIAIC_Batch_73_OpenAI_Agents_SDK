from agents import TracingProcessor




class CustomProcessor(TracingProcessor):
    def __init__(self):
        self.active_traces = {}
        self.active_spans = {}

    def on_trace_start(self, trace):
        print(f"[TRACING] Trace started: {trace.trace_id}")
        print(f"[TRACING] Trace name: {getattr(trace, 'name', 'Unknown')}")
        self.active_traces[trace.trace_id] = trace

    def on_trace_end(self, trace):
        print(f"[TRACING] Trace ended: {trace.trace_id}")
        print(f"[TRACING] Total active traces: {len(self.active_traces)}")
        # Process completed trace
        del self.active_traces[trace.trace_id]

    def on_span_start(self, span):
        print(f"[TRACING] Span started: {span.span_id}")
        print(f"[TRACING] Span name: {getattr(span, 'name', 'Unknown')}")
        print(f"[TRACING] Span trace_id: {getattr(span, 'trace_id', 'Unknown')}")
        self.active_spans[span.span_id] = span

    def on_span_end(self, span):
        print(f"[TRACING] Span ended: {span.span_id}")
        print(f"[TRACING] Span duration: {getattr(span, 'duration', 'Unknown')}")
        print(f"[TRACING] Total active spans: {len(self.active_spans)}")
        # Process completed span
        del self.active_spans[span.span_id]

    def shutdown(self):
        print("[TRACING] CustomProcessor shutting down...")
        print(f"[TRACING] Cleaning up {len(self.active_traces)} traces and {len(self.active_spans)} spans")
        # Clean up resources
        self.active_traces.clear()
        self.active_spans.clear()

    def force_flush(self):
        print("[TRACING] Force flushing CustomProcessor...")
        # Force processing of any queued items
        pass