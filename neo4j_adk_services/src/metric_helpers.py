from dataclasses import dataclass
from time import time
from uuid import uuid4

@dataclass
class Metric:
    name: str
    value: float
    unit: str | None = None
    ts: float = time()
    id: str = uuid4().hex

# Example usage (not part of the service, just for illustration):
# metric_data = Metric(name="token_count", value=150.0, unit="tokens")
# cypher_query_for_metric = "CREATE (m:Metric $metric_props)-[:FOR_EVENT]->(e)"
# params_for_metric = {"metric_props": metric_data.__dict__} # Or use asdict(metric_data)