"""Resource limits and information boundaries, beyond happy-path schedules."""

from concurrent.futures import ThreadPoolExecutor
import copy
import inspect
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import pytest

from swarm_scheduler_lab import backends, search
from swarm_scheduler_lab.backends import Generation, OpenAICompatibleBackend, make_prompt, parse_candidate
from swarm_scheduler_lab.budget import BudgetExceeded, BudgetLedger
from swarm_scheduler_lab.instances import make_datasets
from swarm_scheduler_lab.policy import Policy, initial_policies
from swarm_scheduler_lab.search import exchange_ring, run_search


@pytest.fixture
def train():
    return make_datasets(12, train_count=2, test_count=2, jobs=3, machines=3)["train"]


def test_concurrent_calls_and_evaluations_cannot_oversubscribe():
    ledger = BudgetLedger(max_calls=12, max_tokens=1200, max_evaluations=15)

    def attempt(_):
        try:
            call = ledger.reserve_call(50, 50)
            ledger.finish_call(call, input_tokens=None, output_tokens=None, failed=True)
            return True
        except BudgetExceeded:
            return False

    with ThreadPoolExecutor(max_workers=16) as executor:
        assert sum(executor.map(attempt, range(100))) == 12
    snapshot = ledger.snapshot()
    assert snapshot["calls"] == snapshot["failures"] == snapshot["unknown_usage_calls"] == 12
    assert snapshot["reserved_or_charged_tokens"] == 1200
    assert snapshot["input_tokens"] is snapshot["output_tokens"] is None

    def evaluation(_):
        try:
            return ledger.reserve_evaluation()
        except BudgetExceeded:
            return None

    with ThreadPoolExecutor(max_workers=16) as executor:
        indices = [i for i in executor.map(evaluation, range(100)) if i is not None]
    assert sorted(indices) == list(range(1, 16))


def test_failed_requests_keep_unknown_token_reservations_and_settle_once():
    ledger = BudgetLedger(4, 200, 1)
    first = ledger.reserve_call(70, 30)
    ledger.finish_call(first, input_tokens=None, output_tokens=None, failed=True)
    second = ledger.reserve_call(70, 30)
    with pytest.raises(BudgetExceeded):
        ledger.reserve_call(1, 1)
    ledger.finish_call(second, input_tokens=10, output_tokens=5)
    assert ledger.snapshot()["reserved_or_charged_tokens"] == 115
    with pytest.raises(ValueError):
        ledger.finish_call(second, input_tokens=10, output_tokens=5)
    assert ledger.snapshot()["calls"] == 2


def test_token_limit_blocks_before_a_call_is_charged():
    ledger = BudgetLedger(12, 100, 1)
    with pytest.raises(BudgetExceeded):
        ledger.reserve_call(60, 41)
    assert ledger.snapshot()["calls"] == 0


def test_explicit_retries_and_failures_share_island_and_global_caps(train, monkeypatch):
    class FailingBackend:
        def __init__(self, seed):
            pass

        def generate(self, *args):
            raise RuntimeError("simulated provider outage")

    monkeypatch.setattr(search, "MockBackend", FailingBackend)
    result = run_search("cooperative", train, seed=3, max_calls=6, max_retries=9,
                        max_evaluations=9, max_tokens=24000)
    resources = result["resources"]
    assert resources["calls"] == resources["failures"] == 6
    assert resources["evaluations"] == 3  # only the shared initial candidates
    assert resources["unknown_usage_calls"] == 6
    assert [sum(a["island"] == i for a in result["attempts"]) for i in range(3)] == [2, 2, 2]
    assert all(a["round"] == 1 for a in result["attempts"])
    assert not result["exchanges"]  # no invented second round after budgets ran out


def test_retry_success_costs_an_extra_call_and_no_hidden_repair(train, monkeypatch):
    class OnceFailingBackend:
        def __init__(self, seed):
            self.n = 0

        def generate(self, prompt, history, limit):
            self.n += 1
            if self.n == 1:
                raise RuntimeError("once")
            return Generation(json.dumps(initial_policies(0)[0].to_dict()), 100, 50, "estimated_mock")

    monkeypatch.setattr(search, "MockBackend", OnceFailingBackend)
    result = run_search("single", train, seed=3, max_calls=2, max_retries=1, max_evaluations=5)
    assert result["resources"]["calls"] == 2
    assert result["resources"]["failures"] == 1
    assert result["resources"]["evaluations"] == 4
    assert [attempt["retry"] for attempt in result["attempts"]] == [0, 1]


def test_ring_exchange_is_simultaneous_and_retains_local_candidates():
    policies = initial_policies(1)
    islands = [[{"evaluation": i + 1, "policy": policy.to_dict(), "train_mean": 100 + i}]
               for i, policy in enumerate(policies)]
    original = copy.deepcopy(islands)
    assert exchange_ring(islands, 1) == []
    events = exchange_ring(islands, 2)
    assert [(event["source"], event["destination"]) for event in events] == [(0, 1), (1, 2), (2, 0)]
    for destination in range(3):
        assert islands[destination][0] == original[destination][0]
        assert islands[destination][-1]["policy"] == original[(destination - 1) % 3][0]["policy"]
        assert events[destination]["receiver_local_candidates_retained"] == 1


def test_same_initial_resources_no_exchange_for_independent(train):
    results = [run_search(method, train, seed=7, max_calls=9, max_evaluations=12)
               for method in ("single", "independent", "cooperative", "random_search")]
    initial = [[row["policy"] for row in result["history"][:3]] for result in results]
    assert all(policies == initial[0] for policies in initial)
    assert all(result["resources"]["evaluations"] == 12 for result in results)
    assert [result["resources"]["calls"] for result in results] == [9, 9, 9, 0]
    assert results[1]["exchanges"] == []
    assert [event["round"] for event in results[2]["exchanges"]] == [2, 2, 2]
    assert all(result["resources"]["validity_rate"] == 1 for result in results)


def test_ring_interval_and_invalid_interval():
    islands = [[{"evaluation": i + 1, "policy": policy.to_dict(), "train_mean": 100 + i}]
               for i, policy in enumerate(initial_policies(1))]
    assert len(exchange_ring(islands, 3, interval=3)) == 3
    assert exchange_ring(islands, 2, interval=3) == []
    with pytest.raises(ValueError):
        exchange_ring(islands, 1, interval=0)


@pytest.mark.parametrize(("interval", "rounds"), [(1, [1, 1, 1, 2, 2, 2, 3, 3, 3]), (2, [2, 2, 2]), (4, [])])
def test_no_migration_after_the_final_generation_round(train, interval, rounds):
    # 12 calls over three islands are four rounds; round 4 is never followed by a proposal.
    result = run_search("cooperative", train, seed=5, max_calls=12, max_evaluations=15,
                        exchange_interval=interval)
    assert [event["round"] for event in result["exchanges"]] == rounds
    last_round = max(attempt["round"] for attempt in result["attempts"])
    assert last_round == 4
    assert all(event["round"] < last_round for event in result["exchanges"])


def test_no_migration_once_evaluation_budget_is_exhausted(train):
    # Four rounds of calls are allowed, but evaluations run out after round 2.
    result = run_search("cooperative", train, seed=5, max_calls=12, max_evaluations=9,
                        exchange_interval=1)
    assert [event["round"] for event in result["exchanges"]] == [1, 1, 1]
    assert result["resources"]["evaluations"] == 9


def test_call_budget_is_not_capped_at_twelve(train):
    result = run_search("independent", train, seed=2, max_calls=24, max_evaluations=27)
    assert result["resources"]["calls"] == 24
    assert [sum(a["island"] == i for a in result["attempts"]) for i in range(3)] == [8, 8, 8]


def test_pruned_search_space_stays_pruned(train):
    for method in ("single", "cooperative", "random_search", "shortest_processing_time"):
        result = run_search(method, train, seed=4, max_calls=6, max_evaluations=9,
                            candidate_filter="pruned")
        assert {row["policy"]["candidate_filter"] for row in result["history"]} == {"pruned"}
        assert result["best_policy"]["candidate_filter"] == "pruned"


def test_pruned_baselines_ignore_configured_search_space(train):
    for method in ("pruned_shortest_processing_time", "pruned_most_work_remaining"):
        result = run_search(method, train, seed=4, candidate_filter="none")
        assert result["best_policy"]["candidate_filter"] == "pruned"
        assert result["resources"]["calls"] == 0 and result["resources"]["evaluations"] == 1
    with pytest.raises(ValueError):
        run_search("single", train, seed=4, candidate_filter="dominated")


def test_default_filter_keeps_unpruned_search_reproducible(train):
    # Policies saved before candidate_filter existed load as the unpruned space.
    legacy = {"weights": initial_policies(0)[0].weights, "tie_break": "job_id"}
    assert Policy.from_dict(legacy).candidate_filter == "none"
    explicit = run_search("cooperative", train, seed=8, candidate_filter="none")
    implicit = run_search("cooperative", train, seed=8)
    assert explicit["history"] == implicit["history"]


def test_candidate_filter_is_fixed_for_generated_proposals():
    spt = initial_policies(0)[0].to_dict()
    del spt["candidate_filter"]
    assert parse_candidate(json.dumps(spt), "pruned").candidate_filter == "pruned"
    assert parse_candidate(json.dumps(spt)).candidate_filter == "none"
    with pytest.raises(ValueError):
        parse_candidate(json.dumps({**spt, "candidate_filter": "none"}), "pruned")
    with pytest.raises(ValueError):
        parse_candidate("[]", "pruned")
    prompt = make_prompt([{"policy": initial_policies(0, "pruned")[0].to_dict(), "train_mean": 1.0}],
                         "A", "pruned")
    assert '"candidate_filter":"pruned"' in prompt and "fixed at pruned" in prompt


def test_fixed_seed_mock_reproduces_candidates_and_scores(train):
    first = run_search("cooperative", train, seed=8, max_calls=12, max_evaluations=15)
    second = run_search("cooperative", train, seed=8, max_calls=12, max_evaluations=15)
    for key in ("history", "attempts", "exchanges", "islands", "best_policy", "best_train_mean"):
        assert first[key] == second[key]
    assert first["resources"]["calls"] == second["resources"]["calls"] == 12
    assert first["resources"]["token_accounting"] == "estimated_mock"


def test_only_training_data_enter_generator_feedback(train, monkeypatch):
    seen = []
    original = search.MockBackend.generate

    def capture(self, prompt, history, max_output_tokens):
        seen.append((prompt, copy.deepcopy(history)))
        return original(self, prompt, history, max_output_tokens)

    monkeypatch.setattr(search.MockBackend, "generate", capture)
    result = run_search("cooperative", train, seed=12, max_calls=6, max_evaluations=9)
    assert "test" not in inspect.signature(run_search).parameters
    assert "benchmark" not in inspect.signature(run_search).parameters
    for prompt, rows in seen:
        assert "ft06" not in prompt.lower() and "known optimum" not in prompt.lower()
        for row in rows:
            assert set(row["train_makespans"]) == {instance.name for instance in train}
            assert "test" not in row and "benchmark" not in row and "reference" not in row
    assert result["selection_split"] == "train"


def test_cpu_pool_has_same_results_as_sequential_dispatch(train):
    sequential = run_search("single", train, seed=3, max_calls=2, max_evaluations=5, cpu_workers=1)
    parallel = run_search("single", train, seed=3, max_calls=2, max_evaluations=5, cpu_workers=2)
    assert parallel["history"] == sequential["history"]
    assert parallel["resources"]["cpu_workers"] == 2


@pytest.mark.parametrize("content", ['{"weights":NaN}', '```json\n{}\n```', '{"weights":Infinity}'])
def test_model_output_is_strict_json(content):
    with pytest.raises(ValueError):
        parse_candidate(content)


def test_real_backend_one_request_usage_and_no_credentials_in_prompt(monkeypatch):
    monkeypatch.setenv("SWARM_TEST_KEY", "test-only-placeholder")
    requests = []
    expected = initial_policies(0)[0].to_dict()

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, limit):
            return json.dumps({"choices": [{"message": {"content": json.dumps(expected)}}],
                               "usage": {"prompt_tokens": 17, "completion_tokens": 31}}).encode()

    def transport(request, timeout):
        requests.append(request)
        return Response()

    monkeypatch.setattr(backends, "urlopen", transport)
    backend = OpenAICompatibleBackend(model="test-model", base_url="https://models.example/v1", api_key_env="SWARM_TEST_KEY")
    output = backend.generate("training-only feedback", [], 64)
    assert len(requests) == 1
    assert output.input_tokens == 17 and output.output_tokens == 31
    assert parse_candidate(output.content).to_dict() == expected
    payload = requests[0].data.decode()
    assert "test-only-placeholder" not in payload
    assert json.loads(payload)["max_tokens"] == 64


def test_real_backend_missing_usage_is_unknown(monkeypatch):
    monkeypatch.setenv("SWARM_TEST_KEY", "test-only-placeholder")

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, limit):
            return b'{"choices":[{"message":{"content":"{}"}}]}'

    monkeypatch.setattr(backends, "urlopen", lambda *args, **kwargs: Response())
    output = OpenAICompatibleBackend(model="test-model", api_key_env="SWARM_TEST_KEY").generate("train", [], 64)
    assert output.input_tokens is output.output_tokens is None


def test_provider_overrun_stops_all_further_generation(train, monkeypatch):
    class OverrunningBackend:
        def __init__(self, seed):
            pass

        def generate(self, prompt, history, max_output_tokens):
            return Generation(json.dumps(initial_policies(0)[0].to_dict()), 100,
                              max_output_tokens + 1, "provider_reported")

    monkeypatch.setattr(search, "MockBackend", OverrunningBackend)
    result = run_search("cooperative", train, seed=1)
    assert result["resources"]["calls"] == 1
    assert result["resources"]["provider_token_overrun_calls"] == 1
    assert "further generation halted" in result["stop_reason"]


def test_real_missing_content_preserves_reported_usage(monkeypatch):
    monkeypatch.setenv("SWARM_TEST_KEY", "test-only-placeholder")

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, limit):
            return b'{"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":7}}'

    monkeypatch.setattr(backends, "urlopen", lambda *args, **kwargs: Response())
    output = OpenAICompatibleBackend(model="test-model", api_key_env="SWARM_TEST_KEY").generate("train", [], 64)
    assert (output.input_tokens, output.output_tokens) == (10, 7)
    with pytest.raises(ValueError):
        parse_candidate(output.content)


@pytest.mark.parametrize("url", ["https:///v1", "https://models.example/v1?key=x", "https://models.example/v1#fragment", "https://user:password@models.example/v1"])
def test_real_backend_rejects_unsafe_endpoint_urls(monkeypatch, url):
    monkeypatch.setenv("SWARM_TEST_KEY", "test-only-placeholder")
    with pytest.raises(ValueError):
        OpenAICompatibleBackend(model="test-model", base_url=url, api_key_env="SWARM_TEST_KEY")


def test_real_redirect_is_rejected_without_forwarding_authorization(monkeypatch):
    monkeypatch.setenv("SWARM_TEST_KEY", "test-only-placeholder")
    source_requests = []
    destination_requests = []

    class Destination(BaseHTTPRequestHandler):
        def do_POST(self):
            destination_requests.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        do_GET = do_POST

        def log_message(self, *args):
            pass

    target = HTTPServer(("127.0.0.1", 0), Destination)

    class Source(BaseHTTPRequestHandler):
        def do_POST(self):
            source_requests.append(True)
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{target.server_port}/elsewhere")
            self.end_headers()

        def log_message(self, *args):
            pass

    source = HTTPServer(("127.0.0.1", 0), Source)
    threads = [Thread(target=server.serve_forever, daemon=True) for server in (source, target)]
    for thread in threads:
        thread.start()
    try:
        backend = OpenAICompatibleBackend(
            model="test-model", base_url=f"http://127.0.0.1:{source.server_port}/v1",
            api_key_env="SWARM_TEST_KEY",
        )
        with pytest.raises(RuntimeError, match="302"):
            backend.generate("training", [], 64)
        assert source_requests == [True]
        assert destination_requests == []
    finally:
        for server in (source, target):
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=2)
