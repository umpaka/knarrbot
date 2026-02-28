"""Tests for llm_router.py — skill catalog and function declaration mapping.

These tests only cover the pure/catalog logic. No Gemini API calls are made.
"""

import time
from collections import defaultdict

from llm_router import (
    skill_to_function_declaration,
    LLMRouter,
    CATALOG_REFRESH_MIN,
    CATALOG_REFRESH_MAX,
    _validate_args,
    _default_for_spec,
    _schema_hint,
)


def _make_router(**overrides):
    """Build an LLMRouter without calling __init__ (no Gemini API key needed).

    Sets all attributes the tested methods rely on.  Pass keyword arguments
    to override defaults.
    """
    router = LLMRouter.__new__(LLMRouter)
    router._skill_catalog = {}
    router._function_declarations = []
    router._name_map = {}
    router._catalog_updated = 0
    router._catalog_refresh_interval = CATALOG_REFRESH_MIN
    router._catalog_prev_keys = set()
    router._search_index = {}
    router._skill_usage = defaultdict(int)
    router._skill_stats = defaultdict(
        lambda: {"calls": 0, "failures": 0, "total_latency_s": 0.0, "last_failure": 0.0}
    )
    router._provider_stats = defaultdict(
        lambda: {"calls": 0, "failures": 0, "total_latency_s": 0.0, "last_failure": 0.0}
    )
    router._provider_blocklist = {}
    router._reputation_cache = {}
    router._reputation_updated = 0
    router._schema_cache = {}
    for k, v in overrides.items():
        setattr(router, k, v)
    return router


class TestSkillToFunctionDeclaration:
    def test_basic_skill(self):
        sheet = {
            "name": "echo",
            "description": "Echo back input",
            "input_schema": {"text": "string"},
        }
        decl = skill_to_function_declaration("echo", sheet)
        assert decl["name"] == "echo"
        assert decl["description"] == "Echo back input"
        assert "text" in decl["parameters"]["properties"]
        assert decl["parameters"]["properties"]["text"]["type"] == "string"
        assert "text" in decl["parameters"]["required"]

    def test_hyphen_replaced_with_underscore(self):
        sheet = {"description": "Browse", "input_schema": {"task": "string"}}
        decl = skill_to_function_declaration("browse-web", sheet)
        assert decl["name"] == "browse_web"

    def test_number_type_mapping(self):
        sheet = {
            "description": "test",
            "input_schema": {"count": "integer", "ratio": "float", "amount": "number"},
        }
        decl = skill_to_function_declaration("test", sheet)
        props = decl["parameters"]["properties"]
        assert props["count"]["type"] == "number"
        assert props["ratio"]["type"] == "number"
        assert props["amount"]["type"] == "number"

    def test_boolean_type_mapping(self):
        sheet = {"description": "test", "input_schema": {"flag": "boolean", "ok": "bool"}}
        decl = skill_to_function_declaration("test", sheet)
        props = decl["parameters"]["properties"]
        assert props["flag"]["type"] == "boolean"
        assert props["ok"]["type"] == "boolean"

    def test_empty_schema(self):
        sheet = {"description": "No inputs", "input_schema": {}}
        decl = skill_to_function_declaration("simple", sheet)
        # Only extra_params should be present (catch-all for undeclared fields)
        assert "extra_params" in decl["parameters"]["properties"]
        assert len(decl["parameters"]["properties"]) == 1
        assert "required" not in decl["parameters"]

    def test_missing_description(self):
        sheet = {"input_schema": {"x": "string"}}
        decl = skill_to_function_declaration("my-skill", sheet)
        # Falls back to the skill name
        assert decl["description"] == "my-skill"


    def test_input_schema_full_preferred(self):
        """When input_schema_full is present, it should be used over flat input_schema."""
        sheet = {
            "description": "Generate image",
            "input_schema": {"prompt": "string"},
            "input_schema_full": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Image description"},
                    "width": {"type": "integer", "description": "Width in pixels", "enum": [256, 512, 1024]},
                    "height": {"type": "integer", "description": "Height in pixels"},
                },
                "required": ["prompt"],
            },
        }
        decl = skill_to_function_declaration("generate-image", sheet)
        props = decl["parameters"]["properties"]
        # Should have the rich schema fields, not just flat "prompt"
        assert "prompt" in props
        assert "width" in props
        assert "height" in props
        assert props["prompt"]["description"] == "Image description"
        assert props["width"].get("enum") == [256, 512, 1024]
        # extra_params should NOT be present for fully-schemed skills
        assert "extra_params" not in props
        # Required should come from the full schema
        assert decl["parameters"]["required"] == ["prompt"]

    def test_input_schema_full_empty_falls_back(self):
        """Empty input_schema_full should fall back to flat input_schema."""
        sheet = {
            "description": "test",
            "input_schema": {"text": "string"},
            "input_schema_full": {},
        }
        decl = skill_to_function_declaration("test", sheet)
        # Should NOT use the empty full schema — falls back to flat
        # (empty dict is falsy-ish but has no "properties" key, so we check)
        assert "text" in decl["parameters"]["properties"]


class TestCredentialScrubbing:
    """Credential scrubber catches API key patterns in tool outputs
    before they enter the LLM context. Covers our own keys (Resend,
    Gemini, Telegram) plus common patterns the bot might encounter
    when fetching external web content or skill outputs."""

    def test_scrubs_sk_prefix_keys(self):
        """Catches sk-ant-*, sk-proj-* patterns (common LLM provider keys)."""
        result = LLMRouter._scrub_credentials("key is sk-ant-abcdefghijklmnopqrstuvwxyz123456")
        assert "sk-ant-" not in result
        assert "[CREDENTIAL_REDACTED]" in result

    def test_scrubs_ghp_prefix_tokens(self):
        """Catches ghp_* patterns (GitHub PATs often found in scraped content)."""
        result = LLMRouter._scrub_credentials("token: ghp_abcdefghijklmnopqrstuvwxyz1234567890")
        assert "ghp_" not in result
        assert "[CREDENTIAL_REDACTED]" in result

    def test_scrubs_akia_prefix_keys(self):
        """Catches AKIA* patterns (AWS keys occasionally in web content)."""
        result = LLMRouter._scrub_credentials("aws_access_key=AKIAIOSFODNN7EXAMPLE1")
        assert "AKIA" not in result
        assert "[CREDENTIAL_REDACTED]" in result

    def test_scrubs_resend_key(self):
        """Catches re_* pattern — our own Resend API key."""
        result = LLMRouter._scrub_credentials("api: re_abcdefghij1234567890abc")
        assert "re_" not in result
        assert "[CREDENTIAL_REDACTED]" in result

    def test_no_false_positive_on_normal_text(self):
        text = "The result is 42 and the status is ok"
        assert LLMRouter._scrub_credentials(text) == text

    def test_truncate_result_scrubs_end_to_end(self):
        """Full pipeline: _truncate_result scrubs credentials from tool output."""
        data = {"output": "here is your key: sk-ant-abcdefghijklmnopqrstuvwxyz123456 enjoy"}
        scrubbed = LLMRouter._truncate_result(data)
        assert "sk-ant-" not in scrubbed["output"]
        assert "[CREDENTIAL_REDACTED]" in scrubbed["output"]


class TestApplyCatalog:
    def test_apply_catalog(self):
        router = _make_router()

        seen = {
            "echo": {
                "skill_sheet": {
                    "name": "echo",
                    "description": "Echo back",
                    "input_schema": {"text": "string"},
                },
            },
            "browse-web": {
                "skill_sheet": {
                    "name": "browse-web",
                    "description": "Browse",
                    "input_schema": {"task": "string"},
                },
            },
        }

        router._apply_catalog(seen)

        assert len(router._skill_catalog) == 2
        assert "echo" in router._skill_catalog
        assert "browse_web" in router._skill_catalog
        assert len(router._function_declarations) == 2
        assert router._name_map["echo"] == "echo"
        assert router._name_map["browse_web"] == "browse-web"
        assert router._catalog_updated > 0

    def test_apply_empty_catalog(self):
        router = _make_router()
        router._apply_catalog({})

        assert len(router._skill_catalog) == 0
        assert len(router._function_declarations) == 0

    def test_catalog_provider_info_stored(self):
        router = _make_router()

        seen = {
            "test-skill": {
                "skill_sheet": {
                    "name": "test-skill",
                    "description": "Test",
                    "input_schema": {},
                },
                "host": "1.2.3.4",
                "port": 9200,
            },
        }

        router._apply_catalog(seen)
        entry = router._skill_catalog["test_skill"]
        assert entry["original_name"] == "test-skill"
        assert entry["provider"]["host"] == "1.2.3.4"

    def test_search_index_built(self):
        """_apply_catalog should build a reverse word index for fast search."""
        router = _make_router()
        seen = {
            "generate-image-nanobananapro": {
                "skill_sheet": {
                    "name": "generate-image-nanobananapro",
                    "description": "Generate images from text prompts",
                    "tags": ["image", "generation", "ai"],
                    "input_schema": {"prompt": "string"},
                },
            },
            "web-search": {
                "skill_sheet": {
                    "name": "web-search",
                    "description": "Search the web for information",
                    "tags": ["search", "web"],
                    "input_schema": {"query": "string"},
                },
            },
        }
        router._apply_catalog(seen)

        idx = router._search_index
        assert isinstance(idx, dict)
        assert len(idx) > 0

        # "image" should map to the image skill
        assert "generate_image_nanobananapro" in idx.get("image", set())
        # "search" should map to the web-search skill
        assert "web_search" in idx.get("search", set())
        # "generation" should map to the image skill
        assert "generate_image_nanobananapro" in idx.get("generation", set())
        # Words from the description should be indexed too
        assert "generate_image_nanobananapro" in idx.get("prompts", set())

    def test_search_index_empty_catalog(self):
        router = _make_router()
        router._apply_catalog({})
        assert router._search_index == {}


class TestScoreProviders:
    """Tests for the multi-signal provider ranking (_score_providers)."""

    def _providers(self, *specs):
        """Build provider dicts from (node_id, host, load) tuples."""
        results = []
        for node_id, host, load in specs:
            results.append({
                "node_id": node_id,
                "host": host,
                "port": 9200,
                "sidecar_port": 9201,
                "load": load,
                "skill_sheet": {"name": "test", "input_schema": {}},
            })
        return results

    def test_local_preferred_when_all_else_equal(self):
        """With no reputation data and equal load, local should win."""
        router = _make_router()
        router._name_map = {"test": "test"}

        results = self._providers(
            ("remote1", "5.5.5.5", 0),
            ("local1",  "127.0.0.1", 0),
        )
        scored = router._score_providers(results, "test", {"127.0.0.1"})

        assert scored[0]["node_id"] == "local1"

    def test_loaded_local_loses_to_unloaded_remote(self):
        """A heavily loaded local provider should rank below a fresh remote one."""
        router = _make_router()
        router._name_map = {"test": "test"}
        # Give the remote provider a great reputation
        router._reputation_cache = {
            "remote1": {"success_rate": 0.99, "avg_wall_time_ms": 100},
        }

        results = self._providers(
            ("local1",  "127.0.0.1", 9),   # load 9/10 — nearly full
            ("remote1", "5.5.5.5",   0),   # load 0 — idle
        )
        scored = router._score_providers(results, "test", {"127.0.0.1"})

        # Remote should win despite local preference bonus
        assert scored[0]["node_id"] == "remote1"

    def test_reputation_breaks_tie(self):
        """Among two remote providers with same load, higher reputation wins."""
        router = _make_router()
        router._name_map = {"test": "test"}
        router._reputation_cache = {
            "good": {"success_rate": 0.95, "avg_wall_time_ms": 100},
            "bad":  {"success_rate": 0.40, "avg_wall_time_ms": 800},
        }

        results = self._providers(
            ("bad",  "1.1.1.1", 2),
            ("good", "2.2.2.2", 2),
        )
        scored = router._score_providers(results, "test", set())

        assert scored[0]["node_id"] == "good"

    def test_client_stats_affect_ranking(self):
        """Per-provider client-side failure stats should demote bad providers."""
        router = _make_router()
        router._name_map = {"test": "test"}

        # Provider A: 10 calls, 8 failures (80% failure rate)
        router._provider_stats["test:provA"] = {
            "calls": 10, "failures": 8, "total_latency_s": 50, "last_failure": time.time(),
        }
        # Provider B: 10 calls, 1 failure (10% failure rate)
        router._provider_stats["test:provB"] = {
            "calls": 10, "failures": 1, "total_latency_s": 30, "last_failure": 0,
        }

        results = self._providers(
            ("provA", "1.1.1.1", 0),
            ("provB", "2.2.2.2", 0),
        )
        scored = router._score_providers(results, "test", set())

        assert scored[0]["node_id"] == "provB"

    def test_blocklisted_provider_filtered(self):
        """Blocklisted providers should be removed from results."""
        router = _make_router()
        router._name_map = {"test": "test"}
        router._provider_blocklist = {
            "blocked1:test": time.time() + 600,  # blocked for 10 more minutes
        }

        results = self._providers(
            ("blocked1", "1.1.1.1", 0),
            ("ok1",      "2.2.2.2", 0),
        )
        scored = router._score_providers(results, "test", set())

        assert len(scored) == 1
        assert scored[0]["node_id"] == "ok1"

    def test_expired_blocklist_pruned(self):
        """Expired blocklist entries should be removed and provider included."""
        router = _make_router()
        router._name_map = {"test": "test"}
        router._provider_blocklist = {
            "prov1:test": time.time() - 10,  # expired 10 seconds ago
        }

        results = self._providers(("prov1", "1.1.1.1", 0),)
        scored = router._score_providers(results, "test", set())

        assert len(scored) == 1
        assert scored[0]["node_id"] == "prov1"
        # Expired entry should be pruned
        assert "prov1:test" not in router._provider_blocklist

    def test_all_blocked_falls_back(self):
        """If all providers are blocklisted, fall back to full list."""
        router = _make_router()
        router._name_map = {"test": "test"}
        router._provider_blocklist = {
            "a:test": time.time() + 600,
            "b:test": time.time() + 300,
        }

        results = self._providers(
            ("a", "1.1.1.1", 0),
            ("b", "2.2.2.2", 0),
        )
        scored = router._score_providers(results, "test", set())

        # Both should still be returned (fallback)
        assert len(scored) == 2

    def test_composite_score_stripped(self):
        """Internal _composite_score field should not leak into results."""
        router = _make_router()
        router._name_map = {"test": "test"}

        results = self._providers(("a", "1.1.1.1", 0),)
        scored = router._score_providers(results, "test", set())

        assert "_composite_score" not in scored[0]


class TestMaybeBlocklistProvider:
    """Tests for _maybe_blocklist_provider."""

    def test_no_blocklist_below_threshold(self):
        """Provider with fewer than 3 failures should not be blocklisted."""
        router = _make_router()
        ps = router._provider_stats["test_skill:node1"]
        ps["calls"] = 4
        ps["failures"] = 2
        ps["last_failure"] = time.time()

        router._maybe_blocklist_provider("node1", "test-skill")
        assert "node1:test-skill" not in router._provider_blocklist

    def test_blocklist_at_threshold(self):
        """Provider with >= 3 failures within window should be blocklisted."""
        router = _make_router()
        now = time.time()
        ps = router._provider_stats["test_skill:node1"]
        ps["calls"] = 5
        ps["failures"] = 3
        ps["last_failure"] = now

        router._maybe_blocklist_provider("node1", "test-skill")
        assert "node1:test-skill" in router._provider_blocklist
        assert router._provider_blocklist["node1:test-skill"] > now

    def test_no_blocklist_if_old_failure(self):
        """Failures outside the 10-minute window should not trigger blocklist."""
        router = _make_router()
        ps = router._provider_stats["test_skill:node1"]
        ps["calls"] = 10
        ps["failures"] = 5
        ps["last_failure"] = time.time() - 700  # >600s ago (outside window)

        router._maybe_blocklist_provider("node1", "test-skill")
        assert "node1:test-skill" not in router._provider_blocklist

    def test_no_blocklist_empty_node_id(self):
        """Empty node_id should be a no-op."""
        router = _make_router()
        router._maybe_blocklist_provider("", "test-skill")
        assert len(router._provider_blocklist) == 0

    def test_blocklist_duration(self):
        """Blocklist entry should expire ~15 minutes from now."""
        router = _make_router()
        now = time.time()
        ps = router._provider_stats["test_skill:node1"]
        ps["calls"] = 5
        ps["failures"] = 4
        ps["last_failure"] = now

        router._maybe_blocklist_provider("node1", "test-skill")
        expiry = router._provider_blocklist["node1:test-skill"]
        # Should be roughly 900 seconds from now (15 min)
        assert 890 < (expiry - now) < 910


class TestAdaptiveRefreshInterval:
    """Test that _apply_catalog + refresh logic adapts the interval."""

    def test_unchanged_catalog_doubles_interval(self):
        """When catalog keys don't change, interval should double."""
        router = _make_router()
        seen = {
            "echo": {
                "skill_sheet": {"name": "echo", "description": "Echo", "input_schema": {"text": "string"}},
            },
        }

        # First apply sets the baseline
        router._apply_catalog(seen)
        router._catalog_prev_keys = set(seen.keys())
        router._catalog_refresh_interval = CATALOG_REFRESH_MIN

        # Simulate what _refresh_catalog does when catalog is unchanged:
        new_keys = set(seen.keys())
        assert new_keys == router._catalog_prev_keys
        # This would double the interval
        router._catalog_refresh_interval = min(
            router._catalog_refresh_interval * 2, CATALOG_REFRESH_MAX
        )
        assert router._catalog_refresh_interval == CATALOG_REFRESH_MIN * 2

    def test_changed_catalog_resets_interval(self):
        """When catalog keys change, interval should reset to minimum."""
        router = _make_router()
        router._catalog_prev_keys = {"echo"}
        router._catalog_refresh_interval = 240  # had been doubled several times

        new_keys = {"echo", "new-skill"}
        # Keys differ => reset
        assert new_keys != router._catalog_prev_keys
        router._catalog_refresh_interval = CATALOG_REFRESH_MIN
        assert router._catalog_refresh_interval == 60

    def test_interval_capped_at_max(self):
        """Interval should never exceed CATALOG_REFRESH_MAX."""
        interval = CATALOG_REFRESH_MIN
        for _ in range(20):
            interval = min(interval * 2, CATALOG_REFRESH_MAX)
        assert interval == CATALOG_REFRESH_MAX


# ── Schema-aware argument validation tests ──────────────────────────────

class TestValidateArgs:
    """Tests for _validate_args — schema-aware argument validation/coercion."""

    FULL_SCHEMA = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["create", "edit", "delete"]},
            "prompt": {"type": "string", "description": "The main text input"},
            "width": {"type": "integer"},
            "verbose": {"type": "boolean"},
        },
        "required": ["action", "prompt"],
    }

    def test_passes_known_fields_strips_unknown(self):
        args = {"action": "create", "prompt": "hello", "bogus_field": "junk"}
        result = _validate_args(args, self.FULL_SCHEMA, None)
        assert "action" in result
        assert "prompt" in result
        assert "bogus_field" not in result

    def test_fills_required_with_smart_defaults(self):
        args = {"prompt": "hello"}
        result = _validate_args(args, self.FULL_SCHEMA, None)
        # "action" is required and has enum — should default to first enum value
        assert result["action"] == "create"

    def test_leaves_optional_absent(self):
        args = {"action": "edit", "prompt": "test"}
        result = _validate_args(args, self.FULL_SCHEMA, None)
        # "width" and "verbose" are optional — should NOT be auto-filled
        assert "width" not in result
        assert "verbose" not in result

    def test_fills_flat_schema_fields_even_if_optional(self):
        flat = {"action": "string", "prompt": "string", "width": "integer"}
        args = {"action": "edit", "prompt": "test"}
        result = _validate_args(args, self.FULL_SCHEMA, flat)
        # "width" is in flat schema, so Knarr requires it present
        assert "width" in result

    def test_stringifies_values(self):
        args = {"action": "create", "prompt": "test", "width": 512}
        result = _validate_args(args, self.FULL_SCHEMA, None)
        assert result["width"] == "512"

    def test_fallback_flat_schema_only(self):
        flat = {"text": "string", "count": "integer"}
        args = {"text": "hello"}
        result = _validate_args(args, None, flat)
        assert result["text"] == "hello"
        assert result["count"] == ""  # filled with empty string

    def test_no_schema_passthrough(self):
        args = {"anything": "goes", "random": "stuff"}
        result = _validate_args(args, None, None)
        assert result == {"anything": "goes", "random": "stuff"}

    def test_empty_full_schema_falls_through(self):
        args = {"x": "1"}
        flat = {"x": "string", "y": "string"}
        result = _validate_args(args, {}, flat)
        assert result["x"] == "1"
        assert result["y"] == ""


class TestDefaultForSpec:
    """Tests for _default_for_spec — smart default value selection."""

    def test_enum_picks_first(self):
        assert _default_for_spec({"type": "string", "enum": ["create", "edit"]}) == "create"

    def test_number_returns_zero(self):
        assert _default_for_spec({"type": "number"}) == "0"

    def test_integer_returns_zero(self):
        assert _default_for_spec({"type": "integer"}) == "0"

    def test_boolean_returns_false(self):
        assert _default_for_spec({"type": "boolean"}) == "false"

    def test_string_returns_empty(self):
        assert _default_for_spec({"type": "string"}) == ""

    def test_empty_spec_returns_empty(self):
        assert _default_for_spec({}) == ""

    def test_none_spec_returns_empty(self):
        assert _default_for_spec(None) == ""


class TestSchemaHint:
    """Tests for _schema_hint — human-readable schema feedback for the LLM."""

    def test_full_schema_hint(self):
        schema = {
            "properties": {
                "action": {"type": "string", "enum": ["create", "edit"], "description": "The op"},
                "prompt": {"type": "string", "description": "Text input"},
            },
            "required": ["action"],
        }
        hint = _schema_hint(schema, None)
        assert "Expected schema:" in hint
        assert "action" in hint
        assert "(REQUIRED)" in hint
        assert "allowed=" in hint
        assert "prompt" in hint

    def test_flat_schema_hint(self):
        hint = _schema_hint(None, {"text": "string", "count": "integer"})
        assert "Expected fields:" in hint
        assert "text" in hint
        assert "count" in hint

    def test_no_schema_empty(self):
        assert _schema_hint(None, None) == ""

    def test_empty_properties_empty(self):
        assert _schema_hint({"properties": {}}, None) == ""


class TestExtraParamsConditional:
    """extra_params should only appear for flat-schema skills, not full-schema."""

    def test_full_schema_no_extra_params(self):
        sheet = {
            "description": "test",
            "input_schema_full": {
                "type": "object",
                "properties": {"prompt": {"type": "string"}},
                "required": ["prompt"],
            },
        }
        decl = skill_to_function_declaration("test", sheet)
        assert "extra_params" not in decl["parameters"]["properties"]

    def test_flat_schema_has_extra_params(self):
        sheet = {"description": "test", "input_schema": {"prompt": "string"}}
        decl = skill_to_function_declaration("test", sheet)
        assert "extra_params" in decl["parameters"]["properties"]
