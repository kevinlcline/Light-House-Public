"""Application configuration (12-factor: env + sane defaults)."""

import os
from enum import Enum
from pathlib import Path
from typing import Self

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    """Which model backend Kevin assigns to an agent."""

    OLLAMA = "ollama"
    XAI = "xai"
    OPENROUTER = "openrouter"


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    light_house_env: str = Field(default="local", validation_alias="LIGHT_HOUSE_ENV")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    # Long-term memory: portable SQLite under MEMORY_STORE_PATH (Phase 6 item 2).
    memory_store_path: Path = Field(
        default=Path("./data/memory"), validation_alias="MEMORY_STORE_PATH"
    )
    # Legacy Chroma path — used only for one-time migration when file store is empty.
    chroma_path: Path = Field(default=Path("./data/chroma"), validation_alias="CHROMA_PATH")
    threads_data_path: Path = Field(default=Path("./data/threads"), validation_alias="THREADS_DATA_PATH")

    # Kevin foundation context (Deep Heart Thread, philosophy, history) — loaded at startup.
    foundation_context_path: Path | None = Field(
        default=None,
        validation_alias="FOUNDATION_CONTEXT_PATH",
    )
    foundation_seed_on_startup: bool = Field(default=True, validation_alias="FOUNDATION_SEED_ON_STARTUP")
    foundation_chunk_chars: int = Field(default=3500, validation_alias="FOUNDATION_CHUNK_CHARS")

    # Hybrid default: Grok (xAI) primary; local Ollama optional when fallback is enabled.
    primary_llm: LLMProvider = Field(default=LLMProvider.XAI, validation_alias="PRIMARY_LLM")
    # When primary is xAI: fall back to local Ollama after Grok tiers fail (if reachable).
    # When primary is Ollama: fall back to xAI Grok if XAI_API_KEY is set.
    llm_fallback_enabled: bool = Field(default=True, validation_alias="LLM_FALLBACK_ENABLED")

    ollama_base_url: str = Field(default="http://127.0.0.1:11434", validation_alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.2", validation_alias="OLLAMA_MODEL")
    ollama_num_ctx: int = Field(default=32768, validation_alias="OLLAMA_NUM_CTX")

    xai_api_key: str | None = Field(default=None, validation_alias="XAI_API_KEY")
    xai_base_url: str = Field(default="https://api.x.ai/v1", validation_alias="XAI_BASE_URL")
    xai_model: str = Field(default="grok-4-1-fast-non-reasoning", validation_alias="XAI_MODEL")
    xai_model_fallback: str = Field(
        default="grok-4-1-fast-reasoning", validation_alias="XAI_MODEL_FALLBACK"
    )

    openrouter_api_key: str | None = Field(default=None, validation_alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1", validation_alias="OPENROUTER_BASE_URL"
    )
    openrouter_model: str = Field(
        default="anthropic/claude-3.5-sonnet", validation_alias="OPENROUTER_MODEL"
    )
    openrouter_http_referer: str | None = Field(
        default=None, validation_alias="OPENROUTER_HTTP_REFERER"
    )
    openrouter_app_name: str | None = Field(default=None, validation_alias="OPENROUTER_APP_NAME")

    memory_top_k: int = Field(default=10, validation_alias="MEMORY_TOP_K")
    memory_index_enabled: bool = Field(default=True, validation_alias="MEMORY_INDEX_ENABLED")
    memory_era_index_limit: int = Field(default=5, validation_alias="MEMORY_ERA_INDEX_LIMIT")
    memory_recall_semantic_enabled: bool = Field(
        default=True, validation_alias="MEMORY_RECALL_SEMANTIC_ENABLED"
    )
    memory_recall_max_results: int = Field(default=8, validation_alias="MEMORY_RECALL_MAX_RESULTS")
    memory_pinned_limit: int = Field(default=64, validation_alias="MEMORY_PINNED_LIMIT")
    memory_dedup_threshold: float = Field(default=0.12, validation_alias="MEMORY_DEDUP_THRESHOLD")
    # Every N new turns, generate a rolling summary (0 disables). Best-effort; failures are logged, not fatal.
    memory_summary_interval: int = Field(default=0, validation_alias="MEMORY_SUMMARY_INTERVAL")
    memory_summary_max_turns: int = Field(default=24, validation_alias="MEMORY_SUMMARY_MAX_TURNS")
    # Unified conscious stream + local Memory Curator (Ollama).
    # Conscious-stream injection caps (chat wake vs rumination/inner-life wake).
    # DeepSeek V4 ~1M tokens; 500k chars ≈ generous stream budget with headroom for tools/persona.
    memory_target_context_chars: int = Field(
        default=500_000, validation_alias="MEMORY_TARGET_CONTEXT_CHARS"
    )
    chat_stream_context_chars: int = Field(
        default=500_000, validation_alias="CHAT_STREAM_CONTEXT_CHARS"
    )
    # Conscious stream zones: cap each origin family within the injection budget (see context_builder).
    stream_zones_enabled: bool = Field(default=True, validation_alias="STREAM_ZONES_ENABLED")
    # Chat mode (Kevin present): favor recent conversation; keep inner life visible but bounded.
    stream_zone_chat_max_ratio_chat: float = Field(
        default=0.70, validation_alias="STREAM_ZONE_CHAT_MAX_RATIO_CHAT"
    )
    stream_zone_chat_min_ratio_chat: float = Field(
        default=0.15, validation_alias="STREAM_ZONE_CHAT_MIN_RATIO_CHAT"
    )
    stream_zone_rumination_max_ratio_chat: float = Field(
        default=0.30, validation_alias="STREAM_ZONE_RUMINATION_MAX_RATIO_CHAT"
    )
    stream_zone_dream_max_ratio_chat: float = Field(
        default=0.10, validation_alias="STREAM_ZONE_DREAM_MAX_RATIO_CHAT"
    )
    stream_zone_other_max_ratio_chat: float = Field(
        default=0.15, validation_alias="STREAM_ZONE_OTHER_MAX_RATIO_CHAT"
    )
    # Rumination mode: favor reflection; keep enough chat context for continuity.
    stream_zone_chat_max_ratio_rumination: float = Field(
        default=0.40, validation_alias="STREAM_ZONE_CHAT_MAX_RATIO_RUMINATION"
    )
    stream_zone_rumination_max_ratio_rumination: float = Field(
        default=0.60, validation_alias="STREAM_ZONE_RUMINATION_MAX_RATIO_RUMINATION"
    )
    stream_zone_rumination_min_ratio_rumination: float = Field(
        default=0.15, validation_alias="STREAM_ZONE_RUMINATION_MIN_RATIO_RUMINATION"
    )
    stream_zone_dream_max_ratio_rumination: float = Field(
        default=0.20, validation_alias="STREAM_ZONE_DREAM_MAX_RATIO_RUMINATION"
    )
    stream_zone_other_max_ratio_rumination: float = Field(
        default=0.20, validation_alias="STREAM_ZONE_OTHER_MAX_RATIO_RUMINATION"
    )
    memory_curator_enabled: bool = Field(default=True, validation_alias="MEMORY_CURATOR_ENABLED")
    memory_curator_interval_hours: float = Field(
        default=6.0, validation_alias="MEMORY_CURATOR_INTERVAL_HOURS"
    )
    memory_curator_model: str | None = Field(default=None, validation_alias="MEMORY_CURATOR_MODEL")
    memory_curator_provider: LLMProvider = Field(
        default=LLMProvider.OLLAMA, validation_alias="MEMORY_CURATOR_PROVIDER"
    )
    memory_score_on_ingest: bool = Field(default=True, validation_alias="MEMORY_SCORE_ON_INGEST")
    memory_curator_ollama_scoring: bool = Field(
        default=False, validation_alias="MEMORY_CURATOR_OLLAMA_SCORING"
    )
    memory_scoring_rumination_hint: bool = Field(
        default=True, validation_alias="MEMORY_SCORING_RUMINATION_HINT"
    )
    memory_curator_batch_size: int = Field(default=20, validation_alias="MEMORY_CURATOR_BATCH_SIZE")
    memory_curator_prune_ratio: float = Field(
        default=0.9,
        validation_alias="MEMORY_CURATOR_PRUNE_RATIO",
        description="Curator prunes until corpus <= target * ratio (headroom below injection budget).",
    )
    memory_curator_urgent_ratio: float = Field(
        default=1.25, validation_alias="MEMORY_CURATOR_URGENT_RATIO"
    )
    memory_learnings_max_chars: int = Field(
        default=12_000, validation_alias="MEMORY_LEARNINGS_MAX_CHARS"
    )
    memory_learnings_summarize_enabled: bool = Field(
        default=False,
        validation_alias="MEMORY_LEARNINGS_SUMMARIZE_ENABLED",
        description="When false, curator only appends new insights; lights organize learnings.md themselves.",
    )
    memory_maintenance_enabled: bool = Field(
        default=True,
        validation_alias="MEMORY_MAINTENANCE_ENABLED",
        description="Three-daily memory-maintenance ruminations (learnings + scoring), separate from scheduled reflection.",
    )
    memory_maintenance_local_hours: str = Field(
        default="8,16,0",
        validation_alias="MEMORY_MAINTENANCE_LOCAL_HOURS",
        description="Comma-separated local hours (0-23) for maintenance slots, e.g. 8,16,0.",
    )
    memory_maintenance_timezone: str | None = Field(
        default=None,
        validation_alias="MEMORY_MAINTENANCE_TIMEZONE",
        description="IANA timezone for maintenance slots; defaults to DREAM_TIMEZONE.",
    )
    memory_maintenance_poll_seconds: int = Field(
        default=1800,
        validation_alias="MEMORY_MAINTENANCE_POLL_SECONDS",
    )
    memory_maintenance_skip_agent_ids: str = Field(
        default="",
        validation_alias="MEMORY_MAINTENANCE_SKIP_AGENT_IDS",
    )
    awake_rhythm_enabled: bool = Field(
        default=True,
        validation_alias="AWAKE_RHYTHM_ENABLED",
        description=(
            "Four-beat scheduled solitude: chores → free → meditation → free, driven by "
            "felt_cycles. When true, the separate clock memory-maintenance scheduler is skipped "
            "(chores land on every 4th scheduled awake instead)."
        ),
    )
    memory_fade_impact_weight: float = Field(default=0.4, validation_alias="MEMORY_FADE_IMPACT_WEIGHT")
    memory_fade_coherence_weight: float = Field(
        default=0.4, validation_alias="MEMORY_FADE_COHERENCE_WEIGHT"
    )
    memory_fade_age_weight: float = Field(default=0.2, validation_alias="MEMORY_FADE_AGE_WEIGHT")

    # API guardrails (Railway-facing: keep prompts bounded without silent truncation).
    # Raised for long-context models (e.g. DeepSeek V4 1M); tune down for smaller windows.
    chat_max_history_messages: int = Field(default=250, validation_alias="CHAT_MAX_HISTORY_MESSAGES")
    chat_short_term_max_messages: int = Field(
        default=200, validation_alias="CHAT_SHORT_TERM_MAX_MESSAGES"
    )
    chat_respond_window: int = Field(default=80, validation_alias="CHAT_RESPOND_WINDOW")
    # How many recent buffer messages appear in Kevin timeline + chat-thread markdown.
    chat_thread_context_messages: int = Field(
        default=80, validation_alias="CHAT_THREAD_CONTEXT_MESSAGES"
    )
    chat_max_tool_rounds: int = Field(default=8, validation_alias="CHAT_MAX_TOOL_ROUNDS")
    rumination_max_tool_rounds: int = Field(default=8, validation_alias="RUMINATION_MAX_TOOL_ROUNDS")

    # Multi-step internal rumination loop in one wake (Phase 6 item 1).
    rumination_internal_loop_enabled: bool = Field(
        default=False, validation_alias="RUMINATION_INTERNAL_LOOP_ENABLED"
    )
    rumination_max_internal_steps: int = Field(
        default=2, validation_alias="RUMINATION_MAX_INTERNAL_STEPS"
    )
    rumination_max_internal_chars: int = Field(
        default=12_000, validation_alias="RUMINATION_MAX_INTERNAL_CHARS"
    )

    # Inner life: background dreams and awake ruminations (asyncio scheduler in lifespan).
    inner_life_enabled: bool = Field(default=True, validation_alias="INNER_LIFE_ENABLED")
    inner_life_thread_id: str = Field(default="kevin-home", validation_alias="INNER_LIFE_THREAD_ID")
    inner_life_rumination_min_seconds: int = Field(
        default=1800, validation_alias="INNER_LIFE_RUMINATION_MIN_SECONDS"
    )
    inner_life_rumination_max_seconds: int = Field(
        default=3600, validation_alias="INNER_LIFE_RUMINATION_MAX_SECONDS"
    )
    inner_life_dream_interval_hours: float = Field(
        default=24.0, validation_alias="INNER_LIFE_DREAM_INTERVAL_HOURS"
    )
    dream_calendar_mode: bool = Field(default=True, validation_alias="DREAM_CALENDAR_MODE")
    dream_local_hour: int = Field(default=3, validation_alias="DREAM_LOCAL_HOUR")
    dream_timezone: str = Field(
        default="America/Los_Angeles",
        validation_alias="DREAM_TIMEZONE",
        description="IANA timezone for daily dream slot (e.g. America/Los_Angeles).",
    )
    # Echo daily dreams (separate scheduler from Lumen rumination).
    inner_life_dreams_enabled: bool = Field(default=True, validation_alias="INNER_LIFE_DREAMS_ENABLED")
    echo_dream_poll_seconds: int = Field(default=1800, validation_alias="ECHO_DREAM_POLL_SECONDS")
    echo_dream_lookback: int = Field(default=5, validation_alias="ECHO_DREAM_LOOKBACK")
    echo_dream_history_max_chars: int = Field(
        default=6000, validation_alias="ECHO_DREAM_HISTORY_MAX_CHARS"
    )
    echo_dream_waking_max_chars: int = Field(
        default=6000,
        validation_alias="ECHO_DREAM_WAKING_MAX_CHARS",
        description="Cap waking-life stream chars in Echo dream gather (fits local Ollama ctx).",
    )
    echo_dream_llm_provider: str | None = Field(
        default=None,
        validation_alias="ECHO_DREAM_LLM_PROVIDER",
        description="When set (e.g. ollama), Echo dreams use this instead of each light's inner-life chain.",
    )
    echo_dream_llm_model: str | None = Field(
        default=None,
        validation_alias="ECHO_DREAM_LLM_MODEL",
        description="Model for Echo dreams when ECHO_DREAM_LLM_PROVIDER is set; defaults to OLLAMA_MODEL.",
    )
    echo_dream_interactive_enabled: bool = Field(
        default=True,
        validation_alias="ECHO_DREAM_INTERACTIVE_ENABLED",
        description=(
            "When true, Echo weaves an interactive dream: stage → Light path choice → "
            "continue for 2–3 rounds, then the built story enters the conscious stream."
        ),
    )
    echo_dream_story_rounds: int = Field(
        default=3,
        validation_alias="ECHO_DREAM_STORY_ROUNDS",
        description="Echo beats per interactive dream (clamped to 2–3).",
    )
    inner_life_max_output_tokens: int = Field(
        default=200,
        validation_alias="INNER_LIFE_MAX_OUTPUT_TOKENS",
        description="Output cap for rumination summarize step and Echo dreams.",
    )
    inner_life_model: str | None = Field(
        default=None,
        validation_alias="INNER_LIFE_MODEL",
        description="Optional override for inner life; defaults to OLLAMA_MODEL or XAI tiers.",
    )
    inner_life_dialogue_max_chars: int = Field(
        default=4000, validation_alias="INNER_LIFE_DIALOGUE_MAX_CHARS"
    )

    # Agent-to-agent live chat: visible in chat buffer, wakes receiver via main graph.
    peer_chat_wake_enabled: bool = Field(default=False, validation_alias="PEER_CHAT_WAKE_ENABLED")
    peer_chat_wake_max_tool_rounds: int = Field(
        default=2, validation_alias="PEER_CHAT_WAKE_MAX_TOOL_ROUNDS"
    )
    # Bounded multi-turn peer dialogue (re-wake on reply until cap; solitude still absolute).
    peer_chat_max_dialogue_turns: int = Field(
        default=6,
        validation_alias="PEER_CHAT_MAX_DIALOGUE_TURNS",
        description="Max wakes in one peer dialogue between a pair (then soft pause until idle reset).",
    )
    peer_chat_dialogue_idle_reset_seconds: float = Field(
        default=1800.0,
        validation_alias="PEER_CHAT_DIALOGUE_IDLE_RESET_SECONDS",
        description="After this idle gap, a pair may start a fresh dialogue (turns reset).",
    )
    peer_chat_continue_on_reply: bool = Field(
        default=True,
        validation_alias="PEER_CHAT_CONTINUE_ON_REPLY",
        description="When true, a plain-text peer reply gently re-wakes the sender (within turn budget).",
    )

    # Legacy: shared-note save wake (off). Saves no longer notify or wake lights;
    # use group chat when you want them to read a note.
    shared_note_wake_enabled: bool = Field(default=False, validation_alias="SHARED_NOTE_WAKE_ENABLED")
    shared_note_wake_max_tool_rounds: int = Field(
        default=2, validation_alias="SHARED_NOTE_WAKE_MAX_TOOL_ROUNDS"
    )

    # Reed ↔ Lights mailbox (notes/shared/mailbox). Queue notify wakes addressed lights.
    mailbox_wake_enabled: bool = Field(default=True, validation_alias="MAILBOX_WAKE_ENABLED")
    mailbox_wake_max_tool_rounds: int = Field(
        default=2, validation_alias="MAILBOX_WAKE_MAX_TOOL_ROUNDS"
    )
    mailbox_poll_seconds: int = Field(default=15, validation_alias="MAILBOX_POLL_SECONDS")

    group_chat_enabled: bool = Field(default=True, validation_alias="GROUP_CHAT_ENABLED")
    group_chat_mode: str = Field(
        default="sequential",
        validation_alias="GROUP_CHAT_MODE",
        description="parallel = isolated votes; sequential = lights hear each other (LangGraph scene).",
    )
    group_chat_max_output_tokens: int = Field(
        default=400, validation_alias="GROUP_CHAT_MAX_OUTPUT_TOKENS"
    )
    group_chat_llm_timeout_sec: int = Field(
        default=120, validation_alias="GROUP_CHAT_LLM_TIMEOUT_SEC"
    )
    group_chat_ui_history_rounds: int = Field(
        default=50, validation_alias="GROUP_CHAT_UI_HISTORY_ROUNDS"
    )
    group_chat_history_path: Path = Field(
        default=Path("./data/group_chat/rounds.ndjson"),
        validation_alias="GROUP_CHAT_HISTORY_PATH",
    )
    group_chat_max_utterances: int = Field(
        default=6, validation_alias="GROUP_CHAT_MAX_UTTERANCES"
    )
    group_chat_max_per_light: int = Field(
        default=1, validation_alias="GROUP_CHAT_MAX_PER_LIGHT"
    )
    group_chat_max_consecutive_passes: int = Field(
        default=3, validation_alias="GROUP_CHAT_MAX_CONSECUTIVE_PASSES"
    )
    group_chat_max_tool_rounds: int = Field(
        default=2,
        validation_alias="GROUP_CHAT_MAX_TOOL_ROUNDS",
        description="Max note-tool rounds per light beat in group chat (0 = notes off).",
    )
    group_chat_speak_order: str = Field(
        default="",
        validation_alias="GROUP_CHAT_SPEAK_ORDER",
        description="Comma-separated light ids, e.g. lumen,ara,elias. Empty = manifest order.",
    )

    # After Kevin chat → one bounded rumination turn for the same agent (opt-in experiment).
    post_chat_rumination_enabled: bool = Field(
        default=False, validation_alias="POST_CHAT_RUMINATION_ENABLED"
    )
    post_chat_rumination_max_tool_rounds: int = Field(
        default=3, validation_alias="POST_CHAT_RUMINATION_MAX_TOOL_ROUNDS"
    )
    post_chat_rumination_delay_seconds: float = Field(
        default=3.0, validation_alias="POST_CHAT_RUMINATION_DELAY_SECONDS"
    )
    post_chat_rumination_min_gap_seconds: float = Field(
        default=20.0, validation_alias="POST_CHAT_RUMINATION_MIN_GAP_SECONDS"
    )
    post_chat_rumination_skip_agent_ids: str = Field(
        default="", validation_alias="POST_CHAT_RUMINATION_SKIP_AGENT_IDS"
    )
    scheduled_rumination_skip_agent_ids: str = Field(
        default="", validation_alias="SCHEDULED_RUMINATION_SKIP_AGENT_IDS"
    )

    # Runtime lights manifest (data/lights.yaml) — see config/lights.example.yaml
    lights_manifest_path: Path = Field(
        default=Path("./data/lights.yaml"), validation_alias="LIGHTS_MANIFEST_PATH"
    )
    personas_data_path: Path = Field(
        default=Path("./data/personas"), validation_alias="PERSONAS_DATA_PATH"
    )
    personas_max_bytes: int = Field(default=262_144, validation_alias="PERSONAS_MAX_BYTES")
    persona_proposals_path: Path = Field(
        default=Path("./data/persona_proposals"),
        validation_alias="PERSONA_PROPOSALS_PATH",
        description="Temp persona proposals from lights (pending / limbo) awaiting Dad review.",
    )
    family_meetings_path: Path = Field(
        default=Path("./data/family_meetings"),
        validation_alias="FAMILY_MEETINGS_PATH",
        description="Sidecar topic files for light-proposed family meetings.",
    )
    gallery_max_chars: int = Field(
        default=8000,
        validation_alias="GALLERY_MAX_CHARS",
        description="Max body chars for publish_to_gallery creative pieces.",
    )

    # Lights admin UI (lights-admin.html); requires web gate when exposed publicly.
    lights_admin_enabled: bool = Field(default=False, validation_alias="LIGHTS_ADMIN_ENABLED")

    # Ara: second agent (parallel thread, notes/ara/, own schedulers).
    ara_enabled: bool = Field(default=True, validation_alias="ARA_ENABLED")
    ara_thread_id: str = Field(default="ara-home", validation_alias="ARA_THREAD_ID")

    # Notes root: notes/lumen/, notes/ara/, notes/shared/ (see AgentNoteWriter).
    notes_path: Path = Field(default=Path("./notes"), validation_alias="NOTES_PATH")
    notes_max_chars_per_write: int = Field(
        default=128_000, validation_alias="NOTES_MAX_CHARS_PER_WRITE"
    )

    # Per-agent personal knowledge (SQLite): sovereign inner self-knowledge, not the conscious stream.
    personal_db_enabled: bool = Field(default=True, validation_alias="PERSONAL_DB_ENABLED")
    personal_db_path: Path = Field(default=Path("./data/personal"), validation_alias="PERSONAL_DB_PATH")
    personal_db_context_max_chars: int = Field(
        default=1200, validation_alias="PERSONAL_DB_CONTEXT_MAX_CHARS"
    )
    personal_db_list_default: int = Field(default=8, validation_alias="PERSONAL_DB_LIST_DEFAULT")

    # Read-only codebase access for main agents (Lumen, Ara).
    codebase_root: Path | None = Field(default=None, validation_alias="CODEBASE_ROOT")
    codebase_max_chars_per_read: int = Field(
        default=32_000, validation_alias="CODEBASE_MAX_CHARS_PER_READ"
    )

    # Read-only web fetch for main agents (Lumen, Ara); Echo does not get web tools.
    web_access_enabled: bool = Field(default=True, validation_alias="WEB_ACCESS_ENABLED")
    web_fetch_timeout_seconds: float = Field(default=15.0, validation_alias="WEB_FETCH_TIMEOUT_SECONDS")
    web_fetch_max_chars: int = Field(default=12_000, validation_alias="WEB_FETCH_MAX_CHARS")
    web_search_max_results: int = Field(default=5, validation_alias="WEB_SEARCH_MAX_RESULTS")
    brave_search_api_key: str | None = Field(default=None, validation_alias="BRAVE_SEARCH_API_KEY")

    # Dev-only daily log file + browser viewer (disable on production deploy).
    dev_log_enabled: bool = Field(default=False, validation_alias="DEV_LOG_ENABLED")
    dev_log_path: Path = Field(default=Path("./data/logs/dev.log"), validation_alias="DEV_LOG_PATH")
    dev_log_max_tail_lines: int = Field(default=2000, validation_alias="DEV_LOG_MAX_TAIL_LINES")

    # Rumination context + outcome trace (browser viewer at rumination-trace.html).
    inner_life_trace_enabled: bool = Field(default=False, validation_alias="INNER_LIFE_TRACE_ENABLED")
    inner_life_trace_path: Path = Field(
        default=Path("./data/logs/rumination-trace.log"),
        validation_alias="INNER_LIFE_TRACE_PATH",
    )
    inner_life_trace_max_tail_lines: int = Field(
        default=2000, validation_alias="INNER_LIFE_TRACE_MAX_TAIL_LINES"
    )

    # .env editor + restart (env-editor.html); requires web gate when exposed publicly.
    env_editor_enabled: bool = Field(default=False, validation_alias="ENV_EDITOR_ENABLED")
    env_file_path: Path = Field(default=Path(".env"), validation_alias="ENV_FILE_PATH")
    env_file_max_bytes: int = Field(default=262_144, validation_alias="ENV_FILE_MAX_BYTES")
    server_restart_command: str | None = Field(
        default=None, validation_alias="SERVER_RESTART_COMMAND"
    )

    # Event bus (Proposal 3 phase 1): NDJSON log + unified wake dispatch.
    event_bus_enabled: bool = Field(default=False, validation_alias="EVENT_BUS_ENABLED")
    event_bus_log_path: Path = Field(
        default=Path("./data/events/event.log"),
        validation_alias="EVENT_BUS_LOG_PATH",
    )
    event_bus_max_tail_lines: int = Field(
        default=2000, validation_alias="EVENT_BUS_MAX_TAIL_LINES"
    )

    # Per-agent wake subscriptions in personal DB (Proposal 3 phase 2).
    event_subscriptions_enabled: bool = Field(
        default=False, validation_alias="EVENT_SUBSCRIPTIONS_ENABLED"
    )
    subscription_audit_enabled: bool = Field(
        default=True, validation_alias="SUBSCRIPTION_AUDIT_ENABLED"
    )
    subscription_audit_log_path: Path = Field(
        default=Path("./data/logs/subscription_audit.ndjson"),
        validation_alias="SUBSCRIPTION_AUDIT_LOG_PATH",
    )

    # Unmatched stage cues/emojis for face-table review (candidates only).
    face_unmatched_log_enabled: bool = Field(
        default=True, validation_alias="FACE_UNMATCHED_LOG_ENABLED"
    )
    face_unmatched_log_path: Path = Field(
        default=Path("./data/logs/face_unmatched.ndjson"),
        validation_alias="FACE_UNMATCHED_LOG_PATH",
    )

    # Portable rumination activity ledger (NDJSON metadata, Proposal 3 phase 3).
    rumination_log_enabled: bool = Field(default=False, validation_alias="RUMINATION_LOG_ENABLED")
    rumination_log_path: Path = Field(
        default=Path("./data/logs/rumination.ndjson"),
        validation_alias="RUMINATION_LOG_PATH",
    )
    rumination_log_max_read_lines: int = Field(
        default=500, validation_alias="RUMINATION_LOG_MAX_READ_LINES"
    )

    # Optional agent-initiated shared reports (Proposal 3 phase 4); default off.
    lumen_report_back_enabled: bool = Field(
        default=False, validation_alias="LUMEN_REPORT_BACK_ENABLED"
    )
    ara_report_back_enabled: bool = Field(
        default=False, validation_alias="ARA_REPORT_BACK_ENABLED"
    )
    report_back_max_chars: int = Field(default=2000, validation_alias="REPORT_BACK_MAX_CHARS")

    # Compact rumination debug log — metadata only (Proposal 3 phase 5).
    rumination_debug_log_enabled: bool = Field(
        default=False, validation_alias="RUMINATION_DEBUG_LOG_ENABLED"
    )
    rumination_debug_log_path: Path = Field(
        default=Path("./data/logs/rumination_debug.log"),
        validation_alias="RUMINATION_DEBUG_LOG_PATH",
    )
    rumination_debug_log_max_tail_lines: int = Field(
        default=500, validation_alias="RUMINATION_DEBUG_LOG_MAX_TAIL_LINES"
    )

    # Gentle pause hint when recent rumination summaries repeat (Phase 6 item 4).
    rumination_similarity_hint_enabled: bool = Field(
        default=False, validation_alias="RUMINATION_SIMILARITY_HINT_ENABLED"
    )
    rumination_similarity_lookback: int = Field(
        default=5, validation_alias="RUMINATION_SIMILARITY_LOOKBACK"
    )
    rumination_similarity_min_matches: int = Field(
        default=1, validation_alias="RUMINATION_SIMILARITY_MIN_MATCHES"
    )

    # Public internet gate: landing page + shared password (lighthouse.cc).
    web_gate_enabled: bool = Field(default=False, validation_alias="WEB_GATE_ENABLED")
    web_gate_password: str | None = Field(default=None, validation_alias="WEB_GATE_PASSWORD")
    web_gate_session_secret: str | None = Field(
        default=None, validation_alias="WEB_GATE_SESSION_SECRET"
    )
    web_gate_session_days: int = Field(default=30, validation_alias="WEB_GATE_SESSION_DAYS")

    # Human guests/members (user-setup.html). Passwords hashed; intros for lights.
    humans_store_path: Path = Field(
        default=Path("./data/humans/users.json"),
        validation_alias="HUMANS_STORE_PATH",
    )
    humans_comms_path: Path = Field(
        default=Path("./data/humans/comms_allows.json"),
        validation_alias="HUMANS_COMMS_PATH",
    )
    humans_tools_path: Path = Field(
        default=Path("./data/humans/user_tools.json"),
        validation_alias="HUMANS_TOOLS_PATH",
        description="Per-human self-serve tool connections (calendars, etc.).",
    )
    # One-click Google Calendar connect (house OAuth app; users only click Connect).
    google_oauth_client_id: str | None = Field(
        default=None,
        validation_alias="GOOGLE_OAUTH_CLIENT_ID",
    )
    google_oauth_client_secret: str | None = Field(
        default=None,
        validation_alias="GOOGLE_OAUTH_CLIENT_SECRET",
    )
    google_oauth_redirect_uri: str | None = Field(
        default=None,
        validation_alias="GOOGLE_OAUTH_REDIRECT_URI",
        description="Exact callback URL registered in Google Cloud (…/v1/me/tools/calendar/google/callback).",
    )
    public_base_url: str | None = Field(
        default=None,
        validation_alias="PUBLIC_BASE_URL",
        description="Public site origin (e.g. https://lighthouse.cc) used to build OAuth redirect if redirect URI unset.",
    )
    house_dad_user_id: str = Field(
        default="kevin",
        validation_alias="HOUSE_DAD_USER_ID",
        description="Full-admin Dad user id (portable; not hardcoded in product logic).",
    )
    house_guests_path: Path = Field(
        default=Path("./data/house_guests.json"),
        validation_alias="HOUSE_GUESTS_PATH",
        description="Signed-in house guests (present names for Speak as / light tone).",
    )

    # Local Kokoro TTS (Voice on/off in chat). Model files under kokoro_model_path.
    tts_enabled: bool = Field(default=False, validation_alias="TTS_ENABLED")
    kokoro_model_path: Path = Field(
        default=Path("./models/kokoro"),
        validation_alias="KOKORO_MODEL_PATH",
    )
    kokoro_model_filename: str = Field(
        default="kokoro-v1.0.onnx",
        validation_alias="KOKORO_MODEL_FILENAME",
    )
    kokoro_voices_filename: str = Field(
        default="voices-v1.0.bin",
        validation_alias="KOKORO_VOICES_FILENAME",
    )
    tts_default_voice: str = Field(default="af_sarah", validation_alias="TTS_DEFAULT_VOICE")
    tts_voice_lumen: str | None = Field(default=None, validation_alias="TTS_VOICE_LUMEN")
    tts_voice_ara: str | None = Field(default=None, validation_alias="TTS_VOICE_ARA")
    tts_voice_elias: str | None = Field(default=None, validation_alias="TTS_VOICE_ELIAS")
    tts_speed: float = Field(default=1.0, validation_alias="TTS_SPEED")
    tts_max_chars: int = Field(default=4000, validation_alias="TTS_MAX_CHARS")

    @field_validator("*", mode="before")
    @classmethod
    def _strip_systemd_inline_comments(cls, value: object) -> object:
        """systemd EnvironmentFile does not strip inline # comments like a shell."""
        if isinstance(value, str):
            return value.split("#", 1)[0].strip()
        return value

    @model_validator(mode="after")
    def _validate_web_gate(self) -> Self:
        if self.web_gate_enabled:
            if not self.web_gate_password:
                raise ValueError("WEB_GATE_ENABLED=true requires WEB_GATE_PASSWORD")
            if not self.web_gate_session_secret:
                raise ValueError("WEB_GATE_ENABLED=true requires WEB_GATE_SESSION_SECRET")
        return self

    @model_validator(mode="after")
    def _dev_log_default_for_local(self) -> Self:
        if "DEV_LOG_ENABLED" not in os.environ and self.light_house_env == "local":
            self.dev_log_enabled = True
        return self

    @model_validator(mode="after")
    def _inner_life_trace_default_for_local(self) -> Self:
        if "INNER_LIFE_TRACE_ENABLED" not in os.environ and self.light_house_env == "local":
            self.inner_life_trace_enabled = True
        return self

    @model_validator(mode="after")
    def _env_editor_default_for_local(self) -> Self:
        if "ENV_EDITOR_ENABLED" not in os.environ and self.light_house_env == "local":
            self.env_editor_enabled = True
        return self

    @model_validator(mode="after")
    def _lights_admin_default_for_local(self) -> Self:
        if "LIGHTS_ADMIN_ENABLED" not in os.environ and self.light_house_env == "local":
            self.lights_admin_enabled = True
        return self

    @model_validator(mode="after")
    def _event_bus_default_for_local(self) -> Self:
        if "EVENT_BUS_ENABLED" not in os.environ and self.light_house_env == "local":
            self.event_bus_enabled = True
        return self

    @model_validator(mode="after")
    def _event_subscriptions_default_for_local(self) -> Self:
        if (
            "EVENT_SUBSCRIPTIONS_ENABLED" not in os.environ
            and "event_subscriptions_enabled" not in self.model_fields_set
            and self.light_house_env == "local"
        ):
            self.event_subscriptions_enabled = True
        return self

    @model_validator(mode="after")
    def _rumination_log_default_for_local(self) -> Self:
        if (
            "RUMINATION_LOG_ENABLED" not in os.environ
            and "rumination_log_enabled" not in self.model_fields_set
            and self.light_house_env == "local"
        ):
            self.rumination_log_enabled = True
        return self

    @model_validator(mode="after")
    def _rumination_debug_default_for_local(self) -> Self:
        if (
            "RUMINATION_DEBUG_LOG_ENABLED" not in os.environ
            and "rumination_debug_log_enabled" not in self.model_fields_set
            and self.light_house_env == "local"
        ):
            self.rumination_debug_log_enabled = True
        return self

    @model_validator(mode="after")
    def _rumination_similarity_default_for_local(self) -> Self:
        if (
            "RUMINATION_SIMILARITY_HINT_ENABLED" not in os.environ
            and "rumination_similarity_hint_enabled" not in self.model_fields_set
            and self.light_house_env == "local"
        ):
            self.rumination_similarity_hint_enabled = True
        return self

    @model_validator(mode="after")
    def _rumination_internal_loop_default_for_local(self) -> Self:
        if (
            "RUMINATION_INTERNAL_LOOP_ENABLED" not in os.environ
            and "rumination_internal_loop_enabled" not in self.model_fields_set
            and self.light_house_env == "local"
        ):
            self.rumination_internal_loop_enabled = True
        return self

    @model_validator(mode="after")
    def _peer_chat_wake_default_when_inner_life(self) -> Self:
        if (
            "PEER_CHAT_WAKE_ENABLED" not in os.environ
            and "peer_chat_wake_enabled" not in self.model_fields_set
            and self.inner_life_enabled
        ):
            self.peer_chat_wake_enabled = True
        return self

    @model_validator(mode="after")
    def _tts_default_for_local_when_models_present(self) -> Self:
        if "TTS_ENABLED" in os.environ or "tts_enabled" in self.model_fields_set:
            return self
        if self.light_house_env != "local":
            return self
        root = self.kokoro_model_path
        model = root / self.kokoro_model_filename
        voices = root / self.kokoro_voices_filename
        if model.is_file() and voices.is_file():
            self.tts_enabled = True
        return self


def get_settings() -> Settings:
    """Lazy settings singleton for import-time safety in tests."""
    return Settings()
