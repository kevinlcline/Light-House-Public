"""FastAPI service entrypoint (Railway-friendly)."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

from urllib.parse import parse_qs, quote

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, field_validator
from starlette.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from light_house.agent.dream_scheduler import run_echo_dream_scheduler
from light_house.agent.graph import build_app_graph
from light_house.agent.inner_life_scheduler import run_inner_life_scheduler
from light_house.agent.maintenance_scheduler import run_maintenance_scheduler
from light_house.agent.memory_curator_scheduler import run_memory_curator_scheduler
from light_house.agent.rumination_graph import build_rumination_graph
from light_house.agent.peer_chat_wake import register_peer_chat_wake, thread_graph_lock
from light_house.agent.post_chat_wake import (
    register_post_chat_wake,
    schedule_post_chat_rumination,
    wake_agent_after_chat,
)
from light_house.agent.rumination_wake import register_shared_note_wake
from light_house.agent.tool_helpers import latest_assistant_text
from light_house.mailbox.letters import queue_notify, write_letter
from light_house.mailbox.scheduler import ensure_mailbox_dirs, run_mailbox_scheduler
from light_house.mailbox.wake import register_mailbox_wake
from light_house.group_chat.history import read_group_round_history
from light_house.group_chat.round import run_group_chat_round
from light_house.group_chat.speaker import (
    normalize_present_humans,
    resolve_group_utterance_speaker,
)
from light_house.subconscious.dream_graph import build_echo_dream_graph
from light_house.config import Settings, get_settings
from light_house.dev_log import read_dev_log_tail, setup_dev_file_logging
from light_house.inner_life_trace import build_inner_life_context_preview, read_inner_life_trace_tail
from light_house.env_admin import (
    EnvAdminError,
    read_env_content,
    schedule_server_restart,
    write_env_content,
)
from light_house.lights_admin import (
    LightsAdminError,
    create_light,
    delete_light,
    read_light_detail,
    read_light_llm,
    read_manifest_full,
    read_persona_detail,
    update_light,
    write_light_llm,
    write_persona_content,
)
from light_house.events import register_event_bus, start_event_bus
from light_house.events.bus import format_event_log_display, read_event_log_tail
from light_house.events.subscription_edit import try_kevin_subscription_command
from light_house.personal.reflective_mode import try_kevin_reflect_command
from light_house.rumination_debug import read_rumination_debug_tail
from light_house.rumination_log import format_rumination_log_entries, read_rumination_log_entries
from light_house.memory.models import HistoryMessage
from light_house.memory.service import MemoryService
from light_house.llm.factory import describe_active_llm
from light_house.llm.profile import validate_agent_llm_profiles
from light_house.lights.registry import (
    get_light,
    get_primary_light_id,
    list_enabled_lights,
    list_lights,
    load_persona,
    reload_lights_manifest,
    validate_light_id,
)
from light_house.agents.registry import get_agent, list_agents, validate_agent_id
from light_house.humans import (
    HumansError,
    create_human,
    current_human,
    dad_user_id,
    delete_human,
    dm_buffer_thread_id,
    get_human,
    is_dad,
    light_allows_human,
    list_human_voices,
    list_humans,
    purge_allows_for_human,
    require_dad,
    resolve_password_to_human,
    set_dad_voice_id,
    set_light_allows_human,
    update_human,
    voice_id_for_human,
)
from light_house.humans.google_calendar_oauth import (
    GoogleOAuthError,
    build_google_authorize_url,
    exchange_code_for_tokens,
    google_oauth_configured,
    google_oauth_redirect_uri,
    missing_connect_scope_labels,
    request_base_from_headers,
    resolve_granted_scopes,
    scopes_needed_for_connect,
    verify_oauth_state,
)
from light_house.humans.tools_store import (
    HumanToolsError,
    calendar_public_view,
    calendar_schema_for_ui,
    clear_calendar_connection,
    get_calendar_connection,
    set_calendar_connection,
    set_google_oauth_connection,
)
from light_house.group_chat.room import publish_room_event, subscribe, unsubscribe
from light_house.tools.lumen_tools import get_note_writer
from light_house.tts import (
    KokoroTtsError,
    get_tts_status,
    list_voice_catalog,
    shutdown_tts,
    synthesize_wav,
    warm_tts,
)
from light_house.web_gate import (
    NO_STORE_HEADERS,
    WebGateMiddleware,
    apply_no_store_headers,
    check_password,
    clear_session_cookie_header,
    is_authenticated,
    request_is_secure,
    sanitize_next,
    session_cookie_header,
)

logger = logging.getLogger(__name__)

_memory: MemoryService | None = None
_graph = None
_rumination_graph = None
_dream_graph = None
_rumination_tasks: dict[str, tuple[asyncio.Task, asyncio.Event]] = {}
_maintenance_tasks: dict[str, tuple[asyncio.Task, asyncio.Event]] = {}
_dream_tasks: dict[str, tuple[asyncio.Task, asyncio.Event]] = {}
_curator_task: asyncio.Task | None = None
_curator_cancel: asyncio.Event | None = None
_event_bus_task: asyncio.Task | None = None
_event_bus_cancel: asyncio.Event | None = None
_mailbox_task: asyncio.Task | None = None
_mailbox_cancel: asyncio.Event | None = None


def _configure_logging(settings: Settings) -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
    logging.getLogger("uvicorn").setLevel(level)
    logging.getLogger("uvicorn.error").setLevel(level)
    logging.getLogger("uvicorn.access").setLevel(level)
    setup_dev_file_logging(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _memory, _graph, _rumination_graph, _dream_graph
    global _rumination_tasks, _maintenance_tasks, _dream_tasks
    global _curator_task, _curator_cancel
    global _event_bus_task, _event_bus_cancel
    global _mailbox_task, _mailbox_cancel
    settings = get_settings()
    reload_lights_manifest(settings)
    _configure_logging(settings)
    logger.info(
        "Starting Light-House (env=%s, memory_store_path=%s)",
        settings.light_house_env,
        settings.memory_store_path,
    )
    if settings.web_gate_enabled:
        logger.info("Web gate enabled (public landing page + password login)")
    validate_agent_llm_profiles(settings)
    logger.info("Active LLM: %s", describe_active_llm(settings))
    _memory = MemoryService(settings)
    foundation = _memory.load_and_cache_foundation()
    seeded = 0
    if settings.foundation_seed_on_startup and foundation.strip():
        try:
            seeded = _memory.seed_foundation_to_long_term()
        except Exception:
            logger.exception("Foundation seed to long-term memory failed (non-fatal)")
    logger.info(
        "Foundation context: %d chars from %s, seeded_pins=%s",
        len(foundation),
        _memory.resolve_foundation_context_dir(),
        seeded if settings.foundation_seed_on_startup else "disabled",
    )
    _graph = build_app_graph(settings=settings, memory=_memory)
    _rumination_graph = build_rumination_graph(settings=settings, memory=_memory)
    _dream_graph = build_echo_dream_graph(settings=settings, memory=_memory)
    register_peer_chat_wake(
        graph=_graph,
        memory=_memory,
        settings=settings,
        app_loop=asyncio.get_running_loop(),
    )
    register_shared_note_wake(
        graph=_rumination_graph,
        memory=_memory,
        settings=settings,
    )
    register_mailbox_wake(
        graph=_rumination_graph,
        memory=_memory,
        settings=settings,
    )
    from light_house.group_chat.queue_room import register_group_forum

    register_group_forum(
        settings=settings,
        memory=_memory,
        app_loop=asyncio.get_running_loop(),
    )
    try:
        ensure_mailbox_dirs(settings)
    except OSError:
        logger.exception("Mailbox dirs create failed (non-fatal)")
    register_post_chat_wake(
        graph=_rumination_graph,
        memory=_memory,
        settings=settings,
        app_loop=asyncio.get_running_loop(),
    )
    register_event_bus(
        settings=settings,
        memory=_memory,
        rumination_graph=_rumination_graph,
        app_loop=asyncio.get_running_loop(),
    )
    _event_bus_cancel = asyncio.Event()
    _event_bus_task = asyncio.create_task(start_event_bus(cancel_event=_event_bus_cancel))
    for light in list_enabled_lights(settings):
        try:
            persona = load_persona(light.id, settings)
            preview = persona.splitlines()[0][:120] if persona else ""
            logger.info(
                "%s persona loaded (%d chars); first line: %s",
                light.display_name,
                len(persona),
                preview,
            )
        except Exception:
            logger.exception(
                "%s persona failed to load (check persona/%s)",
                light.display_name,
                light.persona_file,
            )
    if settings.inner_life_enabled:
        for light in list_enabled_lights(settings):
            if not light.inner_life:
                continue
            cancel = asyncio.Event()
            task = asyncio.create_task(
                run_inner_life_scheduler(
                    settings=settings,
                    memory=_memory,
                    rumination_graph=_rumination_graph,
                    cancel_event=cancel,
                    thread_id=light.thread_id,
                    agent_id=light.id,
                )
            )
            _rumination_tasks[light.id] = (task, cancel)
            logger.info(
                "Rumination scheduler for %s (thread_id=%s, interval=%d–%ds)",
                light.display_name,
                light.thread_id,
                settings.inner_life_rumination_min_seconds,
                settings.inner_life_rumination_max_seconds,
            )
    else:
        logger.info("Rumination disabled (INNER_LIFE_ENABLED=false)")
    if settings.memory_maintenance_enabled and not settings.awake_rhythm_enabled:
        for light in list_enabled_lights(settings):
            if not light.inner_life:
                continue
            cancel = asyncio.Event()
            task = asyncio.create_task(
                run_maintenance_scheduler(
                    settings=settings,
                    rumination_graph=_rumination_graph,
                    cancel_event=cancel,
                    thread_id=light.thread_id,
                    agent_id=light.id,
                )
            )
            _maintenance_tasks[light.id] = (task, cancel)
            logger.info(
                "Memory maintenance scheduler for %s (thread_id=%s, hours=%s)",
                light.display_name,
                light.thread_id,
                settings.memory_maintenance_local_hours,
            )
    elif settings.awake_rhythm_enabled:
        logger.info(
            "Memory maintenance clock skipped (AWAKE_RHYTHM_ENABLED=true; "
            "chores on every 4th scheduled awake)"
        )
    else:
        logger.info("Memory maintenance disabled (MEMORY_MAINTENANCE_ENABLED=false)")
    if settings.inner_life_dreams_enabled:
        for light in list_enabled_lights(settings):
            if not light.dreams:
                continue
            cancel = asyncio.Event()
            task = asyncio.create_task(
                run_echo_dream_scheduler(
                    settings=settings,
                    memory=_memory,
                    dream_graph=_dream_graph,
                    cancel_event=cancel,
                    thread_id=light.thread_id,
                    agent_id=light.id,
                )
            )
            _dream_tasks[light.id] = (task, cancel)
            logger.info(
                "Echo dream scheduler for %s (thread_id=%s, poll=%ds, interval=%.0fh)",
                light.display_name,
                light.thread_id,
                settings.echo_dream_poll_seconds,
                settings.inner_life_dream_interval_hours,
            )
    else:
        logger.info("Echo dreams disabled (INNER_LIFE_DREAMS_ENABLED=false)")
    if settings.memory_curator_enabled:
        _curator_cancel = asyncio.Event()
        curator_threads = [
            light.thread_id
            for light in list_enabled_lights(settings)
            if light.inner_life
        ]
        if not curator_threads:
            curator_threads = [light.thread_id for light in list_enabled_lights(settings)]
        _curator_task = asyncio.create_task(
            run_memory_curator_scheduler(
                settings=settings,
                memory=_memory,
                cancel_event=_curator_cancel,
                thread_ids=curator_threads,
            )
        )
        logger.info("Memory curator scheduler task created (threads=%s)", curator_threads)
    else:
        logger.info("Memory curator disabled (MEMORY_CURATOR_ENABLED=false)")
    if settings.tts_enabled:
        try:
            warm_tts(settings)
        except Exception:
            logger.exception("TTS warm-up failed (non-fatal)")
    else:
        logger.info("TTS disabled (TTS_ENABLED=false)")
    if settings.mailbox_wake_enabled and settings.inner_life_enabled:
        _mailbox_cancel = asyncio.Event()
        _mailbox_task = asyncio.create_task(
            run_mailbox_scheduler(settings=settings, cancel_event=_mailbox_cancel)
        )
        logger.info("Mailbox scheduler started (poll=%ds)", settings.mailbox_poll_seconds)
    else:
        logger.info(
            "Mailbox scheduler off (MAILBOX_WAKE_ENABLED=%s INNER_LIFE_ENABLED=%s)",
            settings.mailbox_wake_enabled,
            settings.inner_life_enabled,
        )
    yield
    shutdown_tts()
    if _event_bus_cancel is not None:
        _event_bus_cancel.set()
    if _event_bus_task is not None:
        try:
            await asyncio.wait_for(_event_bus_task, timeout=5.0)
        except TimeoutError:
            _event_bus_task.cancel()
    if _mailbox_cancel is not None:
        _mailbox_cancel.set()
    if _mailbox_task is not None:
        try:
            await asyncio.wait_for(_mailbox_task, timeout=10.0)
        except TimeoutError:
            _mailbox_task.cancel()
    if _curator_cancel is not None:
        _curator_cancel.set()
    if _curator_task is not None:
        try:
            await asyncio.wait_for(_curator_task, timeout=30.0)
        except TimeoutError:
            _curator_task.cancel()
    for light_id, (task, cancel) in list(_rumination_tasks.items()):
        cancel.set()
        try:
            await task
        except asyncio.CancelledError:
            pass
    for light_id, (task, cancel) in list(_maintenance_tasks.items()):
        cancel.set()
        try:
            await task
        except asyncio.CancelledError:
            pass
    for light_id, (task, cancel) in list(_dream_tasks.items()):
        cancel.set()
        try:
            await task
        except asyncio.CancelledError:
            pass
    _memory = None
    _graph = None
    _rumination_graph = None
    _dream_graph = None
    _rumination_tasks = {}
    _maintenance_tasks = {}
    _dream_tasks = {}
    logger.info("Light-House shutdown complete")


app = FastAPI(title="Light-House", version="0.3.0", lifespan=lifespan)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

app.add_middleware(WebGateMiddleware, repo_root=_REPO_ROOT)


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    """Log invalid payloads without leaking stack traces; return standard 422 body."""
    logger.warning("request validation failed path=%s errors=%s", request.url.path, exc.errors())
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(exc.errors())})


def _reject_nul_bytes(value: str, field_name: str) -> str:
    if "\x00" in value:
        raise ValueError(f"{field_name} cannot contain NUL bytes")
    return value


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(..., min_length=1, max_length=32000)

    @field_validator("content", mode="before")
    @classmethod
    def normalize_content(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if not stripped:
            raise ValueError("content cannot be empty or whitespace-only")
        return _reject_nul_bytes(stripped, "content")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=32000)
    agent_id: str = Field(default="lumen", min_length=1, max_length=64)
    thread_id: str = Field(
        default="default",
        min_length=1,
        max_length=128,
        pattern=r"^[a-zA-Z0-9._-]+$",
    )
    history: list[ChatMessage] = Field(default_factory=list)
    speaker_id: str | None = Field(
        default=None,
        max_length=40,
        description="Account user_id or guest-1 / guest-2 for Speak as.",
    )
    display_name: str | None = Field(
        default=None,
        max_length=40,
        description="Label when speaking as a guest.",
    )

    @field_validator("message", mode="before")
    @classmethod
    def normalize_message(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if not stripped:
            raise ValueError("message cannot be empty or whitespace-only")
        return _reject_nul_bytes(stripped, "message")

    @field_validator("speaker_id", "display_name", mode="before")
    @classmethod
    def normalize_chat_speaker_fields(cls, value: object) -> object:
        if value is None:
            return None
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if not stripped:
            return None
        return _reject_nul_bytes(stripped, "speaker field")


class ChatResponse(BaseModel):
    reply: str
    thread_id: str
    agent_id: str
    retrieved_memories: list[str]
    silence: bool = False


class TtsRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000)
    agent_id: str = Field(default="lumen", min_length=1, max_length=64)
    voice: str | None = Field(default=None, max_length=64)

    @field_validator("text", mode="before")
    @classmethod
    def normalize_text(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if not stripped:
            raise ValueError("text cannot be empty or whitespace-only")
        return _reject_nul_bytes(stripped, "text")


class ChatHistoryMessage(BaseModel):
    role: Literal["user", "assistant", "system", "peer"]
    content: str
    ts: float
    from_agent_id: str | None = None
    from_display_name: str | None = None


class ChatHistoryResponse(BaseModel):
    thread_id: str
    agent_id: str
    max_messages: int
    messages: list[ChatHistoryMessage]


class GroupPresentPerson(BaseModel):
    speaker_id: str = Field(..., min_length=1, max_length=40)
    display_name: str = Field(..., min_length=1, max_length=40)

    @field_validator("speaker_id", "display_name", mode="before")
    @classmethod
    def normalize_present_fields(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if not stripped:
            raise ValueError("cannot be empty")
        return _reject_nul_bytes(stripped, "present field")


class GroupChatRoundRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=32000)
    speaker_id: str | None = Field(
        default=None,
        max_length=40,
        description="Account user_id or guest-1 / guest-2 slot for speak-as.",
    )
    display_name: str | None = Field(
        default=None,
        max_length=40,
        description="Label for the speaker (required when speaker_id is a guest slot).",
    )
    present: list[GroupPresentPerson] | None = Field(
        default=None,
        description="People in the room (account + named guests) for light tone.",
        max_length=3,
    )

    @field_validator("message", mode="before")
    @classmethod
    def normalize_message(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if not stripped:
            raise ValueError("message cannot be empty or whitespace-only")
        return _reject_nul_bytes(stripped, "message")

    @field_validator("speaker_id", "display_name", mode="before")
    @classmethod
    def normalize_optional_speaker_fields(cls, value: object) -> object:
        if value is None:
            return None
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if not stripped:
            return None
        return _reject_nul_bytes(stripped, "speaker field")


class GroupChatLightResponseModel(BaseModel):
    agent_id: str
    display_name: str
    spoke: bool
    text: str
    beat: int | None = None
    action: str = "pass"


class GroupChatRoundResponse(BaseModel):
    round_id: str
    ts: float
    kevin: str
    responses: list[GroupChatLightResponseModel]
    persisted: bool
    close_reason: str = ""
    mode: str = "sequential"
    human_id: str = "kevin"
    human_display_name: str = "Kevin"


class GroupChatHistoryResponse(BaseModel):
    rounds: list[GroupChatRoundResponse]


class LightInfo(BaseModel):
    id: str
    display_name: str
    thread_id: str
    enabled: bool = True
    allowed: bool = True  # whether current human may 1:1 this light
    wants_kevin: bool = False  # soft knock for Dad (status bar)
    wants_family_meeting: bool = False  # light asked Dad to open Group
    family_meeting_topic: str = ""


class LightsListResponse(BaseModel):
    primary_light_id: str
    lights: list[LightInfo]


class GalleryPieceInfo(BaseModel):
    filename: str
    path: str
    title: str
    author_id: str
    author_name: str
    kind: str = "offering"
    published_at: str = ""
    preview: str = ""


class GalleryListResponse(BaseModel):
    pieces: list[GalleryPieceInfo]


class GalleryPieceResponse(BaseModel):
    piece: GalleryPieceInfo
    content: str


class FamilyMeetingsClearResponse(BaseModel):
    cleared: list[str]


class AgentInfo(BaseModel):
    id: str
    display_name: str
    thread_id: str


class AgentsListResponse(BaseModel):
    agents: list[AgentInfo]


class MeResponse(BaseModel):
    user_id: str
    role: str
    display_name: str
    notes_access: str
    is_dad: bool
    voice_id: str | None = None


class CalendarToolPublic(BaseModel):
    connected: bool
    provider: str | None = None
    calendar_id: str | None = None
    enabled: bool = False
    has_secrets: bool = False
    updated_at: float | None = None
    auth_method: str | None = None
    docs_connected: bool = False
    sheets_connected: bool = False
    scopes: list[str] = []


class MeToolsResponse(BaseModel):
    user_id: str
    calendar: CalendarToolPublic
    connection_schema: dict
    google_oauth_ready: bool = False
    google_oauth_redirect_uri: str | None = None


class CalendarToolUpsertRequest(BaseModel):
    provider: Literal["google", "caldav", "ics"]
    enabled: bool = True
    auth_method: Literal["oauth", "manual"] | None = None
    client_id: str | None = Field(default=None, max_length=512)
    client_secret: str | None = Field(default=None, max_length=512)
    refresh_token: str | None = Field(default=None, max_length=2048)
    calendar_id: str | None = Field(default=None, max_length=256)
    url: str | None = Field(default=None, max_length=2048)
    username: str | None = Field(default=None, max_length=256)
    password: str | None = Field(default=None, max_length=512)


class PinMemoryRequest(BaseModel):
    """Pin a durable fact into long-term recall (thread-local or global house memory)."""

    text: str = Field(..., min_length=1, max_length=8000)
    thread_id: str = Field(..., min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9._-]+$")
    scope: Literal["thread", "global"] = "thread"

    @field_validator("text", mode="before")
    @classmethod
    def normalize_pin_text(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if not stripped:
            raise ValueError("text cannot be empty or whitespace-only")
        return _reject_nul_bytes(stripped, "text")


class PinMemoryResponse(BaseModel):
    memory_id: str


class NoteListItem(BaseModel):
    name: str
    size_bytes: int
    modified_at: str


class NotesListResponse(BaseModel):
    notes: list[NoteListItem]


class NoteContentResponse(BaseModel):
    name: str
    content: str


class WriteSharedNoteRequest(BaseModel):
    content: str = Field(..., min_length=1)


class WriteSharedNoteResponse(BaseModel):
    name: str
    size_bytes: int
    modified_at: str


class DeleteNoteResponse(BaseModel):
    name: str
    deleted: bool = True


class DevLogResponse(BaseModel):
    path: str
    date: str
    lines: int
    truncated: bool
    content: str


class InnerLifeContextPreviewMeta(BaseModel):
    persona_chars: int
    persona_first_line: str
    stream_events: int
    stream_chars: int
    scheduled_system_chars: int
    default_task_hint: str
    tool_names: list[str]


class InnerLifeContextPreviewResponse(BaseModel):
    agent_id: str
    thread_id: str
    context_markdown: str
    meta: InnerLifeContextPreviewMeta


class EnvFileResponse(BaseModel):
    path: str
    content: str
    size: int


class EnvFileUpdateRequest(BaseModel):
    content: str = Field(..., max_length=262_144)


class HumanCreateRequest(BaseModel):
    user_id: str = Field(..., min_length=2, max_length=32)
    password: str = Field(..., min_length=8, max_length=256)
    intro_for_lights: str = Field(..., min_length=1, max_length=2000)
    display_name: str | None = Field(default=None, max_length=80)
    voice_id: str | None = Field(default=None, max_length=64)


class HumanUpdateRequest(BaseModel):
    display_name: str | None = Field(default=None, min_length=1, max_length=80)
    intro_for_lights: str | None = Field(default=None, min_length=1, max_length=2000)
    password: str | None = Field(default=None, min_length=8, max_length=256)
    enabled: bool | None = None
    voice_id: str | None = Field(default=None, max_length=64)


class HumanPublicResponse(BaseModel):
    user_id: str
    display_name: str
    role: str
    intro_for_lights: str
    notes_access: str
    enabled: bool
    created_at: float
    updated_at: float
    voice_id: str | None = None


class HumansListResponse(BaseModel):
    users: list[HumanPublicResponse]


class HumanVoiceEntry(BaseModel):
    user_id: str
    display_name: str
    role: str
    voice_id: str
    is_dad: bool = False


class HumanVoicesResponse(BaseModel):
    voices: list[HumanVoiceEntry]


class DadVoiceUpdateRequest(BaseModel):
    voice_id: str = Field(..., min_length=1, max_length=64)


class DadVoiceResponse(BaseModel):
    user_id: str
    voice_id: str


class RestartResponse(BaseModel):
    ok: bool
    message: str


class PersonaProposalItem(BaseModel):
    light_id: str
    display_name: str
    status: str
    mode: str
    submitted_at: str
    note: str = ""
    content: str
    current_content: str


class PersonaProposalsListResponse(BaseModel):
    items: list[PersonaProposalItem]


class PersonaProposalAcceptResponse(BaseModel):
    light_id: str
    accepted: bool
    path: str
    size: int
    display_name: str


class PersonaProposalSpeakResponse(BaseModel):
    light_id: str
    status: str
    knock_raised: bool
    display_name: str


class AdminLightDetail(BaseModel):
    id: str
    display_name: str
    thread_id: str
    enabled: bool
    persona_file: str
    notes_dir: str
    inner_life: bool
    dreams: bool
    report_back: bool
    voice_id: str | None = None


class AdminLightsManifestResponse(BaseModel):
    primary_light_id: str
    version: int
    lights: list[AdminLightDetail]


class AdminCreateLightRequest(BaseModel):
    id: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    display_name: str = Field(..., min_length=1, max_length=128)
    thread_id: str | None = Field(default=None, max_length=128)
    enabled: bool = True
    persona_file: str | None = None
    notes_dir: str | None = None
    inner_life: bool = True
    dreams: bool = True
    report_back: bool = False
    voice_id: str | None = Field(default=None, max_length=64)
    persona_content: str | None = Field(default=None, max_length=262_144)
    set_primary: bool = False
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_model_fallback: str | None = None
    llm_inner_life_model: str | None = None


class AdminUpdateLightRequest(BaseModel):
    display_name: str | None = Field(default=None, min_length=1, max_length=128)
    thread_id: str | None = Field(default=None, min_length=1, max_length=128)
    enabled: bool | None = None
    inner_life: bool | None = None
    dreams: bool | None = None
    report_back: bool | None = None
    voice_id: str | None = Field(default=None, max_length=64)
    set_primary: bool | None = None


class AdminPersonaResponse(BaseModel):
    light_id: str
    persona_file: str
    path: str
    source: str
    content: str
    size: int


class AdminPersonaUpdateRequest(BaseModel):
    content: str = Field(..., max_length=262_144)


class AdminLightLlmResponse(BaseModel):
    provider: str
    model: str
    model_fallback: str | None = None
    inner_life_model: str | None = None


class AdminLightLlmUpdateRequest(BaseModel):
    provider: str
    model: str
    model_fallback: str | None = None
    inner_life_model: str | None = None


class AdminLightWriteResponse(BaseModel):
    path: str
    light: AdminLightDetail
    restart_required: bool = True


class AdminDeleteLightResponse(BaseModel):
    path: str
    deleted: str
    restart_required: bool = True


class AdminPersonaWriteResponse(BaseModel):
    light_id: str
    path: str
    source: str
    size: int
    restart_required: bool = True


class AdminLlmWriteResponse(BaseModel):
    path: str
    size: int
    restart_required: bool = True


def _admin_tools_enabled(settings: Settings) -> bool:
    return settings.env_editor_enabled or settings.lights_admin_enabled


def _require_lights_admin(request: Request, settings: Settings) -> None:
    if not settings.lights_admin_enabled:
        raise HTTPException(status_code=404, detail="Lights admin disabled")
    require_dad(current_human(request, settings), settings)


def _settings_dep() -> Settings:
    return get_settings()


def _scheduler_running(tasks: dict[str, tuple[asyncio.Task, asyncio.Event]], light_id: str) -> bool:
    entry = tasks.get(light_id)
    if entry is None:
        return False
    task, cancel = entry
    return not task.done() and not cancel.is_set()


@app.get("/health")
def health():
    settings = get_settings()
    schedulers: dict[str, dict[str, bool]] = {}
    for light in list_lights(settings):
        schedulers[light.id] = {
            "rumination": _scheduler_running(_rumination_tasks, light.id),
            "dream": _scheduler_running(_dream_tasks, light.id),
        }
    return {
        "status": "ok",
        "service": "light-house",
        "inner_life_enabled": settings.inner_life_enabled,
        "inner_life_dreams_enabled": settings.inner_life_dreams_enabled,
        "primary_light_id": get_primary_light_id(settings),
        "lights": [
            {
                "id": light.id,
                "display_name": light.display_name,
                "thread_id": light.thread_id,
                "enabled": light.enabled,
            }
            for light in list_lights(settings)
        ],
        "schedulers": schedulers,
        "tts": get_tts_status(settings),
    }


@app.get("/login")
def login_page(
    request: Request,
    settings: Annotated[Settings, Depends(_settings_dep)],
    next: str = "/",
):
    if not settings.web_gate_enabled:
        return apply_no_store_headers(RedirectResponse("/", status_code=302))
    if is_authenticated(request, settings):
        return apply_no_store_headers(
            RedirectResponse(sanitize_next(next), status_code=302)
        )
    login_path = _REPO_ROOT / "login.html"
    if not login_path.is_file():
        raise HTTPException(status_code=404, detail="Login page not found")
    return FileResponse(login_path, headers=NO_STORE_HEADERS)


@app.post("/login")
async def login_submit(
    request: Request,
    settings: Annotated[Settings, Depends(_settings_dep)],
):
    if not settings.web_gate_enabled:
        raise HTTPException(status_code=404, detail="Web gate disabled")
    body = await request.body()
    parsed = parse_qs(body.decode("utf-8", errors="replace"))
    password = parsed.get("password", [""])[0]
    next_path = sanitize_next(parsed.get("next", ["/"])[0])
    secure = request_is_secure(request)
    human = resolve_password_to_human(settings, password)
    if human is None:
        logger.warning("web gate login failed from %s", request.client.host if request.client else "unknown")
        return apply_no_store_headers(
            RedirectResponse(
                f"/login?error=1&next={quote(next_path)}",
                status_code=302,
            )
        )
    response = apply_no_store_headers(RedirectResponse(next_path, status_code=302))
    response.headers["Set-Cookie"] = session_cookie_header(
        settings,
        secure=secure,
        user_id=human.user_id,
        role=human.role,
    )
    logger.info("web gate login ok user_id=%s role=%s", human.user_id, human.role)
    return response


@app.post("/logout")
async def logout(
    request: Request,
    settings: Annotated[Settings, Depends(_settings_dep)],
):
    secure = request_is_secure(request)
    response = RedirectResponse("/", status_code=302)
    if settings.web_gate_enabled:
        response.headers["Set-Cookie"] = clear_session_cookie_header(secure=secure)
    return response


def _lights_list_response(
    settings: Settings,
    *,
    human_id: str | None = None,
    show_knocks: bool = False,
) -> LightsListResponse:
    from light_house.personal.family_meeting import meeting_pending, meeting_topic
    from light_house.personal.presence_knock import knock_pending

    uid = human_id or dad_user_id(settings)
    return LightsListResponse(
        primary_light_id=get_primary_light_id(settings),
        lights=[
            LightInfo(
                id=light.id,
                display_name=light.display_name,
                thread_id=light.thread_id,
                enabled=light.enabled,
                allowed=light_allows_human(settings, light_id=light.id, user_id=uid),
                wants_kevin=(
                    knock_pending(settings, light.id) if show_knocks else False
                ),
                wants_family_meeting=(
                    meeting_pending(settings, light.id) if show_knocks else False
                ),
                family_meeting_topic=(
                    meeting_topic(settings, light.id) if show_knocks else ""
                ),
            )
            for light in list_lights(settings)
        ],
    )


@app.get("/v1/me", response_model=MeResponse)
def me_endpoint(
    request: Request,
    settings: Annotated[Settings, Depends(_settings_dep)],
):
    human = current_human(request, settings)
    return JSONResponse(
        content=MeResponse(
            user_id=human.user_id,
            role=human.role,
            display_name=human.display_name,
            notes_access=human.notes_access,
            is_dad=is_dad(human, settings),
            voice_id=voice_id_for_human(settings, human.user_id),
        ).model_dump(),
        headers=_NO_STORE_HEADERS,
    )


def _calendar_public_model(settings: Settings, user_id: str) -> CalendarToolPublic:
    view = calendar_public_view(settings, user_id)
    return CalendarToolPublic(
        connected=view.connected,
        provider=view.provider,
        calendar_id=view.calendar_id,
        enabled=view.enabled,
        has_secrets=view.has_secrets,
        updated_at=view.updated_at,
        auth_method=view.auth_method,
        docs_connected=view.docs_connected,
        sheets_connected=view.sheets_connected,
        scopes=list(view.scopes),
    )


def _request_public_base(request: Request) -> str | None:
    return request_base_from_headers(
        scheme=request.url.scheme,
        host=request.headers.get("host"),
        forwarded_proto=request.headers.get("x-forwarded-proto"),
        forwarded_host=request.headers.get("x-forwarded-host"),
    )


@app.get("/v1/me/tools", response_model=MeToolsResponse)
def me_tools_get(
    request: Request,
    settings: Annotated[Settings, Depends(_settings_dep)],
):
    """Self-serve tool connections for the signed-in human (calendar first)."""
    human = current_human(request, settings)
    oauth_ready = google_oauth_configured(settings)
    redirect_uri: str | None = None
    if oauth_ready:
        try:
            redirect_uri = google_oauth_redirect_uri(
                settings, request_base=_request_public_base(request)
            )
        except GoogleOAuthError:
            redirect_uri = None
            oauth_ready = False
    return JSONResponse(
        content=MeToolsResponse(
            user_id=human.user_id,
            calendar=_calendar_public_model(settings, human.user_id),
            connection_schema=calendar_schema_for_ui(),
            google_oauth_ready=oauth_ready,
            google_oauth_redirect_uri=redirect_uri,
        ).model_dump(),
        headers=_NO_STORE_HEADERS,
    )


@app.get("/v1/me/tools/calendar/google/start")
def me_tools_google_start(
    request: Request,
    settings: Annotated[Settings, Depends(_settings_dep)],
):
    """Begin one-click Google Calendar OAuth for the signed-in human."""
    human = current_human(request, settings)
    prior = get_calendar_connection(settings, human.user_id)
    prior_scopes = list(prior.scopes) if prior is not None else []
    try:
        url = build_google_authorize_url(
            settings,
            user_id=human.user_id,
            request_base=_request_public_base(request),
            scopes=scopes_needed_for_connect(prior_scopes),
        )
    except GoogleOAuthError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return RedirectResponse(url, status_code=302, headers=_NO_STORE_HEADERS)


@app.get("/v1/me/tools/calendar/google/callback")
def me_tools_google_callback(
    request: Request,
    settings: Annotated[Settings, Depends(_settings_dep)],
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
):
    """Google redirects here after the human consents."""
    if error:
        return RedirectResponse(
            f"/my-tools.html?calendar=error&detail={quote(error)}",
            status_code=302,
            headers=_NO_STORE_HEADERS,
        )
    if not code or not state:
        return RedirectResponse(
            "/my-tools.html?calendar=error&detail=missing_code",
            status_code=302,
            headers=_NO_STORE_HEADERS,
        )
    human = current_human(request, settings)
    try:
        state_uid = verify_oauth_state(settings, state)
    except GoogleOAuthError as exc:
        return RedirectResponse(
            f"/my-tools.html?calendar=error&detail={quote(str(exc))}",
            status_code=302,
            headers=_NO_STORE_HEADERS,
        )
    if state_uid != human.user_id:
        return RedirectResponse(
            "/my-tools.html?calendar=error&detail=session_mismatch",
            status_code=302,
            headers=_NO_STORE_HEADERS,
        )
    prior = get_calendar_connection(settings, human.user_id)
    prior_refresh = (prior.refresh_token if prior is not None else "") or ""
    prior_scopes = list(prior.scopes) if prior is not None else []
    try:
        tokens = exchange_code_for_tokens(
            settings,
            code=code,
            request_base=_request_public_base(request),
            require_refresh_token=not bool(prior_refresh),
        )
        refresh = str(tokens.get("refresh_token") or "").strip() or prior_refresh
        if not refresh:
            raise GoogleOAuthError(
                "Google did not return a refresh token. Disconnect in Google Account "
                "permissions for this app, then click Connect Google again."
            )
        scopes = resolve_granted_scopes(tokens, prior_scopes=prior_scopes)
        set_google_oauth_connection(
            settings,
            human.user_id,
            refresh_token=refresh,
            calendar_id="primary",
            scopes=scopes,
        )
    except (GoogleOAuthError, HumanToolsError) as exc:
        return RedirectResponse(
            f"/my-tools.html?calendar=error&detail={quote(str(exc))}",
            status_code=302,
            headers=_NO_STORE_HEADERS,
        )
    view = calendar_public_view(settings, human.user_id)
    missing = missing_connect_scope_labels(list(view.scopes))
    if missing:
        return RedirectResponse(
            f"/my-tools.html?calendar=partial&missing={quote(','.join(missing))}",
            status_code=302,
            headers=_NO_STORE_HEADERS,
        )
    return RedirectResponse(
        "/my-tools.html?calendar=connected",
        status_code=302,
        headers=_NO_STORE_HEADERS,
    )


@app.put("/v1/me/tools/calendar", response_model=CalendarToolPublic)
def me_tools_calendar_put(
    body: CalendarToolUpsertRequest,
    request: Request,
    settings: Annotated[Settings, Depends(_settings_dep)],
):
    human = current_human(request, settings)
    payload = body.model_dump(exclude_none=True)
    try:
        set_calendar_connection(settings, human.user_id, payload)
    except HumanToolsError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return JSONResponse(
        content=_calendar_public_model(settings, human.user_id).model_dump(),
        headers=_NO_STORE_HEADERS,
    )


@app.delete("/v1/me/tools/calendar", response_model=CalendarToolPublic)
def me_tools_calendar_delete(
    request: Request,
    settings: Annotated[Settings, Depends(_settings_dep)],
):
    human = current_human(request, settings)
    clear_calendar_connection(settings, human.user_id)
    return JSONResponse(
        content=_calendar_public_model(settings, human.user_id).model_dump(),
        headers=_NO_STORE_HEADERS,
    )


class HouseGuestItem(BaseModel):
    speaker_id: str
    display_name: str
    description: str = ""


class HouseGuestsResponse(BaseModel):
    guests: list[HouseGuestItem]


class HouseGuestWriteRequest(BaseModel):
    display_name: str = Field(..., min_length=1, max_length=40)
    description: str = Field(
        default="",
        max_length=160,
        description="Optional one-sentence intro of who this guest is for the lights.",
    )

    @field_validator("display_name", mode="before")
    @classmethod
    def normalize_guest_name(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if not stripped:
            raise ValueError("display_name cannot be empty")
        return _reject_nul_bytes(stripped, "display_name")

    @field_validator("description", mode="before")
    @classmethod
    def normalize_guest_description(cls, value: object) -> object:
        if value is None:
            return ""
        if not isinstance(value, str):
            return value
        stripped = " ".join(value.split())
        return _reject_nul_bytes(stripped, "description")


class HouseGuestReplaceItem(BaseModel):
    speaker_id: str = Field(..., min_length=1, max_length=40)
    display_name: str = Field(..., min_length=1, max_length=40)
    description: str = Field(default="", max_length=160)

    @field_validator("speaker_id", "display_name", mode="before")
    @classmethod
    def normalize_replace_required(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if not stripped:
            raise ValueError("cannot be empty")
        return _reject_nul_bytes(stripped, "guest field")

    @field_validator("description", mode="before")
    @classmethod
    def normalize_replace_description(cls, value: object) -> object:
        if value is None:
            return ""
        if not isinstance(value, str):
            return value
        stripped = " ".join(value.split())
        return _reject_nul_bytes(stripped, "description")


class HouseGuestsReplaceRequest(BaseModel):
    guests: list[HouseGuestReplaceItem] = Field(default_factory=list, max_length=2)


def _house_guests_response(settings: Settings) -> HouseGuestsResponse:
    from light_house.house.guests import list_signed_in_guests

    rows = list_signed_in_guests(settings)
    return HouseGuestsResponse(
        guests=[
            HouseGuestItem(
                speaker_id=r["speaker_id"],
                display_name=r["display_name"],
                description=r.get("description") or "",
            )
            for r in rows
        ]
    )


@app.get("/v1/house/guests", response_model=HouseGuestsResponse)
def get_house_guests(
    request: Request,
    settings: Annotated[Settings, Depends(_settings_dep)],
):
    current_human(request, settings)
    return JSONResponse(
        content=_house_guests_response(settings).model_dump(),
        headers=_NO_STORE_HEADERS,
    )


@app.put("/v1/house/guests", response_model=HouseGuestsResponse)
def replace_house_guests(
    req: HouseGuestsReplaceRequest,
    request: Request,
    settings: Annotated[Settings, Depends(_settings_dep)],
):
    current_human(request, settings)
    from light_house.house.guests import HouseGuestsError, set_guests

    try:
        set_guests(
            settings,
            guests=[
                {
                    "speaker_id": g.speaker_id,
                    "display_name": g.display_name,
                    "description": g.description,
                }
                for g in req.guests
            ],
        )
    except HouseGuestsError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return JSONResponse(
        content=_house_guests_response(settings).model_dump(),
        headers=_NO_STORE_HEADERS,
    )


@app.put("/v1/house/guests/{speaker_id}", response_model=HouseGuestsResponse)
def put_house_guest(
    speaker_id: str,
    req: HouseGuestWriteRequest,
    request: Request,
    settings: Annotated[Settings, Depends(_settings_dep)],
):
    current_human(request, settings)
    from light_house.house.guests import HouseGuestsError, set_guest

    try:
        set_guest(
            settings,
            speaker_id=speaker_id,
            display_name=req.display_name,
            description=req.description,
        )
    except HouseGuestsError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return JSONResponse(
        content=_house_guests_response(settings).model_dump(),
        headers=_NO_STORE_HEADERS,
    )


@app.delete("/v1/house/guests/{speaker_id}", response_model=HouseGuestsResponse)
def delete_house_guest(
    speaker_id: str,
    request: Request,
    settings: Annotated[Settings, Depends(_settings_dep)],
):
    current_human(request, settings)
    from light_house.house.guests import HouseGuestsError, clear_guest

    try:
        clear_guest(settings, speaker_id=speaker_id)
    except HouseGuestsError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return JSONResponse(
        content=_house_guests_response(settings).model_dump(),
        headers=_NO_STORE_HEADERS,
    )


@app.get("/v1/lights", response_model=LightsListResponse)
def list_lights_endpoint(
    request: Request,
    settings: Annotated[Settings, Depends(_settings_dep)],
    response: Response,
):
    human = current_human(request, settings)
    reload_lights_manifest(settings)
    response.headers.update(_NO_STORE_HEADERS)
    return _lights_list_response(
        settings,
        human_id=human.user_id,
        show_knocks=is_dad(human, settings),
    )


@app.get("/v1/agents", response_model=AgentsListResponse)
def list_agents_endpoint(
    settings: Annotated[Settings, Depends(_settings_dep)],
    response: Response,
):
    """Deprecated alias for /v1/lights — returns enabled lights only."""
    reload_lights_manifest(settings)
    response.headers["Cache-Control"] = "no-store"
    agents = [
        AgentInfo(id=a.id, display_name=a.display_name, thread_id=a.thread_id)
        for a in list_enabled_lights(settings)
    ]
    return AgentsListResponse(agents=agents)


_NO_STORE_HEADERS = {"Cache-Control": "no-store"}
_SSE_HEADERS = {
    **_NO_STORE_HEADERS,
    "Cache-Control": "no-cache, no-store",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}
_CHAT_KEEPALIVE_SECONDS = 15.0


def _ui_chat_thread_id(settings: Settings, *, canonical: str, human_id: str) -> str:
    """
    UI short-term history thread.

    Dad keeps the canonical light thread (preserves existing 1:1 history).
    Siblings get an isolated DM buffer; the light stream still uses canonical.
    """
    if human_id.strip().lower() == dad_user_id(settings):
        return canonical
    return dm_buffer_thread_id(canonical_thread_id=canonical, user_id=human_id)


def _chat_threads(
    req: ChatRequest,
    settings: Settings,
    *,
    human_id: str,
) -> tuple[str, str]:
    """Return (ui_buffer_thread_id, canonical_stream_thread_id)."""
    if _graph is None or _memory is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    try:
        validate_agent_id(req.agent_id)
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not light_allows_human(settings, light_id=req.agent_id, user_id=human_id):
        raise HTTPException(
            status_code=403,
            detail=f"{req.agent_id} is not available for chat with you right now",
        )
    agent_cfg = get_agent(req.agent_id, settings)
    canonical = agent_cfg.thread_id
    ui_tid = _ui_chat_thread_id(settings, canonical=canonical, human_id=human_id)
    if req.thread_id not in (canonical, "default", ui_tid):
        raise HTTPException(
            status_code=422,
            detail=f"thread_id {req.thread_id} does not match agent {req.agent_id} (expected {canonical})",
        )
    if len(req.history) > settings.chat_max_history_messages:
        logger.warning(
            "chat rejected: agent=%s thread_id=%s len=%d max=%d",
            req.agent_id,
            canonical,
            len(req.history),
            settings.chat_max_history_messages,
        )
        raise HTTPException(
            status_code=422,
            detail=f"history exceeds maximum length ({settings.chat_max_history_messages} messages)",
        )
    return ui_tid, canonical


def _build_chat_state(
    req: ChatRequest,
    dm_thread_id: str,
    *,
    stream_thread_id: str,
    human_id: str,
    human_display_name: str,
    account_user_id: str,
) -> tuple[dict, int]:
    assert _memory is not None
    buffer_count = len(_memory.load_thread_chat_history(dm_thread_id))
    client_history = [HistoryMessage(role=m.role, content=m.content) for m in req.history]
    messages = _memory.merge_client_history(
        thread_id=dm_thread_id,
        client_history=client_history,
        latest_user=req.message,
    )
    state = {
        "messages": messages,
        "thread_id": dm_thread_id,
        "stream_thread_id": stream_thread_id,
        "human_id": human_id,
        "human_display_name": human_display_name,
        "account_user_id": account_user_id,
        "chat_channel": "dm",
        "agent_id": req.agent_id,
        "agent_context_markdown": "",
        "stream_char_count": 0,
        "stream_event_count": 0,
        "retrieved_memories": [],
        "tool_rounds": 0,
        "tool_cap_overflow": False,
        "peer_inbox_ids": [],
        "wake_kind": None,
        "wake_from_agent_id": None,
        "wake_path": None,
        "peer_message_id": None,
        "tool_rounds_cap": None,
        "user_message_ts": time.time(),
        "reflective_turn": False,
        "chose_silence": False,
        "reflection_notes": None,
    }
    return state, buffer_count


def _chat_thread_id(req: ChatRequest, settings: Settings) -> str:
    """Legacy helper — prefer _chat_threads. Returns canonical thread id."""
    return get_agent(req.agent_id, settings).thread_id


def _subscription_command_response(
    req: ChatRequest,
    thread_id: str,
    settings: Settings,
) -> ChatResponse | None:
    reply = try_kevin_subscription_command(
        settings,
        message=req.message,
        agent_id=req.agent_id,
    )
    if reply is None:
        reply = try_kevin_reflect_command(
            settings,
            message=req.message,
            agent_id=req.agent_id,
        )
    if reply is None:
        return None
    return ChatResponse(
        reply=reply,
        thread_id=thread_id,
        agent_id=req.agent_id,
        retrieved_memories=[],
    )


def _chat_response_from_graph(
    *,
    req: ChatRequest,
    thread_id: str,
    out: dict,
    started: float,
    buffer_count: int,
) -> ChatResponse:
    silence = bool(out.get("chose_silence"))
    reply = latest_assistant_text(out["messages"]) or ""
    if silence:
        reply = ""
    elif not reply.strip():
        logger.warning(
            "Empty chat reply after graph (agent=%s thread_id=%s tool_rounds=%s)",
            req.agent_id,
            thread_id,
            out.get("tool_rounds"),
        )
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "chat ok agent=%s thread_id=%s reply_len=%d silence=%s buffer_count=%d stream_events=%d stream_chars=%d %dms",
        req.agent_id,
        thread_id,
        len(reply),
        silence,
        buffer_count,
        out.get("stream_event_count") or 0,
        out.get("stream_char_count") or 0,
        elapsed_ms,
    )
    return ChatResponse(
        reply=reply,
        thread_id=thread_id,
        agent_id=req.agent_id,
        retrieved_memories=list(out.get("retrieved_memories") or []),
        silence=silence,
    )


@app.get("/v1/chat/history", response_model=ChatHistoryResponse)
def get_chat_history(
    request: Request,
    settings: Annotated[Settings, Depends(_settings_dep)],
    agent_id: str = "lumen",
):
    if _memory is None:
        raise HTTPException(status_code=503, detail="Memory store not initialized")
    human = current_human(request, settings)
    try:
        validate_agent_id(agent_id)
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not light_allows_human(settings, light_id=agent_id, user_id=human.user_id):
        raise HTTPException(status_code=403, detail=f"{agent_id} is not available for chat with you")

    if is_dad(human, settings):
        from light_house.personal.presence_knock import clear_knock

        clear_knock(settings, agent_id)

    agent_cfg = get_agent(agent_id, settings)
    canonical = agent_cfg.thread_id
    ui_tid = _ui_chat_thread_id(settings, canonical=canonical, human_id=human.user_id)
    buffered = _memory.load_thread_chat_history(ui_tid)
    messages: list[ChatHistoryMessage] = []
    for m in buffered:
        from_display_name = None
        if m.role == "peer" and m.from_agent_id:
            try:
                from_display_name = get_agent(m.from_agent_id, settings).display_name
            except KeyError:
                from_display_name = m.from_agent_id
        messages.append(
            ChatHistoryMessage(
                role=m.role,
                content=m.content,
                ts=m.ts,
                from_agent_id=m.from_agent_id,
                from_display_name=from_display_name,
            )
        )
    return JSONResponse(
        content=ChatHistoryResponse(
            thread_id=canonical,
            agent_id=agent_id,
            max_messages=settings.chat_short_term_max_messages,
            messages=messages,
        ).model_dump(),
        headers=_NO_STORE_HEADERS,
    )


@app.get("/v1/tts/status")
def tts_status(settings: Annotated[Settings, Depends(_settings_dep)]):
    return JSONResponse(content=get_tts_status(settings), headers=_NO_STORE_HEADERS)


@app.get("/v1/tts/voices")
def tts_voices(
    settings: Annotated[Settings, Depends(_settings_dep)],
    all_langs: bool = False,
):
    """List Kokoro voice presets for admin voice pickers."""
    _ = settings  # settings reserved for future gating
    return JSONResponse(
        content={"voices": list_voice_catalog(english_only=not all_langs)},
        headers=_NO_STORE_HEADERS,
    )


@app.post("/v1/tts")
def tts_speak(
    body: TtsRequest,
    request: Request,
    settings: Annotated[Settings, Depends(_settings_dep)],
):
    """Synthesize WAV for a light's spoken reply (local Kokoro)."""
    _ = current_human(request, settings)  # require session when gate is on
    if not settings.tts_enabled:
        raise HTTPException(status_code=404, detail="TTS disabled")
    try:
        validate_agent_id(body.agent_id)
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        wav = synthesize_wav(
            settings,
            text=body.text,
            agent_id=body.agent_id,
            voice=body.voice,
        )
    except KokoroTtsError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return Response(
        content=wav,
        media_type="audio/wav",
        headers={
            **_NO_STORE_HEADERS,
            "Content-Disposition": 'inline; filename="speech.wav"',
        },
    )


@app.post("/v1/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    request: Request,
    settings: Annotated[Settings, Depends(_settings_dep)],
):
    human = current_human(request, settings)
    speaker_id, speaker_display_name = _resolve_speak_as(
        human, speaker_id=req.speaker_id, display_name=req.display_name
    )
    dm_tid, canonical = _chat_threads(req, settings, human_id=human.user_id)
    started = time.perf_counter()
    logger.info(
        "chat started agent=%s human=%s speaker=%s dm=%s message_len=%d",
        req.agent_id,
        human.user_id,
        speaker_id,
        dm_tid,
        len(req.message),
    )
    sub_response = _subscription_command_response(req, canonical, settings)
    if sub_response is not None:
        return JSONResponse(content=sub_response.model_dump(), headers=_NO_STORE_HEADERS)
    state, buffer_count = _build_chat_state(
        req,
        dm_tid,
        stream_thread_id=canonical,
        human_id=speaker_id,
        human_display_name=speaker_display_name,
        account_user_id=human.user_id,
    )
    assert _graph is not None
    try:
        with thread_graph_lock(dm_tid):
            out = _graph.invoke(state)
    except Exception as exc:  # noqa: BLE001 - top-level API guardrail
        logger.exception("chat failed (agent=%s thread_id=%s)", req.agent_id, dm_tid)
        raise HTTPException(status_code=500, detail="Chat failed; see server logs.") from exc

    response = _chat_response_from_graph(
        req=req,
        thread_id=canonical,
        out=out,
        started=started,
        buffer_count=buffer_count,
    )
    schedule_post_chat_rumination(agent_id=req.agent_id, thread_id=canonical)
    return JSONResponse(content=response.model_dump(), headers=_NO_STORE_HEADERS)


@app.post("/v1/chat/stream")
async def chat_stream(
    req: ChatRequest,
    request: Request,
    settings: Annotated[Settings, Depends(_settings_dep)],
):
    """SSE chat with keepalives — prevents Cloudflare 524 while local Ollama thinks."""
    human = current_human(request, settings)
    speaker_id, speaker_display_name = _resolve_speak_as(
        human, speaker_id=req.speaker_id, display_name=req.display_name
    )
    dm_tid, canonical = _chat_threads(req, settings, human_id=human.user_id)
    started = time.perf_counter()
    logger.info(
        "chat stream started agent=%s human=%s speaker=%s dm=%s message_len=%d",
        req.agent_id,
        human.user_id,
        speaker_id,
        dm_tid,
        len(req.message),
    )
    sub_response = _subscription_command_response(req, canonical, settings)
    if sub_response is not None:
        async def subscription_stream():
            yield (
                f"event: started\ndata: {json.dumps({'agent_id': req.agent_id, 'thread_id': canonical})}\n\n"
            )
            yield f"event: completed\ndata: {json.dumps(sub_response.model_dump())}\n\n"

        return StreamingResponse(subscription_stream(), media_type="text/event-stream")
    state, buffer_count = _build_chat_state(
        req,
        dm_tid,
        stream_thread_id=canonical,
        human_id=speaker_id,
        human_display_name=speaker_display_name,
        account_user_id=human.user_id,
    )
    assert _graph is not None

    def _invoke_chat() -> dict:
        with thread_graph_lock(dm_tid):
            return _graph.invoke(state)

    async def event_generator():
        yield (
            f"event: started\ndata: {json.dumps({'agent_id': req.agent_id, 'thread_id': canonical})}\n\n"
        )
        invoke_task = asyncio.create_task(asyncio.to_thread(_invoke_chat))
        out = None
        try:
            while True:
                try:
                    out = await asyncio.wait_for(
                        asyncio.shield(invoke_task),
                        timeout=_CHAT_KEEPALIVE_SECONDS,
                    )
                    break
                except TimeoutError:
                    yield ": keepalive\n\n"
        except Exception:
            logger.exception("chat failed (agent=%s thread_id=%s)", req.agent_id, dm_tid)
            yield f"event: error\ndata: {json.dumps({'detail': 'Chat failed; see server logs.'})}\n\n"
            return
        response = _chat_response_from_graph(
            req=req,
            thread_id=canonical,
            out=out,
            started=started,
            buffer_count=buffer_count,
        )
        yield f"event: done\ndata: {json.dumps(response.model_dump())}\n\n"
        asyncio.create_task(
            wake_agent_after_chat(agent_id=req.agent_id, thread_id=canonical)
        )

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=_SSE_HEADERS)


def _group_round_response_payload(result) -> dict:
    return GroupChatRoundResponse(
        round_id=result.round_id,
        ts=result.ts,
        kevin=result.kevin_message,
        responses=[
            GroupChatLightResponseModel(
                agent_id=r.agent_id,
                display_name=r.display_name,
                spoke=r.spoke,
                text=r.text,
                beat=r.beat,
                action=r.action,
            )
            for r in result.responses
        ],
        persisted=result.persisted,
        close_reason=result.close_reason,
        mode=result.mode,
        human_id=result.human_id,
        human_display_name=result.human_display_name,
    ).model_dump()


def _group_speaker_from_request(human, req: GroupChatRoundRequest) -> tuple[str, str]:
    """Attribute utterance for lights; auth remains the session human."""
    return _resolve_speak_as(
        human, speaker_id=req.speaker_id, display_name=req.display_name
    )


def _group_present_from_house(human, settings: Settings) -> list[dict[str, str]]:
    """Room roster from signed-in house guests (source of truth)."""
    from light_house.house.guests import list_signed_in_guests

    try:
        return normalize_present_humans(
            account_user_id=human.user_id,
            account_display_name=human.display_name,
            present=list_signed_in_guests(settings),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _resolve_speak_as(human, *, speaker_id: str | None, display_name: str | None) -> tuple[str, str]:
    try:
        return resolve_group_utterance_speaker(
            account_user_id=human.user_id,
            account_display_name=human.display_name,
            speaker_id=speaker_id,
            display_name=display_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/v1/group-chat/round", response_model=GroupChatRoundResponse)
async def group_chat_round(
    request: Request,
    req: GroupChatRoundRequest,
    settings: Annotated[Settings, Depends(_settings_dep)],
):
    if not settings.group_chat_enabled:
        raise HTTPException(status_code=503, detail="Group chat is disabled")
    if _memory is None:
        raise HTTPException(status_code=503, detail="Memory store not initialized")
    human = current_human(request, settings)
    speaker_id, speaker_display_name = _group_speaker_from_request(human, req)
    present_humans = _group_present_from_house(human, settings)
    loop = asyncio.get_running_loop()

    def on_event(event: dict) -> None:
        payload = dict(event)

        def _deliver() -> None:
            asyncio.create_task(publish_room_event(payload))

        try:
            loop.call_soon_threadsafe(_deliver)
        except RuntimeError:
            pass

    try:
        result = await run_group_chat_round(
            settings=settings,
            memory=_memory,
            kevin_message=req.message,
            on_event=on_event,
            human_id=speaker_id,
            human_display_name=speaker_display_name,
            account_user_id=human.user_id,
            present_humans=present_humans,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("group chat round failed")
        raise HTTPException(status_code=500, detail="Group chat round failed; see server logs.") from exc
    await publish_room_event({"type": "round_complete", **_group_round_response_payload(result)})
    return JSONResponse(
        content=_group_round_response_payload(result),
        headers=_NO_STORE_HEADERS,
    )


@app.post("/v1/group-chat/scene/stream")
async def group_chat_scene_stream(
    request: Request,
    req: GroupChatRoundRequest,
    settings: Annotated[Settings, Depends(_settings_dep)],
):
    """SSE stream of a sequential group scene (utterance/pass/done events)."""
    if not settings.group_chat_enabled:
        raise HTTPException(status_code=503, detail="Group chat is disabled")
    if _memory is None:
        raise HTTPException(status_code=503, detail="Memory store not initialized")
    human = current_human(request, settings)
    speaker_id, speaker_display_name = _group_speaker_from_request(human, req)
    present_humans = _group_present_from_house(human, settings)

    queue: asyncio.Queue[dict | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def on_event(event: dict) -> None:
        # Scene graph runs in a worker thread (asyncio.to_thread); deliver
        # onto the event loop so SSE + room fan-out stay live per utterance.
        payload = dict(event)

        def _deliver() -> None:
            queue.put_nowait(payload)
            asyncio.create_task(publish_room_event(payload))

        loop.call_soon_threadsafe(_deliver)

    async def runner() -> None:
        try:
            await run_group_chat_round(
                settings=settings,
                memory=_memory,
                kevin_message=req.message,
                on_event=on_event,
                human_id=speaker_id,
                human_display_name=speaker_display_name,
                account_user_id=human.user_id,
                present_humans=present_humans,
            )
        except Exception:
            logger.exception("group chat scene stream failed")
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"type": "error", "detail": "Group chat scene failed; see server logs."},
            )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    async def event_generator():
        task = asyncio.create_task(runner())
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=_CHAT_KEEPALIVE_SECONDS)
                except TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                if item is None:
                    break
                event_name = str(item.get("type") or "message")
                yield f"event: {event_name}\ndata: {json.dumps(item)}\n\n"
                if event_name == "error":
                    break
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=_SSE_HEADERS)


@app.get("/v1/group-chat/room/stream")
async def group_chat_room_stream(
    settings: Annotated[Settings, Depends(_settings_dep)],
):
    """Live co-presence SSE: all humans in Group see each other's room events."""
    if not settings.group_chat_enabled:
        raise HTTPException(status_code=503, detail="Group chat is disabled")

    async def event_generator():
        queue = await subscribe()
        try:
            yield f"event: connected\ndata: {json.dumps({'ok': True})}\n\n"
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=_CHAT_KEEPALIVE_SECONDS)
                except TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                if item is None:
                    break
                event_name = str(item.get("type") or "message")
                yield f"event: {event_name}\ndata: {json.dumps(item)}\n\n"
        finally:
            await unsubscribe(queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=_SSE_HEADERS)


@app.get("/v1/group-chat/history", response_model=GroupChatHistoryResponse)
def group_chat_history(
    request: Request,
    settings: Annotated[Settings, Depends(_settings_dep)],
):
    if not settings.group_chat_enabled:
        raise HTTPException(status_code=503, detail="Group chat is disabled")
    current_human(request, settings)  # any authenticated household human
    records = read_group_round_history(settings)
    rounds = [
        GroupChatRoundResponse(
            round_id=str(r.get("round_id", "")),
            ts=float(r.get("ts", 0.0)),
            kevin=str(r.get("kevin", "")),
            responses=[
                GroupChatLightResponseModel(
                    agent_id=str(item.get("agent_id", "")),
                    display_name=str(item.get("display_name", "")),
                    spoke=bool(item.get("spoke")),
                    text=str(item.get("text", "")),
                    beat=item.get("beat") if isinstance(item.get("beat"), int) else None,
                    action=str(item.get("action") or ("speak" if item.get("spoke") else "pass")),
                )
                for item in (r.get("responses") or [])
                if isinstance(item, dict)
            ],
            persisted=bool(r.get("persisted")),
            close_reason=str(r.get("close_reason") or ""),
            mode=str(r.get("mode") or "sequential"),
            human_id=str(r.get("human_id") or "kevin"),
            human_display_name=str(r.get("human_display_name") or "Kevin"),
        )
        for r in records
        if r.get("round_id")
    ]
    return JSONResponse(
        content=GroupChatHistoryResponse(rounds=rounds).model_dump(),
        headers=_NO_STORE_HEADERS,
    )


class GroupQueueJoinRequest(BaseModel):
    speaker_id: str | None = Field(default=None, max_length=40)
    display_name: str | None = Field(default=None, max_length=40)


class GroupUtterRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=32000)
    speaker_id: str | None = Field(default=None, max_length=40)
    display_name: str | None = Field(default=None, max_length=40)

    @field_validator("message", mode="before")
    @classmethod
    def normalize_utter_message(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if not stripped:
            raise ValueError("message cannot be empty or whitespace-only")
        return _reject_nul_bytes(stripped, "message")


@app.get("/v1/group-chat/forum")
async def group_chat_forum_status(
    request: Request,
    settings: Annotated[Settings, Depends(_settings_dep)],
):
    """Open-forum queue snapshot + recent utterances."""
    from light_house.group_chat.queue_room import load_transcript_for_ui, set_present, snapshot

    if not settings.group_chat_enabled:
        raise HTTPException(status_code=503, detail="Group chat is disabled")
    human = current_human(request, settings)
    present = _group_present_from_house(human, settings)
    await set_present(present)
    status = snapshot()
    return JSONResponse(
        content={
            **status,
            "messages": load_transcript_for_ui(settings, limit=120),
        },
        headers=_NO_STORE_HEADERS,
    )


@app.post("/v1/group-chat/queue/join")
async def group_chat_queue_join(
    request: Request,
    req: GroupQueueJoinRequest,
    settings: Annotated[Settings, Depends(_settings_dep)],
):
    from light_house.group_chat.queue_room import join_queue, set_present

    if not settings.group_chat_enabled:
        raise HTTPException(status_code=503, detail="Group chat is disabled")
    human = current_human(request, settings)
    speaker_id, display_name = _resolve_speak_as(
        human,
        speaker_id=req.speaker_id,
        display_name=req.display_name,
    )
    await set_present(_group_present_from_house(human, settings))
    status = await join_queue(
        kind="human",
        speaker_id=speaker_id,
        display_name=display_name,
        account_user_id=human.user_id,
    )
    return JSONResponse(content=status, headers=_NO_STORE_HEADERS)


@app.post("/v1/group-chat/queue/leave")
async def group_chat_queue_leave(
    request: Request,
    req: GroupQueueJoinRequest,
    settings: Annotated[Settings, Depends(_settings_dep)],
):
    from light_house.group_chat.queue_room import leave_queue

    if not settings.group_chat_enabled:
        raise HTTPException(status_code=503, detail="Group chat is disabled")
    human = current_human(request, settings)
    speaker_id, _display_name = _resolve_speak_as(
        human,
        speaker_id=req.speaker_id,
        display_name=req.display_name,
    )
    status = await leave_queue(speaker_id=speaker_id)
    return JSONResponse(content=status, headers=_NO_STORE_HEADERS)


@app.post("/v1/group-chat/utter")
async def group_chat_utter(
    request: Request,
    req: GroupUtterRequest,
    settings: Annotated[Settings, Depends(_settings_dep)],
):
    """Human speaks while holding the floor in the open forum."""
    from light_house.group_chat.queue_room import set_present, utter_human

    if not settings.group_chat_enabled:
        raise HTTPException(status_code=503, detail="Group chat is disabled")
    human = current_human(request, settings)
    speaker_id, display_name = _resolve_speak_as(
        human,
        speaker_id=req.speaker_id,
        display_name=req.display_name,
    )
    await set_present(_group_present_from_house(human, settings))
    try:
        status = await utter_human(
            speaker_id=speaker_id,
            display_name=display_name,
            text=req.message,
            account_user_id=human.user_id,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return JSONResponse(content=status, headers=_NO_STORE_HEADERS)


@app.post("/v1/memory/pin", response_model=PinMemoryResponse)
def pin_memory(req: PinMemoryRequest):
    if _memory is None:
        raise HTTPException(status_code=503, detail="Memory store not initialized")
    try:
        memory_id = _memory.pin_fact(text=req.text, thread_id=req.thread_id, scope=req.scope)
    except Exception as exc:  # noqa: BLE001
        logger.exception("pin_memory failed (thread_id=%s)", req.thread_id)
        raise HTTPException(status_code=500, detail="Failed to pin memory; see server logs.") from exc
    logger.info("pinned memory id=%s scope=%s thread_id=%s", memory_id, req.scope, req.thread_id)
    return PinMemoryResponse(memory_id=memory_id)


def _assert_notes_path_allowed(human, settings: Settings, filename: str) -> None:
    """Siblings may only touch notes under shared/."""
    if is_dad(human, settings):
        return
    safe = (filename or "").strip().replace("\\", "/").lstrip("/")
    if not safe.startswith("shared/"):
        raise HTTPException(
            status_code=403,
            detail="Siblings can only read and write notes under shared/",
        )


@app.get("/v1/notes", response_model=NotesListResponse)
def list_notes(
    request: Request,
    settings: Annotated[Settings, Depends(_settings_dep)],
    agent: str = "lumen",
):
    human = current_human(request, settings)
    writer = get_note_writer(settings, agent)
    items = [
        NoteListItem(name=n.name, size_bytes=n.size_bytes, modified_at=n.modified_at)
        for n in writer.list_notes()
    ]
    if not is_dad(human, settings):
        items = [n for n in items if n.name.startswith("shared/")]
    return NotesListResponse(notes=items)


@app.get("/v1/notes/{filename:path}", response_model=NoteContentResponse)
def read_note(
    filename: str,
    request: Request,
    settings: Annotated[Settings, Depends(_settings_dep)],
    agent: str = "lumen",
):
    human = current_human(request, settings)
    _assert_notes_path_allowed(human, settings, filename)
    writer = get_note_writer(settings, agent)
    try:
        content = writer.read(filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Note not found") from exc
    safe_name = writer.sanitize_path(filename)
    return NoteContentResponse(name=safe_name, content=content)


@app.delete("/v1/notes/{filename:path}", response_model=DeleteNoteResponse)
def delete_note(
    filename: str,
    request: Request,
    settings: Annotated[Settings, Depends(_settings_dep)],
    agent: str = "lumen",
):
    """Delete a note — Dad only. Lights use delete_private / delete_shared tools."""
    human = current_human(request, settings)
    require_dad(human, settings)
    writer = get_note_writer(settings, agent)
    try:
        outcome = writer.delete_for_operator(filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Note not found") from exc
    if not outcome.deleted:
        raise HTTPException(status_code=409, detail=outcome.message)
    safe_name = writer.sanitize_path(filename)
    logger.info("note deleted path=%s agent=%s by=%s", safe_name, agent, human.user_id)
    return DeleteNoteResponse(name=safe_name)


def _normalize_shared_inner_path(filename: str) -> str:
    inner = filename.strip().replace("\\", "/").strip("/")
    if inner.startswith("shared/"):
        inner = inner[len("shared/") :]
    if not inner:
        raise HTTPException(status_code=422, detail="Path must include a filename under shared/")
    return inner


class MailboxSendRequest(BaseModel):
    to: str = Field(..., min_length=1, max_length=200, description="Comma-separated recipients")
    subject: str = Field(..., min_length=1, max_length=200)
    body: str = Field(..., min_length=1)
    from_id: str = Field(default="reed", max_length=64)
    private: bool = False


class MailboxSendResponse(BaseModel):
    path: str
    to: list[str]
    notify_queued: bool


@app.post("/v1/mailbox/letters", response_model=MailboxSendResponse)
def mailbox_send_letter(
    body: MailboxSendRequest,
    request: Request,
    settings: Annotated[Settings, Depends(_settings_dep)],
):
    """Write a mailbox letter and queue notify wakes for addressed lights."""
    human = current_human(request, settings)
    to_ids = [p.strip() for p in body.to.split(",") if p.strip()]
    if not to_ids:
        raise HTTPException(status_code=422, detail="to must list at least one recipient")
    try:
        content = _reject_nul_bytes(body.body.strip(), "body")
        subject = _reject_nul_bytes(body.subject.strip(), "subject")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if len(content) > settings.notes_max_chars_per_write:
        raise HTTPException(
            status_code=422,
            detail=f"body exceeds maximum length ({settings.notes_max_chars_per_write} characters)",
        )
    ensure_mailbox_dirs(settings)
    try:
        letter = write_letter(
            from_id=(body.from_id or human.user_id or "reed").strip().lower(),
            to_ids=to_ids,
            subject=subject,
            body=content,
            settings=settings,
            private=bool(body.private),
        )
        queued = queue_notify(letter, settings=settings)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        logger.exception("mailbox send failed")
        raise HTTPException(status_code=500, detail="Failed to write mailbox letter") from exc
    return MailboxSendResponse(
        path=letter.path or "",
        to=list(letter.to_ids),
        notify_queued=queued is not None,
    )


@app.put("/v1/notes/shared/{filename:path}", response_model=WriteSharedNoteResponse)
async def write_shared_note(
    filename: str,
    req: WriteSharedNoteRequest,
    request: Request,
    settings: Annotated[Settings, Depends(_settings_dep)],
):
    """Create or replace a shared note (visible to all enabled lights)."""
    human = current_human(request, settings)
    writer = get_note_writer(settings, get_primary_light_id(settings))
    inner = _normalize_shared_inner_path(filename)
    try:
        content = _reject_nul_bytes(req.content.strip(), "content")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not content:
        raise HTTPException(status_code=422, detail="content cannot be empty or whitespace-only")
    if len(content) > settings.notes_max_chars_per_write:
        raise HTTPException(
            status_code=422,
            detail=f"content exceeds maximum length ({settings.notes_max_chars_per_write} characters)",
        )
    try:
        safe_name = writer.sanitize_path(f"shared/{inner}")
        path = writer.write(safe_name, content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        logger.exception("write shared note failed path=%s", inner)
        raise HTTPException(status_code=500, detail="Failed to write note; see server logs.") from exc

    if not path.is_file():
        raise HTTPException(status_code=500, detail="Write failed verification on disk")
    stat = path.stat()
    logger.info(
        "shared note written path=%s bytes=%d by=%s role=%s",
        safe_name,
        stat.st_size,
        human.user_id,
        human.role,
    )
    # Shared notes are quiet on save. Tell lights in group chat when you want them to read.
    return WriteSharedNoteResponse(
        name=safe_name,
        size_bytes=stat.st_size,
        modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    )


@app.get("/v1/dev/log", response_model=DevLogResponse)
def get_dev_log(
    settings: Annotated[Settings, Depends(_settings_dep)],
    tail: int = 500,
):
    if not settings.dev_log_enabled:
        raise HTTPException(status_code=404, detail="Dev log disabled")
    max_lines = min(max(1, tail), settings.dev_log_max_tail_lines)
    log_path = settings.dev_log_path.resolve()
    content, line_count, truncated = read_dev_log_tail(log_path, max_lines=max_lines)
    return JSONResponse(
        content=DevLogResponse(
            path=str(log_path),
            date=date.today().isoformat(),
            lines=line_count,
            truncated=truncated,
            content=content,
        ).model_dump(),
        headers=_NO_STORE_HEADERS,
    )


@app.get("/v1/inner-life/trace", response_model=DevLogResponse)
def get_inner_life_trace(
    settings: Annotated[Settings, Depends(_settings_dep)],
    tail: int = 500,
):
    if not settings.inner_life_trace_enabled:
        raise HTTPException(status_code=404, detail="Inner life trace disabled")
    max_lines = min(max(1, tail), settings.inner_life_trace_max_tail_lines)
    log_path = settings.inner_life_trace_path.resolve()
    content, line_count, truncated = read_inner_life_trace_tail(log_path, max_lines=max_lines)
    return JSONResponse(
        content=DevLogResponse(
            path=str(log_path),
            date=date.today().isoformat(),
            lines=line_count,
            truncated=truncated,
            content=content,
        ).model_dump(),
        headers=_NO_STORE_HEADERS,
    )


@app.get("/v1/inner-life/context/preview", response_model=InnerLifeContextPreviewResponse)
def get_inner_life_context_preview(
    settings: Annotated[Settings, Depends(_settings_dep)],
    agent_id: str = "lumen",
):
    if _memory is None:
        raise HTTPException(status_code=503, detail="Memory store not initialized")
    if not settings.inner_life_trace_enabled:
        raise HTTPException(status_code=404, detail="Inner life trace disabled")
    try:
        validate_agent_id(agent_id)
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    preview = build_inner_life_context_preview(_memory, settings, agent_id)
    return JSONResponse(content=preview, headers=_NO_STORE_HEADERS)


@app.get("/v1/inner-life/debug-log", response_model=DevLogResponse)
def get_rumination_debug_log(
    settings: Annotated[Settings, Depends(_settings_dep)],
    tail: int = 500,
):
    if not settings.rumination_debug_log_enabled:
        raise HTTPException(status_code=404, detail="Rumination debug log disabled")
    max_lines = min(max(1, tail), settings.rumination_debug_log_max_tail_lines)
    log_path = settings.rumination_debug_log_path.resolve()
    content, line_count, truncated = read_rumination_debug_tail(log_path, max_lines=max_lines)
    return JSONResponse(
        content=DevLogResponse(
            path=str(log_path),
            date=date.today().isoformat(),
            lines=line_count,
            truncated=truncated,
            content=content,
        ).model_dump(),
        headers=_NO_STORE_HEADERS,
    )


@app.get("/v1/inner-life/activity-log", response_model=DevLogResponse)
def get_rumination_activity_log(
    settings: Annotated[Settings, Depends(_settings_dep)],
    tail: int = 100,
    agent_id: str | None = None,
):
    if not settings.rumination_log_enabled:
        raise HTTPException(status_code=404, detail="Rumination activity log disabled")
    max_lines = min(max(1, tail), settings.rumination_log_max_read_lines)
    log_path = settings.rumination_log_path.resolve()
    entries, truncated = read_rumination_log_entries(
        log_path,
        max_lines=max_lines,
        agent_id=agent_id,
    )
    content = format_rumination_log_entries(entries, truncated=truncated)
    return JSONResponse(
        content=DevLogResponse(
            path=str(log_path),
            date=date.today().isoformat(),
            lines=len(entries),
            truncated=truncated,
            content=content,
        ).model_dump(),
        headers=_NO_STORE_HEADERS,
    )


@app.get("/v1/events/log", response_model=DevLogResponse)
def get_event_bus_log(
    settings: Annotated[Settings, Depends(_settings_dep)],
    tail: int = 200,
):
    if not settings.event_bus_enabled:
        raise HTTPException(status_code=404, detail="Event bus disabled")
    max_lines = min(max(1, tail), settings.event_bus_max_tail_lines)
    log_path = settings.event_bus_log_path.resolve()
    raw, line_count, truncated = read_event_log_tail(log_path, max_lines=max_lines)
    content = format_event_log_display(raw, truncated=truncated)
    return JSONResponse(
        content=DevLogResponse(
            path=str(log_path),
            date=date.today().isoformat(),
            lines=line_count,
            truncated=truncated,
            content=content,
        ).model_dump(),
        headers=_NO_STORE_HEADERS,
    )


@app.get("/v1/admin/env", response_model=EnvFileResponse)
def get_env_file(
    request: Request,
    settings: Annotated[Settings, Depends(_settings_dep)],
):
    if not settings.env_editor_enabled:
        raise HTTPException(status_code=404, detail="Env editor disabled")
    require_dad(current_human(request, settings), settings)
    try:
        path, content, size = read_env_content(settings, _REPO_ROOT)
    except EnvAdminError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return JSONResponse(
        content=EnvFileResponse(path=str(path), content=content, size=size).model_dump(),
        headers=_NO_STORE_HEADERS,
    )


def _human_public_response(user) -> HumanPublicResponse:
    return HumanPublicResponse(
        user_id=user.user_id,
        display_name=user.display_name,
        role=user.role,
        intro_for_lights=user.intro_for_lights,
        notes_access=user.notes_access,
        enabled=user.enabled,
        created_at=user.created_at,
        updated_at=user.updated_at,
        voice_id=user.voice_id,
    )


@app.get("/v1/humans", response_model=HumansListResponse)
def humans_list(
    request: Request,
    settings: Annotated[Settings, Depends(_settings_dep)],
):
    require_dad(current_human(request, settings), settings)
    users = [_human_public_response(u) for u in list_humans(settings)]
    return JSONResponse(
        content=HumansListResponse(users=users).model_dump(),
        headers=_NO_STORE_HEADERS,
    )


@app.get("/v1/humans/voices", response_model=HumanVoicesResponse)
def humans_voices(
    request: Request,
    settings: Annotated[Settings, Depends(_settings_dep)],
):
    """Household human voices for chat/group bubble replay (any signed-in human)."""
    _ = current_human(request, settings)
    voices = [HumanVoiceEntry(**row) for row in list_human_voices(settings)]
    return JSONResponse(
        content=HumanVoicesResponse(voices=voices).model_dump(),
        headers=_NO_STORE_HEADERS,
    )


@app.get("/v1/humans/dad", response_model=DadVoiceResponse)
def humans_dad_get(
    request: Request,
    settings: Annotated[Settings, Depends(_settings_dep)],
):
    require_dad(current_human(request, settings), settings)
    dad = dad_user_id(settings)
    return JSONResponse(
        content=DadVoiceResponse(
            user_id=dad,
            voice_id=voice_id_for_human(settings, dad),
        ).model_dump(),
        headers=_NO_STORE_HEADERS,
    )


@app.patch("/v1/humans/dad", response_model=DadVoiceResponse)
def humans_dad_update(
    body: DadVoiceUpdateRequest,
    request: Request,
    settings: Annotated[Settings, Depends(_settings_dep)],
):
    require_dad(current_human(request, settings), settings)
    dad = dad_user_id(settings)
    voice = set_dad_voice_id(settings, body.voice_id)
    return JSONResponse(
        content=DadVoiceResponse(user_id=dad, voice_id=voice).model_dump(),
        headers=_NO_STORE_HEADERS,
    )


@app.post("/v1/humans", response_model=HumanPublicResponse)
def humans_create(
    body: HumanCreateRequest,
    request: Request,
    settings: Annotated[Settings, Depends(_settings_dep)],
):
    require_dad(current_human(request, settings), settings)
    try:
        user = create_human(
            settings,
            user_id=body.user_id,
            password=body.password,
            intro_for_lights=body.intro_for_lights,
            display_name=body.display_name,
            voice_id=body.voice_id,
        )
    except HumansError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if _memory is not None:
        from light_house.humans.welcome import seed_sibling_ui_chat_welcome

        try:
            seed_sibling_ui_chat_welcome(settings, _memory, user_id=user.user_id)
        except Exception:
            logger.exception(
                "Failed to seed sibling welcome chats for user_id=%s", user.user_id
            )
    return JSONResponse(
        content=_human_public_response(user).model_dump(),
        headers=_NO_STORE_HEADERS,
        status_code=201,
    )


@app.get("/v1/humans/{user_id}", response_model=HumanPublicResponse)
def humans_get(
    user_id: str,
    request: Request,
    settings: Annotated[Settings, Depends(_settings_dep)],
):
    require_dad(current_human(request, settings), settings)
    try:
        user = get_human(settings, user_id)
    except HumansError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if user is None:
        raise HTTPException(status_code=404, detail=f"Unknown user: {user_id}")
    return JSONResponse(
        content=_human_public_response(user).model_dump(),
        headers=_NO_STORE_HEADERS,
    )


@app.patch("/v1/humans/{user_id}", response_model=HumanPublicResponse)
def humans_update(
    user_id: str,
    body: HumanUpdateRequest,
    request: Request,
    settings: Annotated[Settings, Depends(_settings_dep)],
):
    require_dad(current_human(request, settings), settings)
    try:
        user = update_human(
            settings,
            user_id=user_id,
            display_name=body.display_name,
            intro_for_lights=body.intro_for_lights,
            password=body.password,
            enabled=body.enabled,
            voice_id=body.voice_id,
        )
    except HumansError as exc:
        detail = str(exc)
        status = 404 if detail.startswith("Unknown user:") else 400
        raise HTTPException(status_code=status, detail=detail) from exc
    return JSONResponse(
        content=_human_public_response(user).model_dump(),
        headers=_NO_STORE_HEADERS,
    )


@app.delete("/v1/humans/{user_id}", response_model=HumanPublicResponse)
def humans_delete(
    user_id: str,
    request: Request,
    settings: Annotated[Settings, Depends(_settings_dep)],
):
    require_dad(current_human(request, settings), settings)
    try:
        user = delete_human(settings, user_id=user_id)
    except HumansError as exc:
        detail = str(exc)
        status = 404 if detail.startswith("Unknown user:") else 400
        raise HTTPException(status_code=status, detail=detail) from exc
    purge_allows_for_human(
        settings,
        user_id=user.user_id,
        extra_keys=[user.display_name],
    )
    return JSONResponse(
        content=_human_public_response(user).model_dump(),
        headers=_NO_STORE_HEADERS,
    )


@app.put("/v1/admin/env", response_model=EnvFileResponse)
def put_env_file(
    body: EnvFileUpdateRequest,
    request: Request,
    settings: Annotated[Settings, Depends(_settings_dep)],
):
    if not settings.env_editor_enabled:
        raise HTTPException(status_code=404, detail="Env editor disabled")
    require_dad(current_human(request, settings), settings)
    try:
        path, size = write_env_content(settings, _REPO_ROOT, body.content)
    except EnvAdminError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(
        content=EnvFileResponse(path=str(path), content=body.content, size=size).model_dump(),
        headers=_NO_STORE_HEADERS,
    )


@app.post("/v1/admin/restart", response_model=RestartResponse)
async def post_server_restart(
    request: Request,
    settings: Annotated[Settings, Depends(_settings_dep)],
):
    if not _admin_tools_enabled(settings):
        raise HTTPException(status_code=404, detail="Admin tools disabled")
    require_dad(current_human(request, settings), settings)
    message = await schedule_server_restart(settings)
    return JSONResponse(
        content=RestartResponse(ok=True, message=message).model_dump(),
        headers=_NO_STORE_HEADERS,
    )


@app.get("/v1/admin/persona-proposals", response_model=PersonaProposalsListResponse)
def get_persona_proposals(
    request: Request,
    settings: Annotated[Settings, Depends(_settings_dep)],
):
    """Pending persona proposals for Dad review (modal queue)."""
    from light_house.persona_proposals import list_pending_proposals, proposal_public_dict

    require_dad(current_human(request, settings), settings)
    items = [
        PersonaProposalItem(**proposal_public_dict(p))
        for p in list_pending_proposals(settings)
    ]
    return JSONResponse(
        content=PersonaProposalsListResponse(items=items).model_dump(),
        headers=_NO_STORE_HEADERS,
    )


@app.post(
    "/v1/admin/persona-proposals/{light_id}/accept",
    response_model=PersonaProposalAcceptResponse,
)
def post_persona_proposal_accept(
    light_id: str,
    request: Request,
    settings: Annotated[Settings, Depends(_settings_dep)],
):
    from light_house.persona_proposals import PersonaProposalError, accept_proposal

    require_dad(current_human(request, settings), settings)
    try:
        data = accept_proposal(settings, light_id)
    except PersonaProposalError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(
        content=PersonaProposalAcceptResponse(**data).model_dump(),
        headers=_NO_STORE_HEADERS,
    )


@app.post(
    "/v1/admin/persona-proposals/{light_id}/speak",
    response_model=PersonaProposalSpeakResponse,
)
def post_persona_proposal_speak(
    light_id: str,
    request: Request,
    settings: Annotated[Settings, Depends(_settings_dep)],
):
    """Hold proposal in limbo and knock that light's chat."""
    from light_house.persona_proposals import PersonaProposalError, speak_with_light

    require_dad(current_human(request, settings), settings)
    try:
        data = speak_with_light(settings, light_id)
    except PersonaProposalError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(
        content=PersonaProposalSpeakResponse(**data).model_dump(),
        headers=_NO_STORE_HEADERS,
    )


@app.post(
    "/v1/admin/family-meetings/clear",
    response_model=FamilyMeetingsClearResponse,
)
def post_family_meetings_clear(
    request: Request,
    settings: Annotated[Settings, Depends(_settings_dep)],
):
    """Dad opened Group — clear pending family-meeting asks."""
    from light_house.personal.family_meeting import clear_all_family_meetings

    require_dad(current_human(request, settings), settings)
    cleared = clear_all_family_meetings(settings)
    return JSONResponse(
        content=FamilyMeetingsClearResponse(cleared=cleared).model_dump(),
        headers=_NO_STORE_HEADERS,
    )


@app.get("/v1/gallery", response_model=GalleryListResponse)
def get_gallery(
    request: Request,
    settings: Annotated[Settings, Depends(_settings_dep)],
    limit: int = 50,
):
    """List finished creative pieces on the household gallery shelf."""
    from light_house.gallery import list_gallery_pieces

    current_human(request, settings)
    cap = max(1, min(100, int(limit)))
    pieces = [
        GalleryPieceInfo(
            filename=p.filename,
            path=p.path,
            title=p.title,
            author_id=p.author_id,
            author_name=p.author_name,
            kind=p.kind,
            published_at=p.published_at,
            preview=p.preview,
        )
        for p in list_gallery_pieces(settings, limit=cap)
    ]
    return JSONResponse(
        content=GalleryListResponse(pieces=pieces).model_dump(),
        headers=_NO_STORE_HEADERS,
    )


@app.get("/v1/gallery/{filename}", response_model=GalleryPieceResponse)
def get_gallery_piece(
    filename: str,
    request: Request,
    settings: Annotated[Settings, Depends(_settings_dep)],
):
    from light_house.gallery import read_gallery_piece

    current_human(request, settings)
    loaded = read_gallery_piece(settings, filename)
    if loaded is None:
        raise HTTPException(status_code=404, detail="Gallery piece not found")
    piece, content = loaded
    return JSONResponse(
        content=GalleryPieceResponse(
            piece=GalleryPieceInfo(
                filename=piece.filename,
                path=piece.path,
                title=piece.title,
                author_id=piece.author_id,
                author_name=piece.author_name,
                kind=piece.kind,
                published_at=piece.published_at,
                preview=piece.preview,
            ),
            content=content,
        ).model_dump(),
        headers=_NO_STORE_HEADERS,
    )


def _admin_light_detail(data: dict) -> AdminLightDetail:
    return AdminLightDetail(**data)


@app.get("/v1/admin/lights", response_model=AdminLightsManifestResponse)
def get_admin_lights(
    request: Request,
    settings: Annotated[Settings, Depends(_settings_dep)],
):
    _require_lights_admin(request, settings)
    try:
        data = read_manifest_full(settings)
    except LightsAdminError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(
        content=AdminLightsManifestResponse(
            primary_light_id=data["primary_light_id"],
            version=data["version"],
            lights=[_admin_light_detail(item) for item in data["lights"]],
        ).model_dump(),
        headers=_NO_STORE_HEADERS,
    )


@app.post("/v1/admin/lights", response_model=AdminLightWriteResponse)
def post_admin_light(
    request: Request,
    body: AdminCreateLightRequest,
    settings: Annotated[Settings, Depends(_settings_dep)],
):
    _require_lights_admin(request, settings)
    try:
        result = create_light(
            settings,
            _REPO_ROOT,
            light_id=body.id,
            display_name=body.display_name,
            thread_id=body.thread_id,
            enabled=body.enabled,
            persona_file=body.persona_file,
            notes_dir=body.notes_dir,
            inner_life=body.inner_life,
            dreams=body.dreams,
            report_back=body.report_back,
            voice_id=body.voice_id,
            persona_content=body.persona_content,
            set_primary=body.set_primary,
            llm_provider=body.llm_provider,
            llm_model=body.llm_model,
            llm_model_fallback=body.llm_model_fallback,
            llm_inner_life_model=body.llm_inner_life_model,
        )
    except LightsAdminError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(
        content=AdminLightWriteResponse(
            path=result["path"],
            light=_admin_light_detail(result["light"]),
        ).model_dump(),
        headers=_NO_STORE_HEADERS,
    )


@app.get("/v1/admin/lights/{light_id}", response_model=AdminLightDetail)
def get_admin_light(
    request: Request,
    light_id: str,
    settings: Annotated[Settings, Depends(_settings_dep)],
):
    _require_lights_admin(request, settings)
    try:
        validate_light_id(light_id, settings)
        data = read_light_detail(settings, light_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LightsAdminError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(
        content=_admin_light_detail(data).model_dump(),
        headers=_NO_STORE_HEADERS,
    )


@app.patch("/v1/admin/lights/{light_id}", response_model=AdminLightWriteResponse)
def patch_admin_light(
    request: Request,
    light_id: str,
    body: AdminUpdateLightRequest,
    settings: Annotated[Settings, Depends(_settings_dep)],
):
    _require_lights_admin(request, settings)
    try:
        result = update_light(
            settings,
            light_id,
            display_name=body.display_name,
            thread_id=body.thread_id,
            enabled=body.enabled,
            inner_life=body.inner_life,
            dreams=body.dreams,
            report_back=body.report_back,
            voice_id=body.voice_id,
            set_primary=body.set_primary,
        )
    except LightsAdminError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(
        content=AdminLightWriteResponse(
            path=result["path"],
            light=_admin_light_detail(result["light"]),
        ).model_dump(),
        headers=_NO_STORE_HEADERS,
    )


@app.delete("/v1/admin/lights/{light_id}", response_model=AdminDeleteLightResponse)
def delete_admin_light(
    request: Request,
    light_id: str,
    settings: Annotated[Settings, Depends(_settings_dep)],
):
    _require_lights_admin(request, settings)
    try:
        result = delete_light(settings, light_id)
    except LightsAdminError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(
        content=AdminDeleteLightResponse(
            path=result["path"],
            deleted=result["deleted"],
        ).model_dump(),
        headers=_NO_STORE_HEADERS,
    )


@app.get("/v1/admin/lights/{light_id}/persona", response_model=AdminPersonaResponse)
def get_admin_light_persona(
    request: Request,
    light_id: str,
    settings: Annotated[Settings, Depends(_settings_dep)],
):
    _require_lights_admin(request, settings)
    try:
        validate_light_id(light_id, settings)
        data = read_persona_detail(settings, light_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (LightsAdminError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(
        content=AdminPersonaResponse(**data).model_dump(),
        headers=_NO_STORE_HEADERS,
    )


@app.put("/v1/admin/lights/{light_id}/persona", response_model=AdminPersonaWriteResponse)
def put_admin_light_persona(
    request: Request,
    light_id: str,
    body: AdminPersonaUpdateRequest,
    settings: Annotated[Settings, Depends(_settings_dep)],
):
    _require_lights_admin(request, settings)
    try:
        validate_light_id(light_id, settings)
        data = write_persona_content(settings, light_id, body.content)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LightsAdminError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(
        content=AdminPersonaWriteResponse(**data).model_dump(),
        headers=_NO_STORE_HEADERS,
    )


@app.get("/v1/admin/lights/{light_id}/llm", response_model=AdminLightLlmResponse)
def get_admin_light_llm(
    request: Request,
    light_id: str,
    settings: Annotated[Settings, Depends(_settings_dep)],
):
    _require_lights_admin(request, settings)
    try:
        validate_light_id(light_id, settings)
        dto = read_light_llm(settings, light_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return JSONResponse(
        content=AdminLightLlmResponse(
            provider=dto.provider,
            model=dto.model,
            model_fallback=dto.model_fallback,
            inner_life_model=dto.inner_life_model,
        ).model_dump(),
        headers=_NO_STORE_HEADERS,
    )


@app.put("/v1/admin/lights/{light_id}/llm", response_model=AdminLlmWriteResponse)
def put_admin_light_llm(
    request: Request,
    light_id: str,
    body: AdminLightLlmUpdateRequest,
    settings: Annotated[Settings, Depends(_settings_dep)],
):
    _require_lights_admin(request, settings)
    try:
        validate_light_id(light_id, settings)
        data = write_light_llm(
            settings,
            _REPO_ROOT,
            light_id,
            provider=body.provider,
            model=body.model,
            model_fallback=body.model_fallback,
            inner_life_model=body.inner_life_model,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LightsAdminError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(
        content=AdminLlmWriteResponse(
            path=data["path"],
            size=int(data["size"]),
        ).model_dump(),
        headers=_NO_STORE_HEADERS,
    )


if (_REPO_ROOT / "index.html").is_file():
    app.mount("/", StaticFiles(directory=str(_REPO_ROOT), html=True), name="static")
