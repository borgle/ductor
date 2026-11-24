"""Interactive model selector wizard for Telegram inline keyboards."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ductor_bot.cli.auth import AuthStatus, check_all_auth
from ductor_bot.config import (
    ANTIGRAVITY_MODELS_ORDERED,
    CLAUDE_MODELS_ORDERED,
    CLAUDE_SUPPORTED_EFFORTS,
    CODEX_SUPPORTED_EFFORTS_FALLBACK,
    GROK_SUPPORTED_EFFORTS,
    PI_MODEL_PREFIX,
    PI_SUPPORTED_EFFORTS,
    get_antigravity_models,
    get_gemini_models,
    get_grok_models_ordered,
    get_pi_models_ordered,
    update_config_file_async,
)
from ductor_bot.i18n import t
from ductor_bot.multiagent.registry import update_agent_fields
from ductor_bot.orchestrator.selectors.models import (
    Button,
    ButtonGrid,
    ModelBrowserState,
    SelectorResponse,
)

if TYPE_CHECKING:
    from ductor_bot.cli.codex_cache import CodexModelCache
    from ductor_bot.orchestrator.core import Orchestrator
    from ductor_bot.session import SessionData
    from ductor_bot.session.key import SessionKey

logger = logging.getLogger(__name__)

MS_PREFIX = "ms:"

_EFFORT_LABELS: dict[str, str] = {
    "none": "None",
    "minimal": "Minimal",
    "low": "Low",
    "medium": "Medium",
    "high": "High",
    "xhigh": "XHigh",
    "max": "Max",
}


@dataclass(frozen=True)
class _SwitchSummaryContext:
    old_model: str
    new_model: str
    old_provider: str
    new_provider: str
    provider_changed: bool
    reasoning_effort: str | None
    effort_only: bool
    resume_session_id: str
    resume_message_count: int


def _resume_state_for_provider(session: SessionData | None, provider: str) -> tuple[str, int]:
    """Return (session_id, message_count) for provider if resumable history exists."""
    if session is None:
        return "", 0
    provider_data = session.provider_sessions.get(provider)
    if provider_data is None or provider_data.message_count <= 0:
        return "", 0
    return provider_data.session_id, provider_data.message_count


def _format_resume_hint(session_id: str, message_count: int, model_id: str) -> str:
    """Build post-switch resume hint text."""
    message_label = t("model.resume_msg_one") if message_count == 1 else t("model.resume_msg_other")
    sid_display = session_id or "pending"
    return "\n" + t(
        "model.resume_hint",
        sid=sid_display,
        count=message_count,
        label=message_label,
        model=model_id,
    )


def _build_switch_summary(ctx: _SwitchSummaryContext) -> str:
    """Build user-facing model switch summary text."""
    parts: list[str] = [t("model.switched")]
    if ctx.old_model == ctx.new_model:
        parts.append(t("model.model_line", model=ctx.new_model))
    else:
        parts.append(t("model.model_change", old=ctx.old_model, new=ctx.new_model))
    if ctx.provider_changed:
        parts.append(t("model.provider_change", old=ctx.old_provider, new=ctx.new_provider))
    if ctx.reasoning_effort:
        parts.append(t("model.reasoning_line", effort=ctx.reasoning_effort))
    if ctx.old_model != ctx.new_model and ctx.resume_message_count > 0:
        parts.append(
            _format_resume_hint(
                ctx.resume_session_id,
                ctx.resume_message_count,
                ctx.new_model,
            )
        )
    if ctx.effort_only:
        parts.append("\n" + t("model.effort_updated", effort=ctx.reasoning_effort))
    return "\n".join(parts)


def _supported_efforts(orch: Orchestrator, model_id: str) -> tuple[str, ...]:
    """Return the reasoning-effort levels supported by *model_id*'s provider.

    Claude accepts a fixed set (incl. ``max``). Codex uses the live model
    cache when available, else a fallback constant so ``max`` is still
    rejected. Other providers don't use reasoning effort.
    """
    provider = orch.models.provider_for(model_id)
    if provider == "claude":
        return CLAUDE_SUPPORTED_EFFORTS
    if provider == "grok":
        return GROK_SUPPORTED_EFFORTS
    if provider == "pi":
        return PI_SUPPORTED_EFFORTS
    if provider == "codex":
        codex_cache = (
            orch._observers.codex_cache_obs.get_cache() if orch._observers.codex_cache_obs else None
        )
        model_info = codex_cache.get_model(model_id) if codex_cache else None
        if model_info is not None:
            return tuple(model_info.supported_efforts)
        return CODEX_SUPPORTED_EFFORTS_FALLBACK
    return ()


def _validate_reasoning_effort(
    orch: Orchestrator,
    model_id: str,
    reasoning_effort: str | None,
) -> str | None:
    """Return a user-facing error when the effort is unsupported by the provider.

    Providers without a reasoning-effort concept (gemini/antigravity) skip
    validation; codex/claude/grok validate against their supported set (an empty
    set rejects any effort).
    """
    if not reasoning_effort:
        return None
    if orch.models.provider_for(model_id) not in ("codex", "claude", "grok", "pi"):
        return None

    supported = _supported_efforts(orch, model_id)
    if reasoning_effort in supported:
        return None

    supported_display = ", ".join(supported) if supported else "none"
    return (
        f"Invalid reasoning effort `{reasoning_effort}` for `{model_id}`. "
        f"Supported: {supported_display}."
    )


# Built-in Gemini aliases accepted by the Gemini CLI as `--model` values.
# `auto` lets the CLI pick the best model per request; `pro`/`flash`/`flash-lite`
# select the latest model of that tier without pinning a specific version.
_GEMINI_ALIAS_ORDER: tuple[str, ...] = ("auto", "pro", "flash", "flash-lite")


def _gemini_models_for_selector() -> list[str]:
    """Return Gemini built-in aliases, then models discovered from the CLI files."""
    models = sorted(get_gemini_models())
    # Prefer stable models before previews in the selector.
    stable = [model for model in models if "preview" not in model]
    preview = [model for model in models if "preview" in model]
    return list(dict.fromkeys([*_GEMINI_ALIAS_ORDER, *stable, *preview]))


def _antigravity_models_for_selector() -> list[str]:
    """Return the agy default, then models discovered from ``agy models``."""
    discovered = sorted(get_antigravity_models())
    return list(dict.fromkeys([*ANTIGRAVITY_MODELS_ORDERED, *discovered]))


# Telegram tolerates large keyboards, but Matrix renders every button as a
# numbered option and reacts to only the first 10. Keep pages within that limit
# there while showing more models at once on Telegram.
_BROWSER_PAGE_SIZE_TG = 8
_BROWSER_PAGE_SIZE_MX = 5
_SEARCH_PROMPT_TTL = 300.0
_SEARCH_QUERY_MAX_LEN = 120

PROVIDERS: tuple[str, ...] = ("claude", "codex", "gemini", "antigravity", "grok", "pi")


@dataclass(frozen=True, slots=True)
class _ModelEntry:
    """One selectable model, with the text used to search it."""

    model_id: str
    label: str
    description: str = ""


def _button_label(model_id: str) -> str:
    """Compact button label while preserving identity in callback data."""
    return model_id.removeprefix("gemini-").removeprefix("auto-")


def _fit_button_label(label: str, *, limit: int = 60) -> str:
    """Clamp a label to Telegram's inline-button text limit."""
    return label if len(label) <= limit else f"{label[: limit - 1]}…"


def _rows_of(buttons: list[Button], *, per_row: int = 2) -> list[list[Button]]:
    """Split buttons into rows of at most *per_row* (readability on mobile)."""
    return [buttons[i : i + per_row] for i in range(0, len(buttons), per_row)]


def _page_size(key: SessionKey) -> int:
    """Number of model buttons per page for the transport in *key*."""
    return _BROWSER_PAGE_SIZE_MX if key.transport == "mx" else _BROWSER_PAGE_SIZE_TG


def _codex_cache(orch: Orchestrator) -> CodexModelCache | None:
    observer = orch._observers.codex_cache_obs
    return observer.get_cache() if observer else None


def _model_entries(  # noqa: PLR0911
    provider: str, codex_cache: CodexModelCache | None
) -> list[_ModelEntry]:
    """Return the ordered, searchable model catalog for *provider*."""
    if provider == "claude":
        return [_ModelEntry(model_id=m, label=m.upper()) for m in CLAUDE_MODELS_ORDERED]
    if provider == "gemini":
        return [
            _ModelEntry(model_id=m, label=_button_label(m)) for m in _gemini_models_for_selector()
        ]
    if provider == "antigravity":
        return [_ModelEntry(model_id=m, label=m) for m in _antigravity_models_for_selector()]
    if provider == "grok":
        return [_ModelEntry(model_id=m, label=m) for m in get_grok_models_ordered()]
    if provider == "pi":
        return [
            _ModelEntry(model_id=m, label=m.removeprefix(PI_MODEL_PREFIX))
            for m in get_pi_models_ordered()
        ]
    if provider == "codex":
        models = codex_cache.models if codex_cache else []
        return [
            _ModelEntry(
                model_id=m.id,
                label=m.display_name or m.id,
                description=m.description,
            )
            for m in models
        ]
    return []


def _filter_entries(entries: list[_ModelEntry], query: str) -> list[_ModelEntry]:
    """Case-insensitive multi-token filter over id, label, and description."""
    tokens = [token for token in query.casefold().split() if token]
    if not tokens:
        return entries
    return [
        entry
        for entry in entries
        if all(
            token in f"{entry.model_id} {entry.label} {entry.description}".casefold()
            for token in tokens
        )
    ]


def _safe_query(query: str) -> str:
    """Keep user-entered filter text from breaking Markdown rendering."""
    return query.replace("`", "'").replace("\n", " ").strip()[:80]


def _select_prompt(provider: str) -> str:
    return t(f"model.select_{provider}")


def _page_count(total: int, size: int) -> int:
    return max(1, (total + size - 1) // size)


def _browser_text(  # noqa: PLR0913
    header: str,
    provider: str,
    *,
    query: str,
    total: int,
    page: int,
    pages: int,
) -> str:
    lines = [header, "", _select_prompt(provider)]
    if pages > 1:
        lines.append(t("model.page_indicator", page=page + 1, total=pages))
    if query:
        lines.append(t("model.filter_indicator", query=_safe_query(query), count=total))
    return "\n".join(lines)


def _no_models_text(header: str, provider: str, query: str) -> str:
    if query:
        return f"{header}\n\n{t('model.no_match', provider=provider.upper(), query=_safe_query(query))}"
    if provider == "codex":
        return f"{header}\n\n{t('model.no_codex')}"
    if provider == "gemini":
        return f"{header}\n\n{t('model.no_gemini')}"
    return f"{header}\n\n{t('model.no_models', provider=provider.upper())}"


def _browser_rows(
    provider: str,
    entries: list[_ModelEntry],
    *,
    page: int,
    pages: int,
    query: str,
) -> list[list[Button]]:
    rows = _rows_of(
        [
            Button(
                text=_fit_button_label(entry.label),
                callback_data=f"ms:m:{entry.model_id}",
            )
            for entry in entries
        ]
    )

    if pages > 1:
        nav: list[Button] = []
        if page > 0:
            nav.append(Button(text=t("model.btn_prev"), callback_data=f"ms:g:{page - 1}"))
        if page < pages - 1:
            nav.append(Button(text=t("model.btn_next"), callback_data=f"ms:g:{page + 1}"))
        if nav:
            rows.append(nav)

    actions: list[Button] = []
    if query:
        actions.append(Button(text=t("model.btn_clear"), callback_data=f"ms:p:{provider}"))
    actions.append(Button(text=t("model.btn_search"), callback_data=f"ms:s:{provider}"))
    actions.append(Button(text=t("model.btn_back"), callback_data="ms:b:root"))
    rows.append(actions)
    return rows


def _store_browser_state(  # noqa: PLR0913
    orch: Orchestrator,
    key: SessionKey,
    *,
    provider: str,
    query: str,
    page: int,
    awaiting_query: bool = False,
) -> None:
    orch._model_browser[key] = ModelBrowserState(
        provider=provider,
        query=query,
        page=page,
        awaiting_query=awaiting_query,
        updated_at=time.monotonic(),
    )


def clear_awaiting_model_search(orch: Orchestrator, key: SessionKey) -> None:
    """Stop treating the next message as a model-filter query."""
    state = orch._model_browser.get(key)
    if state is not None and state.awaiting_query:
        state.awaiting_query = False


async def consume_model_search(
    orch: Orchestrator,
    key: SessionKey,
    text: str,
) -> SelectorResponse | None:
    """Consume *text* as a model filter when the browser asked for one.

    Returns ``None`` for anything that is not a pending search query (including
    stale prompts and slash commands) so normal message routing continues.
    """
    state = orch._model_browser.get(key)
    if state is None or not state.awaiting_query:
        return None
    if time.monotonic() - state.updated_at > _SEARCH_PROMPT_TTL:
        orch._model_browser.pop(key, None)
        return None

    query = text.strip()
    # Never swallow commands, directives, or full prompts: only short plain
    # text is treated as a filter keyword.
    if not query or query.startswith(("/", "@")) or len(query) > _SEARCH_QUERY_MAX_LEN:
        clear_awaiting_model_search(orch, key)
        return None
    if query.casefold() in {"cancel", "abort"}:
        clear_awaiting_model_search(orch, key)
        return await _show_models(orch, key, state.provider)

    return await _show_models(orch, key, state.provider, query=query)


async def show_provider_models(
    orch: Orchestrator,
    key: SessionKey,
    provider: str,
    *,
    query: str = "",
    page: int = 0,
) -> SelectorResponse:
    """Public entry point for ``/model <provider> [keyword]``."""
    return await _show_models(orch, key, provider, query=query, page=page)


async def _show_models(
    orch: Orchestrator,
    key: SessionKey,
    provider: str,
    *,
    query: str = "",
    page: int = 0,
) -> SelectorResponse:
    """Render one page of a provider's (optionally filtered) model catalog."""
    entries = _filter_entries(_model_entries(provider, _codex_cache(orch)), query)
    size = _page_size(key)
    pages = _page_count(len(entries), size)
    page = max(0, min(page, pages - 1))
    _store_browser_state(orch, key, provider=provider, query=query, page=page)

    header = await _status_line(orch, key)
    if not entries:
        rows = [
            [Button(text=t("model.btn_back"), callback_data="ms:b:root")],
        ]
        if query:
            rows.insert(
                0,
                [Button(text=t("model.btn_clear"), callback_data=f"ms:p:{provider}")],
            )
        return SelectorResponse(
            text=_no_models_text(header, provider, query),
            buttons=ButtonGrid(rows=rows),
        )

    start = page * size
    rows = _browser_rows(
        provider,
        entries[start : start + size],
        page=page,
        pages=pages,
        query=query,
    )
    text = _browser_text(
        header,
        provider,
        query=query,
        total=len(entries),
        page=page,
        pages=pages,
    )
    return SelectorResponse(text=text, buttons=ButtonGrid(rows=rows))


async def _search_prompt(orch: Orchestrator, key: SessionKey, provider: str) -> SelectorResponse:
    """Ask for a filter keyword and arm the next plain message as the query."""
    _store_browser_state(
        orch,
        key,
        provider=provider,
        query="",
        page=0,
        awaiting_query=True,
    )
    header = await _status_line(orch, key)
    keyboard = ButtonGrid(rows=[[Button(text=t("model.btn_cancel"), callback_data="ms:c")]])
    return SelectorResponse(
        text=f"{header}\n\n{t('model.search_prompt', provider=provider.upper())}",
        buttons=keyboard,
    )


def is_model_selector_callback(data: str) -> bool:
    """Return True if *data* belongs to the model selector wizard."""
    return data.startswith(MS_PREFIX)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def model_selector_start(
    orch: Orchestrator,
    key: SessionKey,
) -> SelectorResponse:
    """Build the initial ``/model`` response with provider buttons.

    Returns a ``SelectorResponse``. Buttons are ``None`` when no providers
    are authenticated.
    """
    auth = await asyncio.to_thread(check_all_auth)
    authed = [name for name, res in auth.items() if res.status == AuthStatus.AUTHENTICATED]

    header = await _status_line(orch, key)

    if not authed:
        return SelectorResponse(
            text=f"{header}\n\n{t('model.no_auth')}",
        )

    if len(authed) == 1:
        return await _show_models(orch, key, authed[0])

    buttons: list[Button] = []
    if "claude" in authed:
        buttons.append(Button(text="CLAUDE", callback_data="ms:p:claude"))
    if "codex" in authed:
        buttons.append(Button(text="CODEX", callback_data="ms:p:codex"))
    if "gemini" in authed:
        buttons.append(Button(text="GEMINI", callback_data="ms:p:gemini"))
    if "antigravity" in authed:
        buttons.append(Button(text="ANTIGRAVITY", callback_data="ms:p:antigravity"))
    if "grok" in authed:
        buttons.append(Button(text="GROK BUILD", callback_data="ms:p:grok"))
    if "pi" in authed:
        buttons.append(Button(text="PI", callback_data="ms:p:pi"))

    provider_rows = [buttons[i : i + 2] for i in range(0, len(buttons), 2)]
    keyboard = ButtonGrid(rows=provider_rows)
    return SelectorResponse(text=f"{header}\n\n{t('model.pick_provider')}", buttons=keyboard)


async def effort_selector_start(
    orch: Orchestrator,
    key: SessionKey,
) -> SelectorResponse:
    """Build the ``/effort`` response: effort buttons for the active provider.

    ``/effort`` always targets the **current active session** (main DM or topic):
    it changes only that session's reasoning effort and never the configured
    default. The buttons use a dedicated ``ms:e:<effort>`` callback so they route
    to the per-session apply path, distinct from the ``/model`` wizard's
    ``ms:r`` step (which sets the global default for the main chat). Providers
    without reasoning effort (gemini/antigravity) return an info message only.
    """
    # Effort range is scoped to the active session's model (main DM session has
    # its own model/effort, like a topic). Fall back to the config default model
    # only when no active session exists yet.
    session = await orch._sessions.get_active(key)
    if session:
        model, provider = session.model, session.provider
    else:
        model, provider = orch.resolve_runtime_target(orch._config.model)

    supported = _supported_efforts(orch, model)
    if not supported:
        return SelectorResponse(text=t("effort.unsupported", provider=provider))

    buttons = [Button(text=_EFFORT_LABELS.get(e, e), callback_data=f"ms:e:{e}") for e in supported]
    keyboard = ButtonGrid(
        rows=[
            *_rows_of(buttons),
            [Button(text=t("model.btn_back"), callback_data="ms:b:root")],
        ]
    )
    header = await _status_line(orch, key)
    return SelectorResponse(text=f"{header}\n\n{t('effort.select', model=model)}", buttons=keyboard)


async def handle_model_callback(  # noqa: C901, PLR0911, PLR0912
    orch: Orchestrator,
    key: SessionKey,
    data: str,
) -> SelectorResponse:
    """Route an ``ms:*`` callback to the correct wizard step.

    Returns a ``SelectorResponse`` for editing the message in-place.
    """
    logger.debug("Model selector step=%s", data[:40])
    parts = data[len(MS_PREFIX) :].split(":", 2)
    action = parts[0] if parts else ""
    payload = parts[1] if len(parts) > 1 else ""
    extra = parts[2] if len(parts) > 2 else ""

    if action == "p":
        # Opening a provider starts a fresh, unfiltered browse.
        if payload not in PROVIDERS:
            return await model_selector_start(orch, key)
        page = int(extra) if extra.isdigit() else 0
        return await _show_models(orch, key, payload, page=page)

    if action == "g":
        # Page navigation preserves the active filter.
        state = orch._model_browser.get(key)
        if state is None:
            return await model_selector_start(orch, key)
        page = int(payload) if payload.isdigit() else 0
        return await _show_models(orch, key, state.provider, query=state.query, page=page)

    if action == "s":
        state = orch._model_browser.get(key)
        provider = payload or (state.provider if state else "")
        if provider not in PROVIDERS:
            return await model_selector_start(orch, key)
        return await _search_prompt(orch, key, provider)

    if action == "c":
        orch._model_browser.pop(key, None)
        return await model_selector_start(orch, key)

    if action == "m":
        return await _handle_model_selected(orch, key, payload)

    if action == "r":
        return await _handle_reasoning_selected(orch, key, effort=payload, model_id=extra)

    if action == "e":
        return await _handle_effort_selected(orch, key, effort=payload)

    if action == "b":
        if payload == "root":
            return await model_selector_start(orch, key)
        state = orch._model_browser.get(key)
        if state is not None and state.provider == payload:
            return await _show_models(orch, key, payload, query=state.query, page=state.page)
        return await _show_models(orch, key, payload)

    logger.warning("Unknown model selector callback: %s", data)
    return SelectorResponse(text=t("model.unknown_action"))


@dataclass(frozen=True)
class _SwitchContext:
    model_id: str
    new_provider: str
    is_topic: bool
    same_model: bool
    provider_changed: bool
    reasoning_effort: str | None


async def _persist_switch_target(
    orch: Orchestrator,
    key: SessionKey,
    active_session: SessionData | None,
    ctx: _SwitchContext,
) -> tuple[str, int]:
    """Persist the new target onto the next-message session; return resume state.

    Kills in-flight processes, writes the picked provider/model/effort onto the
    session the next message resolves, and returns that session's target-provider
    ``(session_id, message_count)`` for the resume hint.

    Reasoning effort is per-session (like model): a topic change touches only
    that topic's session; a main/DM change updates the global default (handled
    by the caller). On a provider switch, re-validate the carried-over effort
    against the new provider and reset to a safe default when it is unsupported
    (e.g. Claude ``max`` -> Codex) so an invalid value never reaches the CLI.
    """
    current_effort = (
        active_session.reasoning_effort
        if active_session and active_session.reasoning_effort
        else orch._config.reasoning_effort
    )
    effort = ctx.reasoning_effort
    if (
        effort is None
        and ctx.provider_changed
        and _validate_reasoning_effort(orch, ctx.model_id, current_effort) is not None
    ):
        effort = "medium"

    resume_source: SessionData | None = active_session
    if not ctx.same_model:
        await orch._process_registry.kill_by_chat_topic(key.chat_id, key.topic_id)
        if ctx.is_topic:
            # A topic never updates global config, so persist the picked target
            # onto the session the next message will actually resolve.
            # resolve_session_target creates it when missing and retargets or
            # replaces it when stale, so the chosen model+effort survive instead
            # of falling back to config on the next turn.
            resolved, created = await orch._sessions.resolve_session_target(
                key,
                provider=ctx.new_provider,
                model=ctx.model_id,
                reasoning_effort=effort,
            )
            resume_source = None if created else resolved
        elif active_session is not None:
            await orch._sessions.sync_session_target(
                active_session,
                provider=ctx.new_provider,
                model=ctx.model_id,
                reasoning_effort=effort,
            )
    elif ctx.is_topic and effort is not None:
        # effort-only change inside a topic: persist to the topic session only.
        if active_session is not None:
            await orch._sessions.sync_session_target(active_session, reasoning_effort=effort)
        else:
            await orch._sessions.resolve_session_target(
                key,
                provider=ctx.new_provider,
                model=ctx.model_id,
                reasoning_effort=effort,
            )

    return _resume_state_for_provider(resume_source, ctx.new_provider)


async def switch_model(
    orch: Orchestrator,
    key: SessionKey,
    model_id: str,
    *,
    reasoning_effort: str | None = None,
) -> str:
    """Execute model switch: kill topic processes, preserve sessions, persist config.

    Shared by ``/model <name>`` text command and the wizard callbacks.
    """
    is_topic = key.topic_id is not None
    active_session = await orch._sessions.get_active(key)

    # Compare against the active session's model when one exists (main DM or
    # topic). This keeps /model re-aligning a session whose model has diverged
    # from config.model (e.g. session opus, config gpt-5.5 -> /model gpt-5.5
    # still syncs the session), while the `if not is_topic:` block below keeps
    # updating the global default only from the main chat.
    old = active_session.model if active_session else orch._config.model
    same_model = old == model_id
    effort_only = same_model and reasoning_effort is not None

    # In a topic, same_model alone is a no-op. In the main chat a /model also
    # re-aligns the global default, so only short-circuit when config.model
    # already matches too (otherwise fall through to update it below).
    if same_model and reasoning_effort is None and (is_topic or orch._config.model == model_id):
        return t("model.already_running", model=model_id)

    old_provider = orch.models.provider_for(old)
    new_provider = orch.models.provider_for(model_id)
    provider_changed = old_provider != new_provider

    validation_error = _validate_reasoning_effort(orch, model_id, reasoning_effort)
    if validation_error is not None:
        return validation_error

    orch._model_browser.pop(key, None)

    resume_session_id, resume_message_count = await _persist_switch_target(
        orch,
        key,
        active_session,
        _SwitchContext(
            model_id=model_id,
            new_provider=new_provider,
            is_topic=is_topic,
            same_model=same_model,
            provider_changed=provider_changed,
            reasoning_effort=reasoning_effort,
        ),
    )

    if not is_topic:
        # Global config: update only from main chat / DM (not from topics).
        orch._config.model = model_id
        orch._cli_service.update_default_model(model_id)
        # Always realign provider to the new model's provider so config.model
        # and config.provider stay consistent. provider_changed is session-based
        # (used for session effort re-validation above) and can be False here
        # even when config.provider is stale (config had diverged from session).
        orch._config.provider = new_provider

        updates: dict[str, object] = {"model": model_id, "provider": new_provider}
        # Keep config.reasoning_effort consistent with the new config.model. Use
        # the explicit effort (e.g. /model wizard reasoning step), NOT the
        # session-mutated ``effort`` whose medium reset is for the session only.
        # Without an explicit value, re-validate the existing default against
        # model_id and reset to medium only when it is unsupported (e.g. config
        # carried Claude ``max`` while switching the default to Codex).
        if reasoning_effort is not None:
            orch._config.reasoning_effort = reasoning_effort
            orch._cli_service.update_reasoning_effort(reasoning_effort)
            updates["reasoning_effort"] = reasoning_effort
        elif _validate_reasoning_effort(orch, model_id, orch._config.reasoning_effort) is not None:
            orch._config.reasoning_effort = "medium"
            orch._cli_service.update_reasoning_effort("medium")
            updates["reasoning_effort"] = "medium"

        await update_config_file_async(orch.paths.config_path, **updates)

        # Sub-agent: also sync model/provider/effort to agents.json so the
        # registry stays current and survives restarts without merge hacks.
        if orch.paths.ductor_home.parent.name == "agents":
            agents_path = orch.paths.ductor_home.parent.parent / "agents.json"
            agent_name = orch._cli_service._config.agent_name
            registry_updates = dict(updates)
            # Codex and Claude both use reasoning_effort — only drop it when
            # switching to a provider that has no notion of effort.
            if (
                new_provider not in {"codex", "claude", "grok", "pi"}
                and "reasoning_effort" not in registry_updates
            ):
                registry_updates["reasoning_effort"] = None
            await asyncio.to_thread(
                update_agent_fields, agents_path, agent_name, **registry_updates
            )

    is_busy = orch.is_chat_busy(key.chat_id, key.topic_id)
    logger.info("Model switch model=%s provider=%s", model_id, orch._config.provider)

    summary = _build_switch_summary(
        _SwitchSummaryContext(
            old_model=old,
            new_model=model_id,
            old_provider=old_provider,
            new_provider=new_provider,
            provider_changed=provider_changed,
            reasoning_effort=reasoning_effort,
            effort_only=effort_only,
            resume_session_id=resume_session_id,
            resume_message_count=resume_message_count,
        )
    )
    if is_busy:
        summary += f"\n\n_{t('model.switched_while_busy')}_"
    return summary


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _status_line(orch: Orchestrator, key: SessionKey) -> str:
    """Current model + reasoning effort as a short header."""
    session = await orch._sessions.get_active(key)
    if session:
        model = session.model
        provider = session.provider
        # Effective (session-first) effort, matching /status and the runtime.
        effort = session.reasoning_effort or orch._config.reasoning_effort
    else:
        model, provider = orch.resolve_runtime_target(orch._config.model)
        effort = orch._config.reasoning_effort

    if provider in ("codex", "claude", "grok", "pi") and effort and effort != "default":
        current = (
            f"{t('model.header')}\n{t('model.current_with_effort', model=model, effort=effort)}"
        )
    else:
        current = f"{t('model.header')}\n{t('model.current', model=model)}"

    if key.topic_id is None:
        configured = orch._config.model
        if model != configured:
            current += f"\n{t('model.configured_default', configured=configured)}"

    return current


async def _handle_model_selected(
    orch: Orchestrator,
    key: SessionKey,
    model_id: str,
) -> SelectorResponse:
    """Handle a model button press.

    Codex and Claude show a reasoning sub-selector; other providers switch
    directly.
    """
    provider = orch.models.provider_for(model_id)

    if provider in ("gemini", "antigravity"):
        result = await switch_model(orch, key, model_id)
        return SelectorResponse(text=result)

    efforts = _supported_efforts(orch, model_id)
    if not efforts:
        result = await switch_model(orch, key, model_id)
        return SelectorResponse(text=result)

    buttons = [
        Button(
            text=_EFFORT_LABELS.get(e, e),
            callback_data=f"ms:r:{e}:{model_id}",
        )
        for e in efforts
    ]
    keyboard = ButtonGrid(
        rows=[
            *_rows_of(buttons),
            [Button(text=t("model.btn_back"), callback_data=f"ms:b:{provider}")],
        ]
    )

    header = await _status_line(orch, key)
    return SelectorResponse(
        text=f"{header}\n\n{t('model.thinking_level', model=model_id)}", buttons=keyboard
    )


async def _handle_reasoning_selected(
    orch: Orchestrator,
    key: SessionKey,
    *,
    effort: str,
    model_id: str,
) -> SelectorResponse:
    """Handle a reasoning effort button press. Final step: switch model + effort."""
    result = await switch_model(orch, key, model_id, reasoning_effort=effort)
    return SelectorResponse(text=result)


def _kill_interactive_repl(orch: Orchestrator, key: SessionKey) -> None:
    """Drop a pooled interactive REPL so it respawns with the new effort.

    Guarded: this base may predate interactive REPLs (#156), where
    ``cli_service`` has no ``kill_interactive_repl`` -- then this is a no-op.
    """
    repl_kill = getattr(orch._cli_service, "kill_interactive_repl", None)
    if repl_kill is not None:
        repl_kill(key.transport, key.chat_id, key.topic_id)


async def _handle_effort_selected(
    orch: Orchestrator,
    key: SessionKey,
    *,
    effort: str,
) -> SelectorResponse:
    """Apply a ``/effort`` selection to the active session only.

    Unlike the ``/model`` wizard, ``/effort`` is per-session: it sets the
    current session's reasoning effort (validated against the session's model,
    reset to ``medium`` when unsupported) and never changes the configured
    default model or effort.
    """
    session = await orch._sessions.get_active(key)
    if session is None:
        # No active session yet: create one scoped to the configured default
        # model and record the chosen effort on it. This stays per-session and
        # never mutates the configured default (no update_default_model /
        # update_reasoning_effort, no orch._config writes).
        model, _ = orch.resolve_runtime_target(orch._config.model)
        if _validate_reasoning_effort(orch, model, effort) is not None:
            effort = "medium"
        await orch._sessions.resolve_session(key, model=model, reasoning_effort=effort)
        _kill_interactive_repl(orch, key)
        orch._model_browser.pop(key, None)
        return SelectorResponse(text=t("model.effort_updated", effort=effort))

    if _validate_reasoning_effort(orch, session.model, effort) is not None:
        effort = "medium"
    orch._model_browser.pop(key, None)
    await orch._sessions.sync_session_target(session, reasoning_effort=effort)
    _kill_interactive_repl(orch, key)
    return SelectorResponse(text=t("model.effort_updated", effort=effort))
