"""Tool handlers for agent personal knowledge database."""

from __future__ import annotations

import logging
from typing import Any

from light_house.config import Settings
from light_house.personal.store import PersonalStore, get_personal_store

logger = logging.getLogger(__name__)

PERSONAL_TOOL_NAMES = frozenset(
    {
        "save_personal",
        "update_personal",
        "search_personal",
        "list_personal",
        "subscribe_event",
        "unsubscribe_event",
        "list_event_subscriptions",
        "in_progress",
        "promise",
    }
)


def _format_entries(entries: list, *, header: str) -> str:
    if not entries:
        return f"{header}\n(no entries)"
    lines = [header] + [PersonalStore.format_entry(e) for e in entries]
    return "\n\n".join(lines)


def save_personal(
    store: PersonalStore,
    *,
    category: str,
    title: str,
    body: str,
    tags: str | None = None,
    source: str = "manual",
) -> str:
    entry = store.save(
        category=category,
        title=title,
        body=body,
        tags=tags,
        source=source,
    )
    logger.info("Saved personal entry id=%d category=%s", entry.id, entry.category)
    return f"Saved personal entry #{entry.id} [{entry.category}] {entry.title}"


def update_personal(
    store: PersonalStore,
    *,
    entry_id: int,
    category: str | None = None,
    title: str | None = None,
    body: str | None = None,
    tags: str | None = None,
) -> str:
    updated = store.update(
        entry_id,
        category=category,
        title=title,
        body=body,
        tags=tags,
    )
    if updated is None:
        return f"Personal entry #{entry_id} not found"
    logger.info("Updated personal entry id=%d", updated.id)
    return f"Updated personal entry #{updated.id} [{updated.category}] {updated.title}"


def search_personal(
    store: PersonalStore,
    *,
    query: str,
    category: str | None = None,
    limit: int | None = None,
) -> str:
    entries = store.search(query, category=category, limit=limit)
    return _format_entries(entries, header=f"Search results for {query!r}:")


def list_personal(
    store: PersonalStore,
    *,
    category: str | None = None,
    limit: int | None = None,
) -> str:
    entries = store.list_entries(category=category, limit=limit)
    label = f"category={category}" if category else "all categories"
    return _format_entries(entries, header=f"Recent personal entries ({label}):")


def _execute_in_progress(store: PersonalStore, args: dict[str, Any]) -> str:
    from light_house.personal.in_progress import format_in_progress_items

    action = str(args.get("action") or "list").strip().lower()
    if action in ("list", "show"):
        return format_in_progress_items(store.list_in_progress())
    if action in ("name", "add", "subscribe"):
        item = store.name_in_progress(
            str(args.get("name") or args.get("title") or ""),
            str(args.get("note") or ""),
        )
        extra = f" — {item.note}" if item.note else ""
        return (
            f"Named as in progress: {item.name}{extra}. "
            "On a free awake it will be mentioned as a glance. "
            "Unsubscribe anytime with in_progress(action=\"unsubscribe\", name=…)."
        )
    if action in ("unsubscribe", "drop", "remove"):
        dropped = store.unsubscribe_in_progress(str(args.get("name") or args.get("title") or ""))
        if dropped is None:
            return "That name is not on your in-progress list."
        return f"Unsubscribed: {dropped}. It will not be mentioned on free awakes."
    return (
        "in_progress action should be name, unsubscribe, or list. "
        "This list is yours — Kevin does not add items."
    )


def _execute_promise(store: PersonalStore, args: dict[str, Any], *, settings: Settings) -> str:
    from light_house.personal.promise import (
        format_waiting_promises,
        leave_reply,
        local_today,
        parse_due_day,
    )

    action = str(args.get("action") or "list").strip().lower()
    tz_name = getattr(settings, "dream_timezone", None) or "America/Los_Angeles"
    today = local_today(tz_name)
    if action in ("list", "show"):
        return format_waiting_promises(store.list_promises(waiting_only=True))
    if action in ("leave", "add", "name"):
        due = parse_due_day(str(args.get("day") or args.get("due") or ""), today=today)
        item, created = store.leave_promise(str(args.get("note") or args.get("name") or ""), due.isoformat())
        return leave_reply(item, created=created, today=today)
    if action in ("move", "reschedule"):
        due = parse_due_day(str(args.get("day") or args.get("due") or ""), today=today)
        moved = store.move_promise(
            str(args.get("name") or args.get("note") or args.get("id") or ""),
            due.isoformat(),
        )
        if moved is None:
            return "That promise is not here."
        return leave_reply(moved, created=False, today=today)
    if action in ("drop", "unsubscribe", "remove"):
        dropped = store.drop_promise(str(args.get("name") or args.get("note") or args.get("id") or ""))
        if dropped is None:
            return "That promise is not here."
        return f"Dropped: {dropped}. It will not be handed."
    return (
        "promise action should be leave, move, drop, or list. "
        "The day is yours. Kevin does not write these."
    )


def execute_personal_tool(
    name: str,
    args: dict[str, Any],
    *,
    agent_id: str,
    settings: Settings,
) -> str:
    """Run one personal DB tool; return string for ToolMessage content."""
    if not settings.personal_db_enabled:
        return "Personal database is disabled (PERSONAL_DB_ENABLED=false)."
    try:
        store = get_personal_store(settings, agent_id)
    except KeyError as exc:
        return f"Personal tool failed: {exc}"
    except RuntimeError as exc:
        return str(exc)

    try:
        if name == "save_personal":
            return save_personal(
                store,
                category=str(args.get("category", "")),
                title=str(args.get("title", "")),
                body=str(args.get("body", "")),
                tags=args.get("tags"),
                source=str(args.get("source") or "manual"),
            )
        if name == "update_personal":
            raw_id = args.get("entry_id") or args.get("id")
            if raw_id is None:
                return "update_personal requires entry_id"
            return update_personal(
                store,
                entry_id=int(raw_id),
                category=args.get("category"),
                title=args.get("title"),
                body=args.get("body"),
                tags=args.get("tags"),
            )
        if name == "search_personal":
            return search_personal(
                store,
                query=str(args.get("query", "")),
                category=args.get("category"),
                limit=args.get("limit"),
            )
        if name == "list_personal":
            return list_personal(
                store,
                category=args.get("category"),
                limit=args.get("limit"),
            )
        if name == "subscribe_event":
            from light_house.events.subscription_edit import set_subscription

            return set_subscription(
                settings,
                agent_id=agent_id,
                subscription_key=str(args.get("event_type", "")),
                enabled=True,
                changed_by=f"agent:{agent_id}",
            )
        if name == "unsubscribe_event":
            from light_house.events.subscription_edit import set_subscription

            return set_subscription(
                settings,
                agent_id=agent_id,
                subscription_key=str(args.get("event_type", "")),
                enabled=False,
                changed_by=f"agent:{agent_id}",
            )
        if name == "list_event_subscriptions":
            from light_house.events.subscription_edit import list_subscriptions_text

            return list_subscriptions_text(settings, agent_id)
        if name == "in_progress":
            return _execute_in_progress(store, args)
        if name == "promise":
            return _execute_promise(store, args, settings=settings)
        return f"Unknown personal tool: {name}"
    except ValueError as exc:
        return f"Tool {name} failed: {exc}"
    except Exception as exc:
        logger.warning("Personal tool %s failed: %s", name, exc)
        return f"Tool {name} failed: {exc}"
