"""CLI: send or list mailbox letters (filesystem + notify queue)."""

from __future__ import annotations

import argparse
import sys

from light_house.config import get_settings
from light_house.mailbox.letters import (
    REED_ID,
    list_letters_for,
    queue_notify,
    write_letter,
)
from light_house.mailbox.scheduler import ensure_mailbox_dirs


def _cmd_send(args: argparse.Namespace) -> int:
    settings = get_settings()
    ensure_mailbox_dirs(settings)
    body = args.body
    if args.body_file:
        body = open(args.body_file, encoding="utf-8").read()  # noqa: SIM115
    if body is None:
        body = sys.stdin.read()
    to_ids = [p.strip() for p in args.to.split(",") if p.strip()]
    letter = write_letter(
        from_id=args.from_id,
        to_ids=to_ids,
        subject=args.subject,
        body=body or "",
        settings=settings,
        private=bool(args.private),
        filename=args.filename,
    )
    queued = queue_notify(letter, settings=settings)
    print(f"Wrote {letter.path}")
    if queued:
        print(f"Notify queued: {queued.name} → {', '.join(letter.to_ids)}")
    else:
        print("No light notify (reed-only or no enabled light recipients).")
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    settings = get_settings()
    letters = list_letters_for(args.for_id, settings=settings, limit=args.limit)
    if not letters:
        print(f"No letters for {args.for_id}.")
        return 0
    print(f"{len(letters)} letter(s) for {args.for_id}:\n")
    for letter in letters:
        print(f"- {letter.created_at}  from={letter.from_id}  to={','.join(letter.to_ids)}")
        print(f"  subject: {letter.subject}")
        print(f"  path: {letter.path}")
        print()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m light_house.mailbox")
    sub = parser.add_subparsers(dest="cmd", required=True)

    send = sub.add_parser("send", help="Write a letter and notify addressed lights")
    send.add_argument("--from", dest="from_id", default=REED_ID)
    send.add_argument("--to", required=True, help="Comma-separated: lumen,ara,elias,all,reed")
    send.add_argument("--subject", required=True)
    send.add_argument("--body", default=None, help="Letter body (or use --body-file / stdin)")
    send.add_argument("--body-file", default=None)
    send.add_argument("--private", action="store_true", help="Single-light private mailbox")
    send.add_argument("--filename", default=None)
    send.set_defaults(func=_cmd_send)

    check = sub.add_parser("check", help="List letters for a recipient (Reed's mail ritual)")
    check.add_argument("--for", dest="for_id", default=REED_ID)
    check.add_argument("--limit", type=int, default=20)
    check.set_defaults(func=_cmd_check)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
