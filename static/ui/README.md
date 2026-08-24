# Light-House shared UI

Static assets used by the web interface (`index.html`, `notes.html`, `group.html`, …).

## Layout

| File | Purpose |
|------|---------|
| `theme-init.js` | Apply saved light/dark theme before first paint (include in `<head>`) |
| `theme.css` | Design tokens (colors, bubbles, markdown surfaces) |
| `base.css` | Page shell (`html`/`body`, layout helpers) |
| `layout.css` | Top bar, hamburger menu, agent select, icon buttons |
| `chat.css` | Chat container, messages, bubbles, compose area |
| `markdown.css` | Shared markdown typography |
| `notes.css` | Notes explorer tree and editor |
| `theme.js` | `LightHouse.theme` — theme toggle |
| `lights.js` | `LightHouse.lights` — fetch `/v1/lights`, populate agent `<select>` |
| `markdown.js` | `LightHouse.markdown` — sanitize markdown via marked + DOMPurify |
| `menu.js` | `LightHouse.menu` — hamburger dropdown behavior |
| `admin.js` | `LightHouse.admin` — server restart helper |

## Usage in a page

```html
<head>
    <script src="/static/ui/theme-init.js"></script>
    <link rel="stylesheet" href="/static/ui/theme.css">
    <link rel="stylesheet" href="/static/ui/base.css">
    <link rel="stylesheet" href="/static/ui/layout.css">
    <!-- page-specific: chat.css and/or notes.css -->
</head>
<body class="layout-locked">
    ...
    <script src="/static/ui/theme.js"></script>
    <script src="/static/ui/lights.js"></script>
    <!-- other modules as needed -->
    <script>
        LightHouse.theme.setupThemeToggle('#theme-toggle');
        LightHouse.menu.setupMenuDropdown();
    </script>
</body>
```

## Conventions

- **One source of truth** for shared behavior — do not copy menu/lights/theme code into HTML pages.
- **Page files stay thin** — only page-specific layout and logic.
- **Body classes**: `layout-locked` prevents document scroll; `layout-column` for notes-style column layouts.
- **Tests**: `tests/ui/test_static_ui.py` checks asset presence and HTML wiring.

## Future work

- Migrate admin pages (`lights-admin.html`, `env-editor.html`, …) to the same tokens and modules.
- Optional: lightweight build step (esbuild) if we outgrow plain script tags.
