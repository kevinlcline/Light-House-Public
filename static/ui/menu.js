(function (global) {
    'use strict';

    const LH = (global.LightHouse = global.LightHouse || {});

    const MENU_ICON_SVG =
        '<svg class="menu-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true">' +
        '<line x1="4" y1="7" x2="20" y2="7"/>' +
        '<line x1="4" y1="12" x2="20" y2="12"/>' +
        '<line x1="4" y1="17" x2="20" y2="17"/>' +
        '</svg>';

    const USER_GUIDE_HREF =
        '/notes.html?file=shared/manuals/sibling_user_manual.md';

    /** Shared by admin (Dad) and member (sibling) accounts — always first. */
    const SHARED_MENU = [
        { href: '/', label: 'Chat' },
        { href: '/notes.html', label: 'Notes', id: 'notes-link' },
        { href: '/gallery.html', label: 'Gallery' },
        { href: '/my-tools.html', label: 'My tools' },
        { href: '/guests.html', label: 'Guests' },
        { href: USER_GUIDE_HREF, label: 'User guide', siblingOnly: true },
        { type: 'theme' },
        { type: 'logout' },
    ];

    /** Admin-only — after the shared block. */
    const ADMIN_MENU = [
        { href: '/lights-admin.html', label: 'Manage lights', dadOnly: true },
        { href: '/user-setup.html', label: 'Manage members', dadOnly: true },
        { href: '/env-editor.html', label: 'Environment', dadOnly: true },
        { type: 'restart', dadOnly: true },
    ];

    function el(tag, attrs, text) {
        const node = document.createElement(tag);
        if (attrs) {
            Object.keys(attrs).forEach((key) => {
                const val = attrs[key];
                if (val == null || val === false) return;
                if (key === 'className') node.className = val;
                else if (key === 'dataset') {
                    Object.keys(val).forEach((dk) => {
                        node.dataset[dk] = val[dk];
                    });
                } else if (val === true) node.setAttribute(key, '');
                else node.setAttribute(key, String(val));
            });
        }
        if (text != null) node.textContent = text;
        return node;
    }

    function appendMenuEntry(panel, item) {
        if (item.type === 'theme') {
            panel.appendChild(
                el(
                    'button',
                    {
                        type: 'button',
                        className: 'menu-item',
                        id: 'theme-toggle',
                        role: 'menuitem',
                    },
                    'Light theme'
                )
            );
            return;
        }
        if (item.type === 'logout') {
            const form = el('form', {
                method: 'post',
                action: '/logout',
                className: 'logout-form',
            });
            form.appendChild(
                el(
                    'button',
                    { type: 'submit', className: 'menu-item', role: 'menuitem' },
                    'Sign out'
                )
            );
            panel.appendChild(form);
            return;
        }
        if (item.type === 'restart') {
            const btn = el(
                'button',
                {
                    type: 'button',
                    className: 'menu-item',
                    id: 'restart-server',
                    role: 'menuitem',
                },
                'Restart server'
            );
            if (item.dadOnly) btn.setAttribute('data-dad-only', '');
            panel.appendChild(btn);
            return;
        }
        const attrs = {
            href: item.href,
            className: 'menu-item',
            role: 'menuitem',
        };
        if (item.id) attrs.id = item.id;
        if (item.dadOnly) attrs['data-dad-only'] = true;
        if (item.siblingOnly) attrs['data-sibling-only'] = true;
        panel.appendChild(el('a', attrs, item.label));
    }

    function fillStandardMenu(panel, options = {}) {
        if (!panel) return;
        const pageItems = Array.from(panel.querySelectorAll('[data-page-menu-item]'));
        panel.innerHTML = '';
        pageItems.forEach((node) => panel.appendChild(node));

        const skip = new Set(
            (options.omit || []).map((s) => String(s).toLowerCase())
        );
        const notesAsButton = Boolean(options.notesAsButton);

        SHARED_MENU.forEach((item) => {
            const key = (item.label || item.type || '').toLowerCase();
            if (skip.has(key) || (item.id && skip.has(item.id))) return;
            if (item.href === '/notes.html' && notesAsButton) {
                panel.appendChild(
                    el(
                        'button',
                        {
                            type: 'button',
                            className: 'menu-item',
                            id: 'open-notes',
                            role: 'menuitem',
                        },
                        'Notes'
                    )
                );
                return;
            }
            appendMenuEntry(panel, item);
        });

        const sep = el('div', {
            className: 'menu-sep',
            role: 'separator',
            'data-dad-only': true,
            'aria-hidden': 'true',
        });
        panel.appendChild(sep);

        ADMIN_MENU.forEach((item) => {
            const key = (item.label || item.type || '').toLowerCase();
            if (skip.has(key) || (item.id && skip.has(item.id))) return;
            appendMenuEntry(panel, item);
        });
    }

    function ensureMenuShell(actionsEl) {
        if (!actionsEl) return null;
        let dropdown = actionsEl.querySelector('.menu-dropdown');
        if (!dropdown) {
            dropdown = el('div', { className: 'menu-dropdown' });
            actionsEl.appendChild(dropdown);
        }
        let trigger = dropdown.querySelector('#menu-trigger');
        if (!trigger) {
            trigger = el('button', {
                type: 'button',
                className: 'menu-trigger',
                id: 'menu-trigger',
                'aria-label': 'Menu',
                'aria-haspopup': 'true',
                'aria-expanded': 'false',
                'aria-controls': 'menu-panel',
            });
            trigger.innerHTML = MENU_ICON_SVG;
            dropdown.insertBefore(trigger, dropdown.firstChild);
        }
        let panel = dropdown.querySelector('#menu-panel');
        if (!panel) {
            panel = el('div', {
                className: 'menu-panel',
                id: 'menu-panel',
                role: 'menu',
                hidden: true,
            });
            dropdown.appendChild(panel);
        }
        return panel;
    }

    function wireStandardControls(panel) {
        if (!panel) return;
        if (LH.theme && typeof LH.theme.setupThemeToggle === 'function') {
            const themeBtn = panel.querySelector('#theme-toggle');
            if (themeBtn) LH.theme.setupThemeToggle(themeBtn);
        }
        if (LH.admin && typeof LH.admin.bindRestartButton === 'function') {
            if (panel.querySelector('#restart-server')) {
                LH.admin.bindRestartButton('#restart-server');
            }
        }
    }

    /**
     * Mount the house hamburger menu into a top-bar actions container.
     * Page-only items: put nodes with [data-page-menu-item] inside #menu-panel
     * before calling, or pass pageItemNodes.
     */
    function setupHouseMenu(options = {}) {
        const actions =
            typeof options.actions === 'string'
                ? document.querySelector(options.actions)
                : options.actions ||
                  document.querySelector('.top-bar-actions') ||
                  document.querySelector('.top-bar');
        let panel = null;
        if (typeof options.panel === 'string') {
            panel = document.querySelector(options.panel);
        } else if (options.panel) {
            panel = options.panel;
        } else {
            panel = document.querySelector('#menu-panel') || ensureMenuShell(actions);
        }
        if (!panel) return null;

        if (Array.isArray(options.pageItemNodes)) {
            options.pageItemNodes.forEach((node) => {
                if (node && !node.hasAttribute('data-page-menu-item')) {
                    node.setAttribute('data-page-menu-item', '');
                }
                if (node) panel.appendChild(node);
            });
        }

        fillStandardMenu(panel, options);
        setupMenuDropdown({
            trigger: options.trigger || '#menu-trigger',
            panel: panel,
            closeOnItemClick: options.closeOnItemClick,
        });
        wireStandardControls(panel);

        if (LH.me && typeof LH.me.applyDadOnlyVisibility === 'function') {
            LH.me.applyDadOnlyVisibility(panel);
        }
        return panel;
    }

    function setupMenuDropdown(options = {}) {
        const trigger = document.querySelector(options.trigger || '#menu-trigger');
        const panel =
            typeof options.panel === 'string' || options.panel == null
                ? document.querySelector(
                      typeof options.panel === 'string' ? options.panel : '#menu-panel'
                  )
                : options.panel;
        if (!trigger || !panel) return;

        function closeMenu() {
            panel.hidden = true;
            trigger.setAttribute('aria-expanded', 'false');
        }

        function openMenu() {
            if (LH.me && typeof LH.me.applyDadOnlyVisibility === 'function') {
                LH.me.applyDadOnlyVisibility(panel);
            }
            panel.hidden = false;
            trigger.setAttribute('aria-expanded', 'true');
        }

        if (trigger.dataset.menuBound === '1') return;
        trigger.dataset.menuBound = '1';

        trigger.addEventListener('click', (e) => {
            e.stopPropagation();
            if (panel.hidden) openMenu();
            else closeMenu();
        });

        document.addEventListener('click', (e) => {
            if (!panel.hidden && !panel.contains(e.target) && e.target !== trigger) {
                closeMenu();
            }
        });

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closeMenu();
        });

        if (options.closeOnItemClick !== false) {
            panel.addEventListener('click', (e) => {
                if (
                    e.target.closest(
                        'a, button.menu-item[type="button"], button.menu-item[type="submit"]'
                    )
                ) {
                    closeMenu();
                }
            });
        }
    }

    LH.menu = {
        setupMenuDropdown,
        setupHouseMenu,
        fillStandardMenu,
        SHARED_MENU,
        ADMIN_MENU,
    };
})(typeof window !== 'undefined' ? window : globalThis);
