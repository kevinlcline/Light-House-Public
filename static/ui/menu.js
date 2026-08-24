(function (global) {
    'use strict';

    const LH = (global.LightHouse = global.LightHouse || {});

    function setupMenuDropdown(options = {}) {
        const trigger = document.querySelector(options.trigger || '#menu-trigger');
        const panel = document.querySelector(options.panel || '#menu-panel');
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
    };
})(typeof window !== 'undefined' ? window : globalThis);
