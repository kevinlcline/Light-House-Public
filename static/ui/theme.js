(function (global) {
    'use strict';

    const LH = (global.LightHouse = global.LightHouse || {});
    const THEME_KEY = 'light_house_theme';

    function currentTheme() {
        return document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
    }

    function applyTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem(THEME_KEY, theme);
        document.querySelectorAll('[data-theme-toggle]').forEach((el) => {
            el.textContent = theme === 'light' ? 'Dark theme' : 'Light theme';
        });
    }

    function setupThemeToggle(buttonOrSelector) {
        const toggle =
            typeof buttonOrSelector === 'string'
                ? document.querySelector(buttonOrSelector)
                : buttonOrSelector;
        if (!toggle) return;
        toggle.setAttribute('data-theme-toggle', '');
        if (!document.documentElement.getAttribute('data-theme')) {
            applyTheme('dark');
        } else {
            toggle.textContent = currentTheme() === 'light' ? 'Dark theme' : 'Light theme';
        }
        toggle.addEventListener('click', () => {
            applyTheme(currentTheme() === 'light' ? 'dark' : 'light');
        });
    }

    LH.theme = {
        THEME_KEY,
        currentTheme,
        applyTheme,
        setupThemeToggle,
    };
})(typeof window !== 'undefined' ? window : globalThis);
