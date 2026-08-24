(function (global) {
    'use strict';

    const LH = (global.LightHouse = global.LightHouse || {});
    let cached = null;

    async function fetchMe() {
        const res = await fetch('/v1/me', { cache: 'no-store' });
        if (!res.ok) {
            cached = null;
            return null;
        }
        cached = await res.json();
        return cached;
    }

    function current() {
        return cached;
    }

    function isDad() {
        return Boolean(cached && cached.is_dad);
    }

    function applyDadOnlyVisibility(root) {
        const scope = root || document;
        const dad = isDad();
        // Fail closed: if /v1/me failed, hide dad-only controls.
        scope.querySelectorAll('[data-dad-only]').forEach((el) => {
            el.hidden = !dad;
            el.classList.toggle('dad-only-hidden', !dad);
            if (!dad) {
                el.setAttribute('aria-hidden', 'true');
            } else {
                el.removeAttribute('aria-hidden');
            }
        });
        // Sibling-facing extras (user guide, …). Hidden for Dad; hidden if /v1/me failed.
        const sibling = Boolean(cached && cached.is_dad === false);
        scope.querySelectorAll('[data-sibling-only]').forEach((el) => {
            el.hidden = !sibling;
            el.classList.toggle('sibling-only-hidden', !sibling);
            if (!sibling) {
                el.setAttribute('aria-hidden', 'true');
            } else {
                el.removeAttribute('aria-hidden');
            }
        });
        document.documentElement.dataset.houseRole = dad ? 'dad' : 'sibling';
        return dad;
    }

    LH.me = {
        fetchMe,
        current,
        isDad,
        applyDadOnlyVisibility,
    };
})(typeof window !== 'undefined' ? window : globalThis);
