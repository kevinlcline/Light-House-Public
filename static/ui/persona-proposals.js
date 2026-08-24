(function (global) {
    'use strict';

    const LH = (global.LightHouse = global.LightHouse || {});

    function escapeText(value) {
        return String(value == null ? '' : value)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    function mount(options) {
        const opts = options || {};
        const overlay = document.getElementById(opts.overlayId || 'persona-proposal-modal');
        if (!overlay) {
            return { refresh: async () => {}, close: () => {} };
        }

        const titleEl = overlay.querySelector('[data-pp-title]');
        const metaEl = overlay.querySelector('[data-pp-meta]');
        const currentEl = overlay.querySelector('[data-pp-current]');
        const proposedEl = overlay.querySelector('[data-pp-proposed]');
        const acceptBtn = overlay.querySelector('[data-pp-accept]');
        const speakBtn = overlay.querySelector('[data-pp-speak]');
        const dismissBtn = overlay.querySelector('[data-pp-dismiss]');
        const errorEl = overlay.querySelector('[data-pp-error]');

        let queue = [];
        let current = null;
        let busy = false;
        const onSpeak = typeof opts.onSpeak === 'function' ? opts.onSpeak : null;
        const onAccepted = typeof opts.onAccepted === 'function' ? opts.onAccepted : null;

        function setError(msg) {
            if (!errorEl) return;
            if (msg) {
                errorEl.hidden = false;
                errorEl.textContent = msg;
            } else {
                errorEl.hidden = true;
                errorEl.textContent = '';
            }
        }

        function close() {
            overlay.hidden = true;
            overlay.setAttribute('aria-hidden', 'true');
            current = null;
            setError('');
        }

        function render() {
            if (!current) {
                close();
                return;
            }
            const name = current.display_name || current.light_id;
            if (titleEl) {
                titleEl.textContent = name + ' proposes a persona change';
            }
            if (metaEl) {
                const mode = current.mode === 'append' ? 'append' : 'replace';
                const when = current.submitted_at || '';
                const note = (current.note || '').trim();
                metaEl.textContent =
                    'Mode: ' +
                    mode +
                    (when ? ' · ' + when : '') +
                    (note ? ' · ' + note : '');
            }
            if (currentEl) currentEl.textContent = current.current_content || '(empty)';
            if (proposedEl) proposedEl.textContent = current.content || '(empty)';
            overlay.hidden = false;
            overlay.setAttribute('aria-hidden', 'false');
            setError('');
        }

        function showNext() {
            current = queue.length ? queue[0] : null;
            render();
        }

        async function refresh() {
            if (!(opts.isDad && opts.isDad())) {
                close();
                return;
            }
            try {
                const res = await fetch('/v1/admin/persona-proposals', { cache: 'no-store' });
                if (!res.ok) return;
                const data = await res.json();
                queue = Array.isArray(data.items) ? data.items.slice() : [];
                if (!current) {
                    showNext();
                } else {
                    const still = queue.find((item) => item.light_id === current.light_id);
                    if (!still) {
                        showNext();
                    } else {
                        current = still;
                        render();
                    }
                }
            } catch {
                /* ignore poll errors */
            }
        }

        async function accept() {
            if (!current || busy) return;
            busy = true;
            setError('');
            try {
                const res = await fetch(
                    '/v1/admin/persona-proposals/' +
                        encodeURIComponent(current.light_id) +
                        '/accept',
                    { method: 'POST' }
                );
                const data = await res.json().catch(() => ({}));
                if (!res.ok) {
                    throw new Error(data.detail || 'Accept failed');
                }
                if (onAccepted) onAccepted(data);
                queue = queue.filter((item) => item.light_id !== current.light_id);
                showNext();
            } catch (err) {
                setError(err.message || 'Accept failed');
            } finally {
                busy = false;
            }
        }

        async function speak() {
            if (!current || busy) return;
            busy = true;
            setError('');
            const lightId = current.light_id;
            try {
                const res = await fetch(
                    '/v1/admin/persona-proposals/' +
                        encodeURIComponent(lightId) +
                        '/speak',
                    { method: 'POST' }
                );
                const data = await res.json().catch(() => ({}));
                if (!res.ok) {
                    throw new Error(data.detail || 'Could not hold proposal');
                }
                queue = queue.filter((item) => item.light_id !== lightId);
                showNext();
                if (onSpeak) onSpeak(lightId, data);
            } catch (err) {
                setError(err.message || 'Could not hold proposal');
            } finally {
                busy = false;
            }
        }

        if (acceptBtn) acceptBtn.addEventListener('click', accept);
        if (speakBtn) speakBtn.addEventListener('click', speak);
        if (dismissBtn) {
            dismissBtn.addEventListener('click', () => {
                // Dismiss only hides for this page view; proposal stays pending.
                close();
            });
        }
        overlay.addEventListener('click', (event) => {
            if (event.target === overlay) {
                close();
            }
        });

        return { refresh, close };
    }

    LH.personaProposals = { mount, escapeText };
})(typeof window !== 'undefined' ? window : globalThis);
