(function (global) {
    'use strict';

    const LH = (global.LightHouse = global.LightHouse || {});

    const COPY_SVG =
        '<svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true" focusable="false">' +
        '<path fill="currentColor" d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/>' +
        '</svg>';

    function ensureActionsTray(messageEl) {
        if (!messageEl) return null;
        let tray = messageEl.querySelector('.message-actions');
        if (!tray) {
            tray = document.createElement('div');
            tray.className = 'message-actions';
            messageEl.appendChild(tray);
        }
        messageEl.classList.add('has-actions');
        return tray;
    }

    async function copyText(text) {
        const body = String(text || '');
        if (!body) return false;
        if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
            await navigator.clipboard.writeText(body);
            return true;
        }
        const ta = document.createElement('textarea');
        ta.value = body;
        ta.setAttribute('readonly', '');
        ta.style.position = 'fixed';
        ta.style.left = '-9999px';
        document.body.appendChild(ta);
        ta.select();
        const ok = document.execCommand('copy');
        document.body.removeChild(ta);
        return ok;
    }

    function attachCopyControl(messageEl, text) {
        try {
            if (!messageEl) return null;
            if (messageEl.querySelector('.message-copy')) return null;
            const body = String(text || '').trim();
            if (!body) return null;

            const tray = ensureActionsTray(messageEl);
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'message-action message-copy';
            btn.setAttribute('aria-label', 'Copy message');
            btn.setAttribute('title', 'Copy');
            btn.innerHTML = COPY_SVG;
            btn.addEventListener('click', async (ev) => {
                ev.preventDefault();
                ev.stopPropagation();
                try {
                    const ok = await copyText(body);
                    if (!ok) throw new Error('Copy failed');
                    btn.classList.add('is-copied');
                    btn.setAttribute('title', 'Copied');
                    btn.setAttribute('aria-label', 'Copied');
                    window.setTimeout(() => {
                        btn.classList.remove('is-copied');
                        btn.setAttribute('title', 'Copy');
                        btn.setAttribute('aria-label', 'Copy message');
                    }, 1200);
                } catch (err) {
                    console.warn('Copy failed:', err);
                    btn.setAttribute('title', 'Copy failed');
                }
            });
            tray.appendChild(btn);
            return btn;
        } catch (err) {
            console.warn('attachCopyControl failed:', err);
            return null;
        }
    }

    /**
     * Attach copy + optional speak controls for a bubble.
     * Speak is delegated to LightHouse.tts when available.
     */
    function attachBubbleActions(messageEl, text, options) {
        const opts = options || {};
        const body = String(text || '').trim();
        if (!messageEl || !body) return;
        attachCopyControl(messageEl, body);
        if (
            opts.speak !== false &&
            LH.tts &&
            typeof LH.tts.attachSpeakControl === 'function'
        ) {
            LH.tts.attachSpeakControl(
                messageEl,
                body,
                opts.agentId,
                opts.voice || null
            );
        }
    }

    LH.bubbleActions = {
        ensureActionsTray,
        attachCopyControl,
        attachBubbleActions,
        copyText,
    };
})(typeof window !== 'undefined' ? window : globalThis);
