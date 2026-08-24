/** House guest presence + color Speak as (group + 1:1). */
(function (global) {
    const root = global.LightHouse || (global.LightHouse = {});

    /** Stable colors by guest slot (not purple). */
    const GUEST_COLORS = {
        'guest-1': '#2a9d8f',
        'guest-2': '#e76f51',
    };
    const FALLBACK_COLORS = ['#457b9d', '#e9c46a', '#6a994e'];

    async function fetchGuests() {
        const res = await fetch('/v1/house/guests', { cache: 'no-store' });
        if (!res.ok) throw new Error('Failed to load house guests');
        const data = await res.json();
        return Array.isArray(data.guests) ? data.guests : [];
    }

    function accountLabel(me) {
        return (me && (me.display_name || me.user_id)) || 'You';
    }

    function accountSpeakerId(me) {
        return (me && me.user_id) || 'you';
    }

    function colorForGuest(speakerId, index) {
        if (GUEST_COLORS[speakerId]) return GUEST_COLORS[speakerId];
        return FALLBACK_COLORS[index % FALLBACK_COLORS.length];
    }

    function readStoredSpeaker(storageKey) {
        if (!storageKey) return '';
        try {
            return sessionStorage.getItem(storageKey) || '';
        } catch {
            return '';
        }
    }

    function writeStoredSpeaker(storageKey, speakerId) {
        if (!storageKey) return;
        try {
            if (speakerId) sessionStorage.setItem(storageKey, speakerId);
            else sessionStorage.removeItem(storageKey);
        } catch {
            /* ignore */
        }
    }

    function escapeText(text) {
        return String(text || '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    /**
     * Mount color-toggle speak UI on a status bar + input.
     * Tap a guest name to select; tap again to return to account holder.
     */
    function mountSpeakBar(options) {
        const statusEl = options.statusEl;
        const inputEl = options.inputEl;
        const storageKey = options.storageKey || '';
        const onChange = typeof options.onChange === 'function' ? options.onChange : null;
        let me = options.me || null;
        let guests = Array.isArray(options.guests) ? options.guests.slice() : [];
        let selectedId = '';

        function guestIds() {
            return new Set(
                (guests || [])
                    .filter((g) => g && g.speaker_id && g.display_name)
                    .map((g) => g.speaker_id)
            );
        }

        function syncSelectionFromStorage() {
            const stored = readStoredSpeaker(storageKey);
            const allowed = guestIds();
            selectedId = stored && allowed.has(stored) ? stored : '';
        }

        function selectedSpeaker() {
            if (selectedId) {
                const hit = (guests || []).find((g) => g.speaker_id === selectedId);
                if (hit && hit.display_name) {
                    return { speaker_id: hit.speaker_id, display_name: hit.display_name };
                }
            }
            return {
                speaker_id: accountSpeakerId(me),
                display_name: accountLabel(me),
            };
        }

        function selectedColor() {
            if (!selectedId) return '';
            const idx = (guests || []).findIndex((g) => g.speaker_id === selectedId);
            return colorForGuest(selectedId, idx < 0 ? 0 : idx);
        }

        function applyInputChrome() {
            if (!inputEl) return;
            const color = selectedColor();
            if (color) {
                inputEl.style.setProperty('--speak-border', color);
                inputEl.style.borderColor = color;
                inputEl.style.boxShadow = '0 0 0 1px ' + color + '55';
                inputEl.dataset.speakColor = color;
            } else {
                inputEl.style.removeProperty('--speak-border');
                inputEl.style.borderColor = '';
                inputEl.style.boxShadow = '';
                delete inputEl.dataset.speakColor;
            }
        }

        function emitChange() {
            applyInputChrome();
            if (onChange) onChange(selectedSpeaker());
        }

        function toggleGuest(speakerId) {
            if (!guestIds().has(speakerId)) return;
            selectedId = selectedId === speakerId ? '' : speakerId;
            writeStoredSpeaker(storageKey, selectedId);
            render();
            emitChange();
        }

        function render() {
            if (!statusEl) return;
            const list = (guests || []).filter((g) => g && g.speaker_id && g.display_name);
            const manage =
                '<a class="linkish house-guest-manage" href="/guests.html">Manage guests</a>';

            if (list.length === 0) {
                statusEl.innerHTML = 'No guests signed in · ' + manage;
                selectedId = '';
                writeStoredSpeaker(storageKey, '');
                applyInputChrome();
                return;
            }

            const chips = list
                .map((g, index) => {
                    const color = colorForGuest(g.speaker_id, index);
                    const active = g.speaker_id === selectedId;
                    const pressed = active ? 'true' : 'false';
                    const cls =
                        'house-guest-chip' + (active ? ' is-selected' : '');
                    return (
                        '<button type="button" class="' +
                        cls +
                        '" data-speaker-id="' +
                        escapeText(g.speaker_id) +
                        '" style="--guest-color:' +
                        color +
                        '" aria-pressed="' +
                        pressed +
                        '">' +
                        escapeText(g.display_name) +
                        '</button>'
                    );
                })
                .join('');

            statusEl.innerHTML =
                '<span class="house-status-label">Present</span> ' +
                '<span class="house-guest-chips">' +
                chips +
                '</span> · ' +
                manage;
        }

        function onStatusClick(event) {
            const btn = event.target.closest('.house-guest-chip');
            if (!btn || !statusEl.contains(btn)) return;
            event.preventDefault();
            toggleGuest(btn.getAttribute('data-speaker-id') || '');
        }

        statusEl.addEventListener('click', onStatusClick);

        function setMe(nextMe) {
            me = nextMe;
        }

        function setGuests(nextGuests) {
            guests = Array.isArray(nextGuests) ? nextGuests.slice() : [];
            syncSelectionFromStorage();
            // Drop selection if that guest signed out.
            if (selectedId && !guestIds().has(selectedId)) {
                selectedId = '';
                writeStoredSpeaker(storageKey, '');
            }
            render();
            emitChange();
        }

        syncSelectionFromStorage();
        render();
        emitChange();

        return {
            setMe,
            setGuests,
            selectedSpeaker,
            selectedColor,
            render,
        };
    }

    // Backward-compatible thin wrappers (unused once pages migrate).
    function selectedSpeaker(selectEl, me, guests) {
        const value = (selectEl && selectEl.value) || accountSpeakerId(me);
        if (value === accountSpeakerId(me)) {
            return { speaker_id: accountSpeakerId(me), display_name: accountLabel(me) };
        }
        const hit = (guests || []).find((g) => g.speaker_id === value);
        if (hit && hit.display_name) {
            return { speaker_id: hit.speaker_id, display_name: hit.display_name };
        }
        return { speaker_id: accountSpeakerId(me), display_name: accountLabel(me) };
    }

    function populateSpeakAs() {}
    function renderStatusBar(el, guests) {
        if (!el) return;
        if (!guests || guests.length === 0) {
            el.innerHTML =
                'No guests signed in · <a class="linkish" href="/guests.html">Manage guests</a>';
        }
    }

    root.houseGuests = {
        fetchGuests,
        accountLabel,
        accountSpeakerId,
        colorForGuest,
        mountSpeakBar,
        selectedSpeaker,
        populateSpeakAs,
        renderStatusBar,
    };
})(typeof window !== 'undefined' ? window : globalThis);
