(function (global) {
    'use strict';

    const LH = (global.LightHouse = global.LightHouse || {});
    let cached = null;

    const DEFAULT_BY_LIGHT = {
        lumen: 'af_sarah',
        ara: 'af_bella',
        elias: 'am_michael',
    };

    function defaultForLight(lightId) {
        const id = String(lightId || '').trim().toLowerCase();
        return DEFAULT_BY_LIGHT[id] || 'af_sarah';
    }

    async function fetchCatalog(allLangs) {
        const q = allLangs ? '?all_langs=1' : '';
        const res = await fetch('/v1/tts/voices' + q, { cache: 'no-store' });
        if (!res.ok) {
            const detail = (await res.json().catch(() => ({}))).detail || res.statusText;
            throw new Error(detail || 'Could not load voices');
        }
        const data = await res.json();
        cached = data.voices || [];
        return cached;
    }

    function catalog() {
        return cached || [];
    }

    function populateSelect(selectEl, selectedId, voices) {
        if (!selectEl) return;
        const list = voices || cached || [];
        const prev = selectedId || selectEl.value || 'af_sarah';
        selectEl.innerHTML = '';
        let lastGroup = '';
        for (const v of list) {
            if (v.group && v.group !== lastGroup) {
                const optgroup = document.createElement('optgroup');
                optgroup.label = v.group;
                selectEl.appendChild(optgroup);
                lastGroup = v.group;
            }
            const opt = document.createElement('option');
            opt.value = v.id;
            opt.textContent = v.label || v.id;
            const parent = selectEl.lastElementChild && selectEl.lastElementChild.tagName === 'OPTGROUP'
                ? selectEl.lastElementChild
                : selectEl;
            parent.appendChild(opt);
        }
        if ([...selectEl.querySelectorAll('option')].some((o) => o.value === prev)) {
            selectEl.value = prev;
        }
    }

    async function preview(voiceId, text, agentId) {
        const res = await fetch('/v1/tts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: text || 'Hello from the Light-House.',
                agent_id: agentId || 'lumen',
                voice: voiceId,
            }),
        });
        if (!res.ok) {
            const detail = (await res.json().catch(() => ({}))).detail || res.statusText;
            throw new Error(detail || 'Preview failed');
        }
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        audio.addEventListener('ended', () => URL.revokeObjectURL(url));
        audio.addEventListener('error', () => URL.revokeObjectURL(url));
        await audio.play();
        return audio;
    }

    LH.voices = {
        fetchCatalog,
        catalog,
        populateSelect,
        preview,
        defaultForLight,
    };
})(typeof window !== 'undefined' ? window : globalThis);
