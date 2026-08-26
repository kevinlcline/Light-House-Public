(function (global) {
    'use strict';

    const LH = (global.LightHouse = global.LightHouse || {});
    const GROUP_AGENT_ID = '__group__';

    function normalizeHex(value) {
        const raw = String(value || '').trim().toLowerCase();
        if (/^#[0-9a-f]{6}$/.test(raw)) return raw;
        if (/^[0-9a-f]{6}$/.test(raw)) return '#' + raw;
        return null;
    }

    function parseRgb(hex) {
        const n = parseInt(hex.slice(1), 16);
        return {
            r: (n >> 16) & 255,
            g: (n >> 8) & 255,
            b: n & 255,
        };
    }

    function toHex(r, g, b) {
        return (
            '#' +
            [r, g, b]
                .map((v) => {
                    const clamped = Math.max(0, Math.min(255, Math.round(v)));
                    return clamped.toString(16).padStart(2, '0');
                })
                .join('')
        );
    }

    function mixHex(a, b, amount) {
        const left = parseRgb(a);
        const right = parseRgb(b);
        const t = Math.max(0, Math.min(1, amount));
        return toHex(
            left.r + (right.r - left.r) * t,
            left.g + (right.g - left.g) * t,
            left.b + (right.b - left.b) * t
        );
    }

    function luminance(hex) {
        const { r, g, b } = parseRgb(hex);
        const lin = [r, g, b].map((c) => {
            const s = c / 255;
            return s <= 0.03928 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
        });
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2];
    }

    function bubbleStyleFromColor(hex) {
        const color = normalizeHex(hex);
        if (!color) return null;
        const theme =
            (document.documentElement.getAttribute('data-theme') || '').toLowerCase() ||
            'dark';
        const bg =
            theme === 'light'
                ? mixHex(color, '#ffffff', 0.78)
                : mixHex(color, '#0a0a0a', 0.58);
        const text = luminance(bg) > 0.45 ? '#1a1210' : '#f5efe6';
        return { background: bg, color: text };
    }

    function applyBubbleColor(el, hex) {
        if (!el) return;
        const style = bubbleStyleFromColor(hex);
        if (!style) return;
        el.style.background = style.background;
        el.style.color = style.color;
        el.classList.add('has-light-color');
    }

    function colorsMapFromList(list) {
        const map = {};
        for (const item of list || []) {
            if (!item || typeof item.id !== 'string') continue;
            const color = normalizeHex(item.color);
            if (color) map[item.id.toLowerCase()] = color;
        }
        return map;
    }

    function agentsFromApi(list, mapEntry, options) {
        const opts = options || {};
        const userId = opts.userId || '';
        const isDad = Boolean(opts.isDad);
        const map = {};
        for (const a of list) {
            if (!a || typeof a.id !== 'string') continue;
            const threadId = a.thread_id || a.id;
            // Dad keeps the legacy localStorage key (no user suffix).
            const historyKey =
                userId && !isDad
                    ? `light_house_chat_history_${threadId}__${userId}`
                    : `light_house_chat_history_${threadId}`;
            const base = {
                displayName: a.display_name || a.id,
                threadId,
                historyKey,
                allowed: a.allowed !== false,
                wantsKevin: Boolean(a.wants_kevin),
                wantsFamilyMeeting: Boolean(a.wants_family_meeting),
                familyMeetingTopic:
                    typeof a.family_meeting_topic === 'string'
                        ? a.family_meeting_topic
                        : '',
                color: normalizeHex(a.color),
            };
            map[a.id] = mapEntry ? mapEntry(a, base) : base;
        }
        return map;
    }

    async function fetchEnabled() {
        const res = await fetch('/v1/lights', { cache: 'no-store' });
        if (!res.ok) throw new Error('Could not load lights (' + res.status + ')');
        const data = await res.json();
        const list = (Array.isArray(data.lights) ? data.lights : []).filter(
            (l) => l && l.enabled !== false
        );
        if (list.length === 0) throw new Error('No enabled lights configured');
        return {
            list,
            primaryLightId: data.primary_light_id || list[0].id,
            agentOrder: list.map((l) => l.id),
            colors: colorsMapFromList(list),
        };
    }

    function populateSelect(selectEl, options) {
        const {
            agents = {},
            agentOrder = Object.keys(agents),
            agentId = '',
            error = null,
            includeGroup = false,
            groupId = GROUP_AGENT_ID,
            groupLabel = 'Group',
        } = options;

        selectEl.innerHTML = '';
        if (error) {
            const opt = document.createElement('option');
            opt.value = '';
            opt.textContent = error;
            selectEl.appendChild(opt);
            return;
        }
        if (includeGroup) {
            const groupOpt = document.createElement('option');
            groupOpt.value = groupId;
            groupOpt.textContent = groupLabel;
            selectEl.appendChild(groupOpt);
        }
        for (const id of agentOrder) {
            if (!agents[id]) continue;
            const opt = document.createElement('option');
            opt.value = id;
            const allowed = agents[id].allowed !== false;
            const knock = Boolean(agents[id].wantsKevin);
            let label = agents[id].displayName;
            if (!allowed) {
                label += ' (unavailable)';
            } else if (knock) {
                label += ' ·';
            }
            opt.textContent = label;
            opt.disabled = !allowed;
            if (knock && allowed) {
                opt.dataset.wantsKevin = '1';
            }
            if (!allowed) {
                opt.style.color = '#888';
            }
            selectEl.appendChild(opt);
        }
        const selectedAllowed =
            agents[agentId] && agents[agentId].allowed !== false;
        if (agentId && ((selectedAllowed) || (includeGroup && agentId === groupId))) {
            selectEl.value = agentId;
        } else {
            const firstAllowed = agentOrder.find(
                (id) => agents[id] && agents[id].allowed !== false
            );
            if (includeGroup) {
                selectEl.value = groupId;
            } else if (firstAllowed) {
                selectEl.value = firstAllowed;
            }
        }
    }

    LH.lights = {
        GROUP_AGENT_ID,
        agentsFromApi,
        fetchEnabled,
        populateSelect,
        normalizeHex,
        applyBubbleColor,
        colorsMapFromList,
        bubbleStyleFromColor,
    };
})(typeof window !== 'undefined' ? window : globalThis);
