(function (global) {
    'use strict';

    const LH = (global.LightHouse = global.LightHouse || {});
    const GROUP_AGENT_ID = '__group__';

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
    };
})(typeof window !== 'undefined' ? window : globalThis);
