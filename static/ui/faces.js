(function (global) {
    'use strict';

    const LH = (global.LightHouse = global.LightHouse || {});
    const STAGE_KEY = 'light_house_face_stage';

    const PALETTES = {
        lumen: { head: '#c4a574', eye: '#2a2218', mouth: '#5c4030' },
        ara: { head: '#c9a3c4', eye: '#2e1f33', mouth: '#6b3d62' },
        elias: { head: '#88b5a4', eye: '#1c2e28', mouth: '#2f5348' },
        echo: { head: '#9b93b8', eye: '#241e33', mouth: '#4a4260' },
    };
    const FALLBACK = { head: '#9a9a9a', eye: '#222', mouth: '#444' };
    /** House defaults until /v1/tts/status voices arrive (match lights.yaml). */
    const DEFAULT_VOICES = {
        lumen: 'af_heart',
        ara: 'bf_isabella',
        elias: 'am_echo',
        echo: 'am_echo',
    };
    const POSES = [
        'smile',
        'laugh',
        'blush',
        'sad',
        'soft',
        'surprise',
        'excited',
        'think',
        'wink',
        'sigh',
        'anger',
        'kiss',
        'pause',
        'pause_smile',
    ];
    const GESTURES = ['nod', 'tilt'];
    const OVERLAYS = ['bright'];
    const EMO_HOLD_MS = 2200;
    const PAUSE_HOLD_MS = 2000;
    const PAUSE_LONG_HOLD_MS = 4000;
    // Keep in sync with src/light_house/tts/stage_cues.py
    const CUE_RE = /\*([^*]{1,160})\*|_([^_]{1,80})_|\(([^)]{1,40})\)/g;
    const SKIP_THINK_RE = /\b(i|we|you|they)\s+think\b/i;
    const BRIGHT_RE =
        /\b(?:eyes?\s+)?(?:light(?:s|ing)?\s+up|glow(?:s|ing)?|gleam(?:s|ing)?|sparkle[sd]?)\b/i;
    const PAUSE_RE = /\b(pauses?|pausing|stillness|still)\b/i;
    const LONG_PAUSE_RE = /\b(long|full)\b/i;
    const PAUSE_SMILE_RE =
        /\b(softens?|softly|soft|gentle(?:ly)?|quiet(?:ly)?|tender(?:ly)?|warm(?:ly)?|hushed)\b/i;
    const RULES = [
        { name: 'laugh', re: /\b(laughs?|laughing|giggles?|giggling|chuckles?|cackles?)\b/i, kind: 'pose' },
        { name: 'wink', re: /\b(winks?|winking)\b/i, kind: 'pose' },
        { name: 'blush', re: /\b(blush(?:es|ing)?|shy|embarrassed|flustered)\b/i, kind: 'pose' },
        { name: 'sad', re: /\b(sads?|sadly|tears?|cries|crying|weeps?|heartbroken)\b/i, kind: 'pose' },
        {
            name: 'anger',
            re: /\b(angr(?:y|ily)|anger|scowls?|scowling|glares?|glaring|furious|fumes?|fuming)\b/i,
            kind: 'pose',
        },
        {
            name: 'kiss',
            re: /\b(kisses?|kissing|blows?\s+a\s+kiss|blown?\s+kiss|air\s+kiss)\b/i,
            kind: 'pose',
        },
        {
            name: 'excited',
            re: /\b(excit(?:ed|edly|ement)|thrilled|eager(?:ly)?|elated)\b/i,
            kind: 'pose',
        },
        { name: 'surprise', re: /\b(surprise[ds]?|gasps?|startled|wide-eyed|astonished)\b/i, kind: 'pose' },
        { name: 'think', re: /\b(thinks?|thinking|ponders?|thoughtful|hmm+)\b/i, kind: 'pose' },
        { name: 'sigh', re: /\b(sighs?|sighing|weary|exhales?)\b/i, kind: 'pose' },
        { name: 'nod', re: /\b(nods?|nodding)\b/i, kind: 'gesture' },
        {
            name: 'tilt',
            re: /\b(tilts?|tilting|cocks?\s+(?:her|his|their|a)\s+head|head\s+tilt)\b/i,
            kind: 'gesture',
        },
        {
            name: 'smile',
            re: /\b(smiles?|smiling|grins?|grinning|beams?|beaming|warmly)\b/i,
            kind: 'pose',
        },
        {
            name: 'soft',
            re: /\b(softens?|softly|gently|quietly|tender(?:ly)?|whispers?|whispering|hushed)\b/i,
            kind: 'pose',
        },
    ];
    const EMOJI_MAP = {
        '😮‍💨': { pose: 'sigh' },
        '🤷‍♀️': { gesture: 'tilt' },
        '🤷‍♂️': { gesture: 'tilt' },
        '☺️': { pose: 'smile' },
        '☹️': { pose: 'sad' },
        '❤️': { pose: 'smile' },
        '😂': { pose: 'laugh' },
        '🤣': { pose: 'laugh' },
        '😆': { pose: 'laugh' },
        '😅': { pose: 'laugh' },
        '😹': { pose: 'laugh' },
        '😊': { pose: 'smile' },
        '🙂': { pose: 'smile' },
        '😄': { pose: 'smile' },
        '😃': { pose: 'smile' },
        '😀': { pose: 'smile' },
        '😁': { pose: 'smile' },
        '🥰': { pose: 'smile' },
        '😍': { pose: 'smile' },
        '🤗': { pose: 'smile' },
        '😇': { pose: 'smile' },
        '😻': { pose: 'smile' },
        '💕': { pose: 'smile' },
        '💖': { pose: 'smile' },
        '💗': { pose: 'smile' },
        '❤': { pose: 'smile' },
        '☺': { pose: 'smile' },
        '😳': { pose: 'blush' },
        '🤭': { pose: 'blush' },
        '🙈': { pose: 'blush' },
        '😢': { pose: 'sad' },
        '😭': { pose: 'sad' },
        '😔': { pose: 'sad' },
        '😞': { pose: 'sad' },
        '☹': { pose: 'sad' },
        '💔': { pose: 'sad' },
        '😿': { pose: 'sad' },
        '🥺': { pose: 'sad' },
        '😮': { pose: 'surprise' },
        '😲': { pose: 'surprise' },
        '😯': { pose: 'surprise' },
        '🤯': { pose: 'surprise' },
        '😱': { pose: 'surprise' },
        '🙀': { pose: 'surprise' },
        '🤩': { pose: 'excited' },
        '✨': { pose: 'excited' },
        '🎉': { pose: 'excited' },
        '🤔': { pose: 'think' },
        '💭': { pose: 'think' },
        '🧐': { pose: 'think' },
        '😉': { pose: 'wink' },
        '😜': { pose: 'wink' },
        '😝': { pose: 'wink' },
        '😋': { pose: 'wink' },
        '😏': { pose: 'wink' },
        '😪': { pose: 'sigh' },
        '😩': { pose: 'sigh' },
        '😫': { pose: 'sigh' },
        '🤫': { pose: 'soft' },
        '😌': { pose: 'soft' },
        '🌸': { pose: 'soft' },
        '😠': { pose: 'anger' },
        '😡': { pose: 'anger' },
        '🤬': { pose: 'anger' },
        '😤': { pose: 'anger' },
        '😘': { pose: 'kiss' },
        '💋': { pose: 'kiss' },
        '😗': { pose: 'kiss' },
        '😙': { pose: 'kiss' },
        '😚': { pose: 'kiss' },
        '👍': { gesture: 'nod' },
        '👌': { gesture: 'nod' },
        '🤷': { gesture: 'tilt' },
        '😕': { gesture: 'tilt' },
    };
    const EMOJI_RE = new RegExp(
        Object.keys(EMOJI_MAP)
            .sort((a, b) => b.length - a.length)
            .map((emoji) => emoji.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
            .join('|'),
        'g'
    );

    let stageEl = null;
    let presentIds = new Set();
    let speakingId = '';
    /** 0..1 speech energy → CSS --mouth-open on the talking wrap. */
    let mouthOpen = 0;
    let lipSyncActive = false;
    /** User preference: show the face stage when faces are present. Default on. */
    let stageVisible = true;
    try {
        stageVisible = localStorage.getItem(STAGE_KEY) !== '0';
    } catch {
        stageVisible = true;
    }
    /** @type {Record<string, string>} */
    const voiceIds = Object.assign({}, DEFAULT_VOICES);
    /** @type {Record<string, string>} */
    const poses = {};
    /** @type {Record<string, string>} */
    const gestures = {};
    /** @type {Record<string, string>} */
    const overlays = {};
    let emotionClearTimer = 0;
    let emotionHoldMs = EMO_HOLD_MS;
    /** @type {{ id: string, steps: Array<{ at: number, classified: object }>, stepIndex: number } | null} */
    let speakPlan = null;
    /** @type {Record<string, { bob: number, tilt: number }>} */
    const idleTimers = {};

    function paletteFor(agentId) {
        const id = String(agentId || '').toLowerCase();
        return PALETTES[id] || FALLBACK;
    }

    /** Presentation gender from Kokoro voice id prefix (af_/bf_ girl, am_/bm_ boy). */
    function genderFromVoice(voiceId) {
        const m = String(voiceId || '')
            .trim()
            .toLowerCase()
            .match(/^([ab][mf])[_-]/);
        if (!m) return '';
        return m[1].charAt(1) === 'f' ? 'girl' : 'boy';
    }

    function genderFor(agentId) {
        const id = String(agentId || '').toLowerCase();
        return genderFromVoice(voiceIds[id] || DEFAULT_VOICES[id] || '');
    }

    function svgFace(agentId) {
        const p = paletteFor(agentId);
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('class', 'light-face');
        // Room for a hair bow past the rim and a bow-tie below the chin.
        svg.setAttribute('viewBox', '-4 0 74 76');
        svg.setAttribute('aria-hidden', 'true');
        const girlBow = '#f0a0b8';
        const boyBow = '#d32f2f';
        svg.innerHTML =
            '<circle class="face-halo face-halo-outer" cx="32" cy="32" r="34" fill="' +
            p.head +
            '" aria-hidden="true"></circle>' +
            '<circle class="face-halo" cx="32" cy="32" r="31" fill="' +
            p.head +
            '" aria-hidden="true"></circle>' +
            '<circle class="face-head" cx="32" cy="32" r="28" fill="' +
            p.head +
            '"></circle>' +
            // Hair bow: tips meet at knot on the face rim at 1:30 (pink).
            '<g class="face-bow face-bow-hair" aria-hidden="true">' +
            '<polygon points="52,12 43,5 43,19" fill="' +
            girlBow +
            '"></polygon>' +
            '<polygon points="52,12 61,5 61,19" fill="' +
            girlBow +
            '"></polygon>' +
            '<circle cx="52" cy="12" r="2.4" fill="' +
            girlBow +
            '"></circle>' +
            '</g>' +
            '<ellipse class="face-cheek face-cheek-l" cx="18" cy="38" rx="6.5" ry="4.2"></ellipse>' +
            '<ellipse class="face-cheek face-cheek-r" cx="46" cy="38" rx="6.5" ry="4.2"></ellipse>' +
            '<path class="face-cheek-arc face-cheek-arc-l" d="M12 37 Q18 30 24 37" fill="none" stroke="#e0899a" stroke-width="2.2" stroke-linecap="round"></path>' +
            '<path class="face-cheek-arc face-cheek-arc-r" d="M40 37 Q46 30 52 37" fill="none" stroke="#e0899a" stroke-width="2.2" stroke-linecap="round"></path>' +
            '<path class="face-brow face-brow-l" d="M15 20 L27 23" fill="none" stroke="' +
            p.eye +
            '" stroke-width="2.4" stroke-linecap="round"></path>' +
            '<path class="face-brow face-brow-r" d="M49 20 L37 23" fill="none" stroke="' +
            p.eye +
            '" stroke-width="2.4" stroke-linecap="round"></path>' +
            '<circle class="face-eye face-eye-l" cx="22" cy="26" r="3.2" fill="' +
            p.eye +
            '"></circle>' +
            '<circle class="face-eye face-eye-r" cx="42" cy="26" r="3.2" fill="' +
            p.eye +
            '"></circle>' +
            '<path class="face-eye-closed face-eye-closed-l" d="M16 26 L28 26" fill="none" stroke="' +
            p.eye +
            '" stroke-width="2.6" stroke-linecap="round"></path>' +
            '<path class="face-eye-closed face-eye-closed-r" d="M36 26 L48 26" fill="none" stroke="' +
            p.eye +
            '" stroke-width="2.6" stroke-linecap="round"></path>' +
            '<circle class="face-gleam face-gleam-l" cx="21" cy="24.6" r="1.15" fill="#fff8ee"></circle>' +
            '<circle class="face-gleam face-gleam-r" cx="41" cy="24.6" r="1.15" fill="#fff8ee"></circle>' +
            '<ellipse class="face-mouth" cx="32" cy="44" rx="8" ry="6" fill="' +
            p.mouth +
            '"></ellipse>' +
            '<circle class="face-mouth-kiss" cx="32" cy="44" r="2.2" fill="' +
            p.mouth +
            '"></circle>' +
            '<path class="face-mouth-arc face-mouth-smile" d="M22 41 Q32 52 42 41" fill="none" stroke="' +
            p.mouth +
            '" stroke-width="2.6" stroke-linecap="round"></path>' +
            '<path class="face-mouth-arc face-mouth-frown" d="M22 48 Q32 39 42 48" fill="none" stroke="' +
            p.mouth +
            '" stroke-width="2.6" stroke-linecap="round"></path>' +
            // Bow tie: tips meet at knot below the chin at 6 o'clock (red).
            '<g class="face-bow face-bow-tie" aria-hidden="true">' +
            '<polygon points="32,64 16,54 16,74" fill="' +
            boyBow +
            '"></polygon>' +
            '<polygon points="32,64 48,54 48,74" fill="' +
            boyBow +
            '"></polygon>' +
            '<circle cx="32" cy="64" r="3.4" fill="' +
            boyBow +
            '"></circle>' +
            '</g>';
        return svg;
    }

    function clearIdle(id) {
        const timers = idleTimers[id];
        if (!timers) return;
        if (timers.bob) clearTimeout(timers.bob);
        if (timers.tilt) clearTimeout(timers.tilt);
        delete idleTimers[id];
    }

    function armIdle(wrap, id) {
        clearIdle(id);
        if (!wrap) return;
        const timers = { bob: 0, tilt: 0 };
        idleTimers[id] = timers;

        const scheduleBob = () => {
            timers.bob = setTimeout(() => {
                const pose = poses[id] || '';
                if (
                    !wrap.isConnected ||
                    gestures[id] ||
                    pose === 'pause' ||
                    pose === 'pause_smile'
                ) {
                    scheduleBob();
                    return;
                }
                wrap.classList.add('is-idle-bob');
                setTimeout(() => wrap.classList.remove('is-idle-bob'), 700);
                scheduleBob();
            }, 5200 + Math.random() * 7400);
        };
        const scheduleTilt = () => {
            timers.tilt = setTimeout(() => {
                const pose = poses[id] || '';
                if (
                    !wrap.isConnected ||
                    gestures[id] ||
                    pose === 'pause' ||
                    pose === 'pause_smile'
                ) {
                    scheduleTilt();
                    return;
                }
                wrap.classList.add('is-idle-tilt');
                setTimeout(() => wrap.classList.remove('is-idle-tilt'), 900);
                scheduleTilt();
            }, 6800 + Math.random() * 9200);
        };
        scheduleBob();
        scheduleTilt();
    }

    function ensureWrap(agentId) {
        if (!stageEl) return null;
        const id = String(agentId || 'lumen').toLowerCase();
        let wrap = stageEl.querySelector('[data-face="' + id + '"]');
        if (wrap) return wrap;
        wrap = document.createElement('div');
        wrap.className = 'light-face-wrap';
        wrap.dataset.face = id;
        const palette = PALETTES[id] || FALLBACK;
        wrap.style.setProperty('--face-glow', palette.head);
        wrap.appendChild(svgFace(id));
        const name = document.createElement('div');
        name.className = 'light-face-name';
        name.textContent = id.charAt(0).toUpperCase() + id.slice(1);
        wrap.appendChild(name);
        stageEl.appendChild(wrap);
        applyStageVisibility();
        armIdle(wrap, id);
        return wrap;
    }

    function applyFaceClasses(el, pose, gesture, overlay, talking) {
        if (!el) return;
        el.classList.toggle('is-talking', talking);
        POSES.forEach((name) => {
            el.classList.toggle('emo-' + name, pose === name);
        });
        GESTURES.forEach((name) => {
            el.classList.toggle('emo-' + name, gesture === name);
        });
        OVERLAYS.forEach((name) => {
            el.classList.toggle('emo-' + name, overlay === name);
        });
    }

    function forgetAgent(id) {
        const key = String(id || '').toLowerCase();
        if (!key) return;
        clearIdle(key);
        delete poses[key];
        delete gestures[key];
        delete overlays[key];
        if (speakingId === key) {
            speakingId = '';
            speakPlan = null;
        }
    }

    /** Drop faces that are no longer in the stage cast (1:1 replace, Group refresh). */
    function pruneTo(keepIds) {
        const keep = new Set(
            (keepIds || []).map((id) => String(id || '').toLowerCase()).filter(Boolean)
        );
        if (speakingId && keep.has(speakingId) === false) {
            // Keep the currently speaking face visible until audio ends,
            // even if the cast list briefly omitted them.
            keep.add(speakingId);
        }
        if (stageEl) {
            stageEl.querySelectorAll('.light-face-wrap').forEach((wrap) => {
                const id = wrap.dataset.face;
                if (keep.has(id)) return;
                forgetAgent(id);
                wrap.remove();
            });
        }
        Object.keys(poses).forEach((id) => {
            if (!keep.has(id)) forgetAgent(id);
        });
        Object.keys(gestures).forEach((id) => {
            if (!keep.has(id)) forgetAgent(id);
        });
        Object.keys(overlays).forEach((id) => {
            if (!keep.has(id)) forgetAgent(id);
        });
        if (!keep.size && stageEl) applyStageVisibility();
    }

    function applyStageVisibility() {
        if (!stageEl) return;
        const wraps = stageEl.querySelectorAll('.light-face-wrap');
        const hasFaces = wraps.length > 0;
        stageEl.hidden = !stageVisible || !hasFaces;
        syncStageToggleLabels();
    }

    function isStageVisible() {
        return stageVisible;
    }

    function setStageVisible(on) {
        stageVisible = Boolean(on);
        try {
            localStorage.setItem(STAGE_KEY, stageVisible ? '1' : '0');
        } catch {
            /* ignore quota / private mode */
        }
        applyStageVisibility();
    }

    function syncStageToggleLabels() {
        document.querySelectorAll('[data-stage-toggle]').forEach((el) => {
            const compact = el.hasAttribute('data-compact');
            if (compact) {
                el.textContent = 'Stage';
            } else {
                el.textContent = stageVisible ? 'Stage off' : 'Stage on';
            }
            el.setAttribute('aria-pressed', stageVisible ? 'true' : 'false');
            el.classList.toggle('is-on', stageVisible);
            el.hidden = false;
            el.disabled = false;
            el.removeAttribute('hidden');
        });
    }

    function setupStageToggle(buttonOrSelector) {
        const toggle =
            typeof buttonOrSelector === 'string'
                ? document.querySelector(buttonOrSelector)
                : buttonOrSelector;
        if (!toggle) return;
        toggle.setAttribute('data-stage-toggle', '');
        syncStageToggleLabels();
        toggle.addEventListener('click', () => {
            setStageVisible(!stageVisible);
        });
    }

    function paint() {
        if (!stageEl) return;
        presentIds.forEach((id) => ensureWrap(id));
        if (speakingId && presentIds.has(speakingId)) ensureWrap(speakingId);
        const wraps = stageEl.querySelectorAll('.light-face-wrap');
        wraps.forEach((wrap) => {
            const id = wrap.dataset.face;
            const talking = Boolean(speakingId && id === speakingId);
            const onStage = presentIds.has(id) || talking;
            const pose = onStage ? poses[id] || '' : '';
            const gesture = onStage ? gestures[id] || '' : '';
            const overlay = onStage ? overlays[id] || '' : '';
            const gender = genderFor(id);
            const palette = PALETTES[id] || FALLBACK;
            wrap.style.setProperty('--face-glow', palette.head);
            wrap.classList.toggle('is-talking', talking);
            wrap.classList.toggle('is-lip-sync', talking && lipSyncActive);
            wrap.classList.toggle('is-present', presentIds.has(id) && !talking);
            wrap.classList.toggle('has-gesture', Boolean(gesture));
            wrap.classList.toggle('face-gender-girl', gender === 'girl');
            wrap.classList.toggle('face-gender-boy', gender === 'boy');
            if (talking && lipSyncActive) {
                const open = 0.18 + mouthOpen * 0.82;
                wrap.style.setProperty('--mouth-open', String(open));
            } else {
                wrap.classList.remove('is-lip-sync');
                wrap.style.removeProperty('--mouth-open');
            }
            applyFaceClasses(wrap, pose, gesture, overlay, talking);
            const face = wrap.querySelector('.light-face');
            applyFaceClasses(face, pose, gesture, overlay, talking);
            if (face) {
                face.classList.toggle('face-gender-girl', gender === 'girl');
                face.classList.toggle('face-gender-boy', gender === 'boy');
            }
            if (onStage) {
                if (!idleTimers[id]) armIdle(wrap, id);
            } else {
                clearIdle(id);
                wrap.classList.remove('is-idle-bob', 'is-idle-tilt');
            }
        });
        applyStageVisibility();
    }

    function mount(selectorOrEl) {
        stageEl =
            typeof selectorOrEl === 'string'
                ? document.querySelector(selectorOrEl)
                : selectorOrEl;
        if (!stageEl) return null;
        stageEl.classList.add('face-stage');
        paint();
        return stageEl;
    }

    function setPresent(agentId) {
        const id = String(agentId || '').toLowerCase();
        presentIds = id ? new Set([id]) : new Set();
        pruneTo([...presentIds]);
        paint();
    }

    function setPresentMany(ids) {
        presentIds = new Set(
            (ids || []).map((id) => String(id || '').toLowerCase()).filter(Boolean)
        );
        pruneTo([...presentIds]);
        paint();
    }

    function setVoices(map) {
        if (!map || typeof map !== 'object') return;
        Object.keys(map).forEach((key) => {
            const id = String(key || '').toLowerCase();
            if (!id || id === 'default') return;
            const voice = String(map[key] || '').trim();
            if (voice) voiceIds[id] = voice;
        });
        paint();
    }

    function iterCues(text) {
        const out = [];
        const raw = String(text || '');
        CUE_RE.lastIndex = 0;
        let match;
        while ((match = CUE_RE.exec(raw))) {
            const inner = (match[1] || match[2] || match[3] || '').trim();
            if (inner) out.push({ start: match.index, inner: inner });
        }
        return out;
    }

    function classifyCue(inner) {
        const text = String(inner || '').trim();
        if (!text || SKIP_THINK_RE.test(text)) return {};
        let pose = '';
        let gesture = '';
        let holdMs = '';
        if (PAUSE_RE.test(text)) {
            pose = PAUSE_SMILE_RE.test(text) ? 'pause_smile' : 'pause';
            holdMs = LONG_PAUSE_RE.test(text)
                ? String(PAUSE_LONG_HOLD_MS)
                : String(PAUSE_HOLD_MS);
        }
        for (let i = 0; i < RULES.length; i += 1) {
            const rule = RULES[i];
            if (!rule.re.test(text)) continue;
            if (rule.kind === 'gesture') {
                if (!gesture) gesture = rule.name;
            } else if (!pose) {
                pose = rule.name;
            }
            if (pose && gesture) break;
        }
        const found = {};
        if (pose) found.pose = pose;
        if (gesture) found.gesture = gesture;
        if (holdMs) found.hold_ms = holdMs;
        if (BRIGHT_RE.test(text)) found.overlay = 'bright';
        return found;
    }

    function iterEvents(text) {
        const raw = String(text || '');
        const events = [];
        iterCues(raw).forEach((cue) => {
            const classified = classifyCue(cue.inner);
            if (classified.pose || classified.gesture || classified.overlay || classified.hold_ms) {
                events.push({ start: cue.start, end: cue.start + cue.inner.length, classified: classified });
            }
        });
        EMOJI_RE.lastIndex = 0;
        let match;
        while ((match = EMOJI_RE.exec(raw))) {
            const mapped = EMOJI_MAP[match[0]];
            if (mapped) {
                events.push({
                    start: match.index,
                    end: match.index + match[0].length,
                    classified: mapped,
                });
            }
        }
        events.sort((a, b) => a.start - b.start);
        return events;
    }

    function emotionFromText(text, options) {
        const firstOnly = Boolean(options && options.first);
        let found = {};
        const events = iterEvents(text);
        for (let i = 0; i < events.length; i += 1) {
            const classified = events[i].classified;
            found = Object.assign({}, found, classified);
            if (firstOnly) break;
        }
        return found;
    }

    /**
     * Build a cue/emoji timeline weighted by character span to the next signal.
     * Only used when more than one face signal appears in the line.
     */
    function emotionTimeline(text) {
        const raw = String(text || '');
        const events = iterEvents(raw);
        if (events.length === 0) return [];
        if (events.length === 1) {
            return [{ at: 0, classified: events[0].classified }];
        }
        const len = Math.max(raw.length, 1);
        const weights = [];
        let total = 0;
        for (let i = 0; i < events.length; i += 1) {
            const nextStart = i + 1 < events.length ? events[i + 1].start : len;
            const weight = Math.max(1, nextStart - events[i].start);
            weights.push(weight);
            total += weight;
        }
        const steps = [];
        let acc = 0;
        for (let i = 0; i < events.length; i += 1) {
            steps.push({ at: acc / total, classified: events[i].classified });
            acc += weights[i];
        }
        return steps;
    }

    function cancelEmotionClear() {
        if (emotionClearTimer) {
            clearTimeout(emotionClearTimer);
            emotionClearTimer = 0;
        }
    }

    function scheduleEmotionClear() {
        cancelEmotionClear();
        const hold = emotionHoldMs || EMO_HOLD_MS;
        emotionClearTimer = setTimeout(() => {
            emotionClearTimer = 0;
            emotionHoldMs = EMO_HOLD_MS;
            if (speakingId) return;
            Object.keys(poses).forEach((id) => {
                delete poses[id];
            });
            Object.keys(gestures).forEach((id) => {
                delete gestures[id];
            });
            Object.keys(overlays).forEach((id) => {
                delete overlays[id];
            });
            paint();
        }, hold);
    }

    function applyEmotion(agentId, parsed, holdIfNone) {
        const id = String(agentId || '').toLowerCase();
        if (!id) return;
        if (parsed.pose) poses[id] = parsed.pose;
        else if (!holdIfNone) delete poses[id];
        if (parsed.gesture) gestures[id] = parsed.gesture;
        else if (!holdIfNone) delete gestures[id];
        if (parsed.overlay) overlays[id] = parsed.overlay;
        else if (!holdIfNone) delete overlays[id];
        if (parsed.hold_ms) {
            const ms = Number(parsed.hold_ms);
            emotionHoldMs = Number.isFinite(ms) && ms > 0 ? ms : EMO_HOLD_MS;
        } else if (parsed.pose === 'pause' || parsed.pose === 'pause_smile') {
            emotionHoldMs = PAUSE_HOLD_MS;
        } else if (parsed.pose || parsed.gesture || parsed.overlay) {
            emotionHoldMs = EMO_HOLD_MS;
        }
        paint();
    }

    function emoteFromText(agentId, text, options) {
        const opts = options || {};
        const parsed = emotionFromText(text, opts);
        if (!parsed.pose && !parsed.gesture && !parsed.overlay && !parsed.hold_ms) {
            return parsed;
        }
        applyEmotion(agentId, parsed, Boolean(opts.holdIfNone));
        if (opts.persist || speakingId) {
            cancelEmotionClear();
        } else {
            scheduleEmotionClear();
        }
        return parsed;
    }

    function applySpeakStep(stepIndex) {
        if (!speakPlan || !speakPlan.steps.length) return;
        const idx = Math.max(0, Math.min(stepIndex, speakPlan.steps.length - 1));
        if (idx === speakPlan.stepIndex && speakPlan.stepIndex >= 0) return;
        speakPlan.stepIndex = idx;
        applyEmotion(speakPlan.id, speakPlan.steps[idx].classified, true);
    }

    function setMouthOpen(amount) {
        const a = Math.max(0, Math.min(1, Number(amount) || 0));
        mouthOpen = a;
        lipSyncActive = true;
        if (!stageEl || !speakingId) return;
        const wrap = stageEl.querySelector(
            '.light-face-wrap[data-face="' + speakingId + '"]'
        );
        if (!wrap) return;
        wrap.classList.add('is-talking', 'is-lip-sync');
        wrap.style.setProperty('--mouth-open', String(0.18 + a * 0.82));
    }

    function setSpeaking(agentId, chunkText) {
        speakingId = String(agentId || '').toLowerCase();
        cancelEmotionClear();
        const steps = emotionTimeline(chunkText || '');
        if (steps.length) {
            speakPlan = { id: speakingId, steps: steps, stepIndex: -1 };
            applySpeakStep(0);
        } else {
            speakPlan = null;
        }
        paint();
    }

    function syncSpeakingProgress(ratio) {
        if (!speakPlan || speakPlan.steps.length <= 1) return;
        const t = Math.max(0, Math.min(1, Number(ratio) || 0));
        let idx = 0;
        for (let i = 0; i < speakPlan.steps.length; i += 1) {
            if (speakPlan.steps[i].at <= t) idx = i;
            else break;
        }
        applySpeakStep(idx);
    }

    function clearSpeaking() {
        speakingId = '';
        speakPlan = null;
        mouthOpen = 0;
        lipSyncActive = false;
        if (stageEl) {
            stageEl.querySelectorAll('.light-face-wrap.is-lip-sync').forEach((wrap) => {
                wrap.classList.remove('is-lip-sync');
                wrap.style.removeProperty('--mouth-open');
            });
        }
        pruneTo([...presentIds]);
        paint();
        scheduleEmotionClear();
    }

    LH.faces = {
        mount,
        setPresent,
        setPresentMany,
        setVoices,
        setSpeaking,
        setMouthOpen,
        syncSpeakingProgress,
        clearSpeaking,
        emoteFromText,
        emotionFromText,
        emotionTimeline,
        classifyCue,
        genderFromVoice,
        isStageVisible,
        setStageVisible,
        setupStageToggle,
    };
})(typeof window !== 'undefined' ? window : globalThis);
