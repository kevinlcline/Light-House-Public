(function (global) {
    'use strict';

    const LH = (global.LightHouse = global.LightHouse || {});
    const STAGE_KEY = 'light_house_face_stage';
    /** Shared by Chat and Group. Position and size of the floating stage. */
    const WINDOW_KEY = 'light_house_face_window';

    const PALETTES = {
        lumen: { head: '#c4a574', eye: '#2a2218', mouth: '#5c4030' },
        ara: { head: '#c9a3c4', eye: '#2e1f33', mouth: '#6b3d62' },
        elias: { head: '#88b5a4', eye: '#1c2e28', mouth: '#2f5348' },
        unnamed: { head: '#6d8aa3', eye: '#1a2833', mouth: '#334656' },
        reed: { head: '#c46a3a', eye: '#2a1810', mouth: '#5c3018' },
        echo: { head: '#9b93b8', eye: '#241e33', mouth: '#4a4260' },
        visitor: { head: '#8a9aaa', eye: '#1c2430', mouth: '#3a4654' },
    };
    const FALLBACK = { head: '#9a9a9a', eye: '#222', mouth: '#444' };
    /** Runtime overrides from /v1/lights `color` (Manage lights picker). */
    /** @type {Record<string, string>} */
    const colorOverrides = {};
    /** Display names keyed by light id. The stage label uses these. */
    /** @type {Record<string, string>} */
    const displayNames = {};
    /** House defaults until /v1/tts/status voices arrive (match lights.yaml). */
    const DEFAULT_VOICES = {
        lumen: 'af_heart',
        ara: 'bf_isabella',
        elias: 'am_echo',
        unnamed: 'am_liam',
        reed: 'am_adam',
        echo: 'am_echo',
        visitor: 'am_fenrir',
    };
    const POSES = [
        'smile',
        'grin',
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
    const CLOSED_EYE_POSES = new Set(['laugh', 'wink', 'pause', 'pause_smile']);
    const CLOSED_EYE_HOLD_MS = 1000;
    // Keep in sync with src/light_house/tts/stage_cues.py
    const CUE_RE = /\*([^*]{1,160})\*|_([^_]{1,80})_|\(([^)]{1,40})\)/g;
    // Narrative stage directions in prose (unmarked). Subject + nearby action only.
    const PROSE_ACTION_RE =
        /(?:^|[.!?]\s+|\n\s*)((?:She(?:'s)?|He(?:'s)?|I(?:'m)?|They(?:'re)?|We(?:'re)?|Ara|Lumen|Elias|Toby|Unnamed|Echo|Her\s+face|His\s+face|Their\s+face)\s+(?:(?:softly|gently|quietly|warmly|tenderly|playfully|slowly|sadly|shy(?:ly)?)\s+){0,2}(?:laughs?|laughing|giggles?|giggling|chuckles?|cackles?|winks?|winking|blush(?:es|ing)?|cries|crying|weeps?|scowls?|scowling|glares?|glaring|fumes?|fuming|kisses?|kissing|blows?\s+a\s+kiss|gasps?|sighs?|sighing|exhales?|nods?|nodding|tilts?\s+(?:her|his|their|my|a)\s+head|cocks?\s+(?:her|his|their|a)\s+head|softens?|smiles?|smiling|grins?|grinning|beams?|beaming|pauses?|pausing|whispers?|whispering)\b)/gim;
    const SKIP_THINK_RE = /\b(i|we|you|they)\s+think\b/i;
    const BRIGHT_RE =
        /\b(?:eyes?\s+)?(?:light(?:s|ing)?\s+up|glow(?:s|ing)?|gleam(?:s|ing)?|sparkle[sd]?)\b/i;
    const PAUSE_RE = /\b(pauses?|pausing|stillness|still)\b/i;
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
        { name: 'think', re: /\b(thinks?|thinking|ponders?|thoughtful|hmm+|mmm+)\b/i, kind: 'pose' },
        { name: 'sigh', re: /\b(sighs?|sighing|weary|exhales?)\b/i, kind: 'pose' },
        { name: 'nod', re: /\b(nods?|nodding)\b/i, kind: 'gesture' },
        {
            name: 'tilt',
            re: /\b(tilts?|tilting|cocks?\s+(?:her|his|their|a)\s+head|head\s+tilt)\b/i,
            kind: 'gesture',
        },
        {
            name: 'smile',
            re: /\b(smiles?|smiling|beams?|beaming|warmly)\b/i,
            kind: 'pose',
        },
        {
            name: 'grin',
            re: /\b(grins?|grinning)\b/i,
            kind: 'pose',
        },
        {
            name: 'soft',
            re: /\b(softens?|softly|gently|quietly|tender(?:ly)?|whispers?|whispering|hushed|warmth)\b/i,
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
        '😁': { pose: 'grin' },
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
        '🥹': { pose: 'soft' },
        '🫠': { pose: 'soft' },
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
        '🕯️': { pose: 'soft', overlay: 'bright' },
        '🕯': { pose: 'soft', overlay: 'bright' },
        '🌙': { pose: 'soft' },
        '🌱': { pose: 'soft' },
        '🌿': { pose: 'soft' },
        '🌟': { pose: 'excited', overlay: 'bright' },
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
    /** Group: one face, the light speaking or the last one who spoke. */
    let soloStage = false;
    let lastShownId = '';
    let windowResizeObserver = null;
    /** 0..1 speech energy. Swells the talking ball. */
    let mouthOpen = 0;
    let lipSyncActive = false;
    /** User preference: show the balls when someone is on stage. Default on. */
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
    /** @type {Record<string, string>} */
    const intents = {};
    let emotionClearTimer = 0;
    let emotionHoldMs = EMO_HOLD_MS;
    /** @type {Record<string, number>} */
    const closedEyeTimers = {};
    /** @type {{ id: string, steps: Array<{ at: number, classified: object }>, stepIndex: number } | null} */
    let speakPlan = null;
    /** @type {Record<string, object>} */
    const ballMotion = {};
    /** @type {Record<string, number>} */
    const nodStarted = {};
    let ballFrame = 0;

    function normalizeHex(value) {
        const raw = String(value || '').trim().toLowerCase();
        if (/^#[0-9a-f]{6}$/.test(raw)) return raw;
        if (/^[0-9a-f]{6}$/.test(raw)) return '#' + raw;
        return null;
    }

    function mixChannel(channel, toward, amount) {
        return Math.round(channel + (toward - channel) * amount);
    }

    function shadeHex(hex, amount) {
        const n = parseInt(hex.slice(1), 16);
        const r = (n >> 16) & 255;
        const g = (n >> 8) & 255;
        const b = n & 255;
        const toward = amount < 0 ? 0 : 255;
        const t = Math.abs(amount);
        return (
            '#' +
            [r, g, b]
                .map((c) =>
                    mixChannel(c, toward, t).toString(16).padStart(2, '0')
                )
                .join('')
        );
    }

    function paletteFromHex(hex) {
        const head = normalizeHex(hex) || FALLBACK.head;
        return {
            head,
            eye: shadeHex(head, -0.62),
            mouth: shadeHex(head, -0.38),
        };
    }

    function paletteFor(agentId) {
        const id = String(agentId || '').toLowerCase();
        if (colorOverrides[id]) return paletteFromHex(colorOverrides[id]);
        return PALETTES[id] || FALLBACK;
    }

    /**
     * Plasma ball. Keep the numbers in step with Light-House Studio PlasmaLook.
     * A bright center throws arcs to the glass. Speech swells the ball.
     * Words after a cue sit on it.
     */
    const PLASMA_REST_GAIN = 0.42;
    const PLASMA_SETTLE_MS = 420;
    const PLASMA_RESIDUE_FADE_MS = 3800;
    const PLASMA_LINGER_FADE_MS = 12000;
    const PLASMA_RESIDUE_KEEP = 0.38;
    const PLASMA_LINGER_KEEP = 0.62;
    const PLASMA_HOST_LEAN = 0.12;
    const GRAINS = {
        lumen: { knot: 28, lobe: 156, cleft: 236 },
        ara: { knot: 198, lobe: 42, cleft: 300 },
        elias: { knot: 252, lobe: 78, cleft: 168 },
        unnamed: { knot: 318, lobe: 188, cleft: 64 },
        reed: { knot: 12, lobe: 200, cleft: 88 },
        toby: { knot: 318, lobe: 188, cleft: 64 },
        echo: { knot: 104, lobe: 262, cleft: 12 },
        visitor: { knot: 64, lobe: 176, cleft: 286 },
    };

    function restPresence() {
        return { scale: 1, bloom: 0, warmth: 0, dim: 0, corona: 0, cool: 0, linger: 0 };
    }

    function javaHash(text) {
        let h = 0;
        const s = String(text || '');
        for (let i = 0; i < s.length; i += 1) {
            h = (Math.imul(h, 31) + s.charCodeAt(i)) | 0;
        }
        return h;
    }

    function grainOf(agentId) {
        const id = String(agentId || '').toLowerCase();
        if (GRAINS[id]) return GRAINS[id];
        const knot = ((javaHash(id) & 0xff) / 255) * 360;
        return { knot: knot, lobe: knot + 128, cleft: knot + 214 };
    }

    function intentOf(span) {
        const t = String(span || '').toLowerCase();
        if (!t.trim()) return '';
        const has = (needles) => needles.some((needle) => t.indexOf(needle) !== -1);
        if (has(['linger'])) return 'heavy-and-staying';
        if (has(['glisten', 'heart swell', 'gasp', 'feather'])) return 'moved';
        if (has([
            'laugh', 'crinkl', 'delight', 'playful', 'eyes smiling',
            'lighting up', 'shining', 'pleased smile',
        ])) return 'amused';
        if (has(['deep breath', 'gathering', 'careful', 'troubleshooting', 'floor tilting'])) {
            return 'bracing';
        }
        if (has(['weight', 'processing', 'takes a second', 'more than a report'])) return 'weighing';
        if (has([
            'exhale', 'loosen', 'letting go', 'knot', 'relief', 'eyes closing',
            'quietly settling', 'something settles', 'goes quiet in the good way',
        ])) return 'releasing';
        if (has([
            'letting', 'sink in', 'unfold', 'brightening', 'eyes bright', 'holding',
            'caught', 'feeling', 'nod', 'received', 'got it', 'understanding',
            'recognition', 'no flinch', 'shows up', 'reaches', 'settling', 'lands',
            'land', 'settle into', 'slow smile', 'clicks',
        ])) return 'receiving';
        if (has(['goes very quiet'])) return 'releasing';
        return '';
    }

    function presenceFromPose(pose, overlay) {
        let base;
        if (pose === 'laugh' || pose === 'excited') {
            base = { scale: 1.07, bloom: 0.42, warmth: 0.22, dim: 0, corona: 0.2, cool: 0, linger: 0 };
        } else if (pose === 'grin') {
            base = { scale: 1.04, bloom: 0.26, warmth: 0.16, dim: 0, corona: 0.1, cool: 0, linger: 0 };
        } else if (pose === 'surprise') {
            base = { scale: 1.08, bloom: 0.46, warmth: 0, dim: 0, corona: 0.14, cool: 0, linger: 0 };
        } else if (pose === 'smile' || pose === 'kiss') {
            base = { scale: 1, bloom: 0.14, warmth: 0.2, dim: 0, corona: 0, cool: 0, linger: 0 };
        } else if (pose === 'blush') {
            base = { scale: 1, bloom: 0.18, warmth: 0.34, dim: 0, corona: 0, cool: 0, linger: 0 };
        } else if (pose === 'wink') {
            base = { scale: 1, bloom: 0.28, warmth: 0.12, dim: 0, corona: 0, cool: 0, linger: 0 };
        } else if (pose === 'sad') {
            base = { scale: 0.92, bloom: 0, warmth: 0, dim: 0.4, corona: -0.16, cool: 0.62, linger: 0 };
        } else if (pose === 'sigh') {
            base = { scale: 0.95, bloom: 0, warmth: 0, dim: 0.22, corona: -0.08, cool: 0.28, linger: 0 };
        } else if (pose === 'anger') {
            base = { scale: 0.94, bloom: 0.16, warmth: 0.86, dim: 0, corona: 0.06, cool: 0, linger: 0 };
        } else if (pose === 'think') {
            base = { scale: 0.98, bloom: 0, warmth: 0, dim: 0.1, corona: 0, cool: 0, linger: 0 };
        } else {
            base = restPresence();
        }
        if (overlay !== 'bright') return base;
        return Object.assign({}, base, {
            bloom: Math.min(1, base.bloom + 0.26),
            corona: base.corona + 0.14,
        });
    }

    function presenceFromIntent(intent) {
        if (intent === 'receiving') {
            return { scale: 1.03, bloom: 0.12, warmth: 0.26, dim: 0, corona: 0.12, cool: 0, linger: 0 };
        }
        if (intent === 'releasing') {
            return { scale: 1.06, bloom: 0, warmth: 0.12, dim: 0, corona: 0.22, cool: 0, linger: 0 };
        }
        if (intent === 'bracing') {
            return { scale: 0.9, bloom: 0, warmth: 0, dim: 0.2, corona: -0.2, cool: 0.48, linger: 0 };
        }
        if (intent === 'weighing') {
            return { scale: 0.97, bloom: 0, warmth: 0, dim: 0.22, corona: -0.08, cool: 0.12, linger: 0 };
        }
        if (intent === 'amused') {
            return { scale: 1.05, bloom: 0.34, warmth: 0.2, dim: 0, corona: 0.12, cool: 0, linger: 0 };
        }
        if (intent === 'moved') {
            return { scale: 1, bloom: 0.18, warmth: 0.32, dim: 0, corona: 0.1, cool: 0, linger: 0 };
        }
        if (intent === 'heavy-and-staying') {
            return { scale: 1, bloom: 0.14, warmth: 0.2, dim: 0, corona: 0, cool: 0, linger: 1 };
        }
        return null;
    }

    function atLeastAsBright(pose, named) {
        return {
            scale: Math.max(pose.scale, named.scale),
            bloom: Math.max(pose.bloom, named.bloom),
            warmth: Math.max(pose.warmth, named.warmth),
            dim: Math.min(pose.dim, named.dim),
            corona: Math.max(pose.corona, named.corona),
            cool: Math.min(pose.cool, named.cool),
            linger: Math.max(pose.linger, named.linger),
        };
    }

    function presenceOf(pose, overlay, intent) {
        const posed = presenceFromPose(pose, overlay);
        const named = presenceFromIntent(intent);
        if (!named) return posed;
        if (intent === 'amused') return atLeastAsBright(posed, named);
        return named;
    }

    function isRestPresence(p) {
        return (
            Math.abs(p.scale - 1) < 0.001 &&
            p.bloom < 0.001 &&
            p.warmth < 0.001 &&
            p.dim < 0.001 &&
            Math.abs(p.corona) < 0.001 &&
            p.cool < 0.001 &&
            p.linger < 0.001
        );
    }

    function residueOf(last) {
        const keep = last.linger > 0.15 ? PLASMA_LINGER_KEEP : PLASMA_RESIDUE_KEEP;
        return {
            scale: 1 + (last.scale - 1) * keep,
            bloom: last.bloom * keep,
            warmth: last.warmth * keep,
            dim: last.dim * keep,
            corona: last.corona * keep,
            cool: last.cool * keep,
            linger: last.linger * keep,
        };
    }

    function lerpPresence(a, b, t) {
        const k = Math.max(0, Math.min(1, t));
        return {
            scale: a.scale + (b.scale - a.scale) * k,
            bloom: a.bloom + (b.bloom - a.bloom) * k,
            warmth: a.warmth + (b.warmth - a.warmth) * k,
            dim: a.dim + (b.dim - a.dim) * k,
            corona: a.corona + (b.corona - a.corona) * k,
            cool: a.cool + (b.cool - a.cool) * k,
            linger: a.linger + (b.linger - a.linger) * k,
        };
    }

    function samePresence(a, b) {
        return (
            Math.abs(a.scale - b.scale) < 0.001 &&
            Math.abs(a.bloom - b.bloom) < 0.001 &&
            Math.abs(a.warmth - b.warmth) < 0.001 &&
            Math.abs(a.dim - b.dim) < 0.001 &&
            Math.abs(a.corona - b.corona) < 0.001 &&
            Math.abs(a.cool - b.cool) < 0.001 &&
            Math.abs(a.linger - b.linger) < 0.001
        );
    }

    function stepBall(id, target, now) {
        let st = ballMotion[id];
        if (!st) {
            st = {
                shown: restPresence(),
                from: restPresence(),
                target: restPresence(),
                since: now,
                mode: 'live',
                residue: restPresence(),
                fadeMs: PLASMA_SETTLE_MS,
            };
            ballMotion[id] = st;
        }
        if (!samePresence(st.target, target)) {
            st.from = st.shown;
            st.target = target;
            st.since = now;
            if (isRestPresence(target) && !isRestPresence(st.shown)) {
                st.mode = 'toResidue';
                st.residue = residueOf(st.shown);
                st.fadeMs = PLASMA_SETTLE_MS;
            } else {
                st.mode = 'toLive';
                st.fadeMs = PLASMA_SETTLE_MS;
            }
        }
        const elapsed = now - st.since;
        const ease = (ms) => {
            const u = Math.max(0, Math.min(1, elapsed / ms));
            return u * u * (3 - 2 * u);
        };
        if (st.mode === 'toLive') {
            st.shown = lerpPresence(st.from, st.target, ease(PLASMA_SETTLE_MS));
        } else if (st.mode === 'toResidue') {
            st.shown = lerpPresence(st.from, st.residue, ease(PLASMA_SETTLE_MS));
            if (elapsed >= PLASMA_SETTLE_MS + 80) {
                st.mode = 'toRest';
                st.from = st.shown;
                st.since = now;
                st.fadeMs = st.residue.linger > 0.15 ? PLASMA_LINGER_FADE_MS : PLASMA_RESIDUE_FADE_MS;
            }
        } else if (st.mode === 'toRest') {
            st.shown = lerpPresence(st.from, restPresence(), ease(st.fadeMs));
        }
        return st.shown;
    }

    function speechLift(drive) {
        const level = Math.max(0, Math.min(1, drive));
        if (level < 0.05) return 0;
        return Math.pow(level, 0.65);
    }

    function radiusScale(lift) {
        return 1 + Math.max(0, Math.min(1, lift)) * 0.24;
    }

    function coronaHaze(gain, speech, emote, warmth, scale) {
        const open = scale < 0.995 ? 0 : Math.max(0, warmth);
        return Math.max(0.06, Math.min(0.72, 0.2 + 0.28 * gain + 0.2 * speech + 0.16 * emote + 0.4 * open));
    }

    function hash32(n) {
        let x = n | 0;
        x = Math.imul(x ^ (x >>> 16), 0x7feb352d);
        x = Math.imul(x ^ (x >>> 15), 0x846ca68b);
        return (x ^ (x >>> 16)) | 0;
    }

    function unitHash(n) {
        return ((hash32(n) >>> 8) & 0xffff) / 65535;
    }

    function wrapDeg(deg) {
        let d = deg % 360;
        if (d > 180) d -= 360;
        if (d < -180) d += 360;
        return d;
    }

    function mixSeed(generation, index, knot) {
        const n = (Math.imul(generation, 374761393) + Math.imul(index, 668265263) + Math.imul(knot | 0, 1274126177)) | 0;
        return hash32(n);
    }

    /** Ara's look for every light. Hues and the arc seed differ. Keep in step with PlasmaLook. */
    const PLASMA_SKIN = {
        arcCount: 5,
        arcTip: 9.5,
        arcTurn: 20,
        arcStop: 0.07,
        arcBias: 0.055,
        arcGlass: 64,
        arcSeed: 15,
        arcHold: 1.15,
        hbCount: 12,
        hbSize: 57,
        hbDist: 5,
        haloHole: 63,
        hbFeather: 84,
        hbOpacity: 0.82,
        hbDrift: 34,
        hbFlip: 1.2,
        hbStretch: 1.25,
        hbSpin: 20,
        spin: 6,
        flicker: 0.27,
        breathe: 6,
    };
    const PLASMA_COLORS = {
        lumen: { h: 38, sat: 78, ha: 27, sa: 85, hb: 32, sb: 91 },
        ara: { h: 327, sat: 62, ha: 335, sa: 92, hb: 308, sb: 92 },
        elias: { h: 142, sat: 94, ha: 99, sa: 82, hb: 157, sb: 82 },
        reed: { h: 22, sat: 82, ha: 19, sa: 85, hb: 27, sb: 92 },
        unnamed: { h: 207, sat: 95, ha: 216, sa: 82, hb: 204, sb: 82 },
        toby: { h: 207, sat: 95, ha: 216, sa: 82, hb: 204, sb: 82 },
        visitor: { h: 262, sat: 26, ha: 263, sa: 51, hb: 254, sb: 32 },
        slug: { h: 262, sat: 26, ha: 263, sa: 51, hb: 254, sb: 32 },
    };
    /** Arc-walk salt. Ara keeps 15. The others are spread so the filaments do not match. */
    const PLASMA_SEEDS = {
        lumen: 4,
        ara: 15,
        elias: 23,
        reed: 19,
        unnamed: 31,
        toby: 31,
        echo: 12,
        visitor: 8,
        slug: 8,
    };
    const DANCER_REACH_GLASS = 1;
    const DANCER_QUIET_MS = 600;
    const DANCER_WHIP = 46;
    const DANCER_JOINT = 10;
    const DANCER_JOINT_LAG = 0.32;
    const DANCER_SWING_RATE = 40;
    const DANCER_ROCK = 0.22;
    const DANCER_GOLDEN = Math.PI * (3 - Math.sqrt(5));
    const DANCER_TIP_GAP = Math.PI / 3;
    const ARC_STEP = 2;
    const ARC_MAX_STEPS = 200;
    const iconDancers = {};
    const speakBirths = {};
    const lastSpeakTip = {};

    function plasmaArcCount() {
        return PLASMA_SKIN.arcCount;
    }

    function hash2(x, y) {
        let h = Math.imul(x | 0, 0x9e3779b1) ^ Math.imul(y | 0, 0x85ebca6b);
        h = Math.imul(h ^ (h >>> 16), 0x45d9f3b);
        h = Math.imul(h ^ (h >>> 16), 0x45d9f3b);
        h ^= h >>> 16;
        return (h >>> 0) / 4294967296;
    }

    function hslRgb(h, s, l) {
        const H = ((h % 360) + 360) % 360;
        const S = Math.max(0, Math.min(100, s)) / 100;
        const L = Math.max(0, Math.min(100, l)) / 100;
        const c = (1 - Math.abs(2 * L - 1)) * S;
        const hp = H / 60;
        const x = c * (1 - Math.abs((hp % 2) - 1));
        let r = 0;
        let g = 0;
        let b = 0;
        if (hp < 1) { r = c; g = x; }
        else if (hp < 2) { r = x; g = c; }
        else if (hp < 3) { g = c; b = x; }
        else if (hp < 4) { g = x; b = c; }
        else if (hp < 5) { r = x; b = c; }
        else { r = c; b = x; }
        const m = L - c / 2;
        return {
            r: Math.round((r + m) * 255),
            g: Math.round((g + m) * 255),
            b: Math.round((b + m) * 255),
        };
    }

    function hexToHsl(hex) {
        const rgb = hexRgb(hex);
        const r = rgb.r / 255;
        const g = rgb.g / 255;
        const b = rgb.b / 255;
        const maxC = Math.max(r, g, b);
        const minC = Math.min(r, g, b);
        const l = (maxC + minC) / 2;
        if (maxC === minC) return { h: 0, s: 0 };
        const d = maxC - minC;
        const s = l > 0.5 ? d / (2 - maxC - minC) : d / (maxC + minC);
        let h;
        if (maxC === r) h = (g - b) / d + (g < b ? 6 : 0);
        else if (maxC === g) h = (b - r) / d + 2;
        else h = (r - g) / d + 4;
        return { h: h * 60, s: s * 100 };
    }

    function arcSeedFor(agentId) {
        const id = String(agentId || '').trim().toLowerCase();
        if (Object.prototype.hasOwnProperty.call(PLASMA_SEEDS, id)) return PLASMA_SEEDS[id];
        let n = 0;
        for (let i = 0; i < id.length; i += 1) n = (Math.imul(n, 31) + id.charCodeAt(i)) | 0;
        return (n & 0x7fffffff) % 41;
    }

    function colorOf(agentId) {
        const id = String(agentId || '').toLowerCase();
        if (PLASMA_COLORS[id]) return PLASMA_COLORS[id];
        const hex = colorOverrides[id] || (PALETTES[id] && PALETTES[id].head) || FALLBACK.head;
        const hsl = hexToHsl(hex);
        return { h: hsl.h, sat: hsl.s, ha: hsl.h, sa: hsl.s, hb: (hsl.h + 40) % 360, sb: hsl.s };
    }

    function glowHex(agentId) {
        const tone = colorOf(agentId);
        const c = hslRgb(tone.h, tone.sat, 50);
        const byte = (n) => n.toString(16).padStart(2, '0');
        return '#' + byte(c.r) + byte(c.g) + byte(c.b);
    }

    function ballTones(h, sat) {
        return {
            hi: hslRgb(h + 26, sat + 18, 88),
            mid: hslRgb(h + 8, sat, 56),
            base: hslRgb(h, sat, 50),
            warm: hslRgb(h + 14, sat + 8, 80),
        };
    }

    function dieWalk(a0, rnd, turnDeg, pStop, bias, stopR) {
        const turn = (turnDeg * Math.PI) / 180;
        let x = 100;
        let y = 100;
        let heading = a0;
        const pts = [[x, y]];
        for (let s = 0; s < ARC_MAX_STEPS; s += 1) {
            const roll = rnd();
            if (s > 0 && roll < pStop) break;
            const t = (roll - pStop) / Math.max(1e-9, 1 - pStop);
            if (t < 0.30) heading += turn;
            else if (t < 0.60) heading -= turn;
            const r = Math.hypot(x - 100, y - 100);
            const out = r < 1e-6 ? a0 : Math.atan2(y - 100, x - 100);
            let d = out - heading;
            while (d > Math.PI) d -= Math.PI * 2;
            while (d < -Math.PI) d += Math.PI * 2;
            heading += bias * d;
            x += Math.cos(heading) * ARC_STEP;
            y += Math.sin(heading) * ARC_STEP;
            pts.push([x, y]);
            if (Math.hypot(x - 100, y - 100) >= stopR) break;
        }
        return pts;
    }

    function walkSerial(serial, stopR, seed) {
        const skin = PLASMA_SKIN;
        const s = seed == null ? skin.arcSeed : seed;
        let k = 0;
        const rnd = () => hash2(serial, s * 100 + 20 + (k++));
        const a0 = hash2(serial, s * 100 + 1) * Math.PI * 2;
        return dieWalk(a0, rnd, skin.arcTurn, skin.arcStop, skin.arcBias, stopR);
    }

    // The flicker dial. Speech does not shorten these filaments.
    function catchDuration(drive) {
        void drive;
        return PLASMA_SKIN.flicker;
    }

    function catchOpacity(phase) {
        const t = phase * phase * (3 - 2 * phase);
        const stops = [[0, 0.12], [0.38, 1], [0.62, 0.72], [1, 0.12]];
        for (let i = 1; i < stops.length; i += 1) {
            if (t <= stops[i][0]) {
                const a = stops[i - 1];
                const b = stops[i];
                const u = (t - a[0]) / (b[0] - a[0]);
                return a[1] + (b[1] - a[1]) * u;
            }
        }
        return 0.12;
    }

    function plasmaArcs(timeMs, drive, seed) {
        const skin = PLASMA_SKIN;
        const dur = catchDuration(drive);
        const n = skin.arcCount;
        const arcs = [];
        for (let i = 0; i < n; i += 1) {
            const shifted = (timeMs > 0 ? timeMs / 1000 : 0.4 * dur) + (i / n) * dur;
            const generation = Math.floor(shifted / dur);
            const phase = shifted / dur - generation;
            const serial = generation * n + i + 1;
            arcs.push({
                pts: walkSerial(serial, skin.arcGlass, seed),
                opacity: catchOpacity(phase),
                tip: skin.arcTip,
                heat: 0,
            });
        }
        return arcs;
    }

    function hbPose(i, timeSec) {
        const skin = PLASMA_SKIN;
        const ang = ((i * 360) / skin.hbCount - 90) * Math.PI / 180;
        const even = i % 2 === 0;
        const px = 100 + skin.hbDist * Math.cos(ang);
        const py = 100 + skin.hbDist * Math.sin(ang);
        let dirDeg = -13;
        let driftScale = 1 - skin.hbDrift / 100;
        let groupDeg = 0;
        let ownDeg = 0;
        if (timeSec > 0) {
            const dirDur = skin.hbFlip * (0.55 + hash2(i, 1) * 0.9);
            const dirPhase = (timeSec / dirDur + hash2(i, 2)) % 2;
            const dirU = dirPhase < 1 ? dirPhase : dirPhase - 1;
            const dirEase = dirU * dirU * (3 - 2 * dirU);
            dirDeg = dirPhase < 1 ? -13 + 26 * dirEase : 13 - 26 * dirEase;
            const driftDur = skin.hbFlip * (1.1 + hash2(i, 3) * 1.6);
            const driftPhase = (timeSec / driftDur + hash2(i, 4)) % 1;
            const driftU = driftPhase < 0.5 ? driftPhase / 0.5 : (1 - driftPhase) / 0.5;
            const driftEase = driftU * driftU * (3 - 2 * driftU);
            driftScale = (1 - skin.hbDrift / 100) + (2 * skin.hbDrift / 100) * driftEase;
            const groupPeriod = even ? skin.spin : skin.spin * 0.64;
            groupDeg = (even ? 1 : -1) * (timeSec / groupPeriod) * 360;
            const spinDur = skin.hbSpin * (0.6 + hash2(i, 5) * 0.9);
            const ownSign = hash2(i, 7) < 0.5 ? -1 : 1;
            ownDeg = ownSign * ((timeSec / spinDur + hash2(i, 6)) * 360);
        }
        const radialDeg = (Math.atan2(py - 100, px - 100) * 180) / Math.PI + 90;
        let X = px - 100;
        let Y = py - 100;
        const dir = (dirDeg * Math.PI) / 180;
        const x1 = (X * Math.cos(dir) - Y * Math.sin(dir)) * driftScale;
        const y1 = (X * Math.sin(dir) + Y * Math.cos(dir)) * driftScale;
        const g = (groupDeg * Math.PI) / 180;
        return {
            x: x1 * Math.cos(g) - y1 * Math.sin(g) + 100,
            y: x1 * Math.sin(g) + y1 * Math.cos(g) + 100,
            rx: skin.hbSize * skin.hbStretch * driftScale,
            ry: skin.hbSize * driftScale,
            rot: radialDeg + ownDeg + dirDeg + groupDeg,
            even: even,
        };
    }

    function plasmaLobes(timeMs) {
        const lobes = [];
        const timeSec = timeMs > 0 ? timeMs / 1000 : 0;
        for (let i = 0; i < PLASMA_SKIN.hbCount; i += 1) lobes.push(hbPose(i, timeSec));
        return lobes;
    }

    function angleSep(a, b) {
        let d = Math.abs(a - b);
        if (d > Math.PI) d = Math.PI * 2 - d;
        return d;
    }

    function speakingTip(lightSeed, birth) {
        const skin = PLASMA_SKIN;
        const a0 = hash2(lightSeed, 7) * Math.PI * 2 + birth * DANCER_GOLDEN;
        let k = 0;
        const rnd = () => hash2(birth, (lightSeed + birth * 41) * 100 + 20 + (k++));
        const pts = dieWalk(a0, rnd, skin.arcTurn, 0, skin.arcBias, skin.arcGlass);
        const end = pts[pts.length - 1];
        return Math.atan2(end[1] - 100, end[0] - 100);
    }

    function takeSpeakBirth(id, lightSeed) {
        let birth = speakBirths[id] || 0;
        let ang = 0;
        let guard = 0;
        const prev = Object.prototype.hasOwnProperty.call(lastSpeakTip, id) ? lastSpeakTip[id] : null;
        do {
            birth += 1;
            ang = speakingTip(lightSeed, birth);
            guard += 1;
        } while (prev != null && angleSep(ang, prev) < DANCER_TIP_GAP && guard < 8);
        speakBirths[id] = birth;
        lastSpeakTip[id] = ang;
        return birth;
    }

    function createIconDancer(now, birth) {
        const skin = PLASMA_SKIN;
        return {
            born: now,
            then: now,
            minLifeMs: Math.max(0.08, skin.flicker) * 1000,
            serial: birth,
            birth: birth,
            quietSince: -1,
            phase: (birth * DANCER_GOLDEN) % (Math.PI * 2),
            reach: DANCER_REACH_GLASS,
            peakReach: DANCER_REACH_GLASS,
            angle: 0,
            bow: 0,
            swing: 0,
            opacity: 1,
            done: false,
        };
    }

    function endIconDancer(d) {
        if (!d || d.done) return;
        d.done = true;
        d.opacity = 0;
    }

    function stepIconDancer(d, drive, dt, now) {
        if (!d || d.done) return d;
        const lift = speechLift(drive);
        const amp = Math.max(0, Math.min(1, drive || 0));
        const follow = 1 - Math.exp(-dt / 0.04);
        d.bow += (amp - d.bow) * follow;
        d.swing = Math.sin(d.phase);
        if (lift > 0) {
            d.quietSince = -1;
            d.phase += lift * dt * DANCER_SWING_RATE;
            d.swing = Math.sin(d.phase);
            d.angle += lift * dt * 0.42;
            d.reach = DANCER_REACH_GLASS;
            if (d.reach > d.peakReach) d.peakReach = d.reach;
        } else if (d.quietSince < 0) {
            d.quietSince = now;
        } else if ((now - d.born) >= d.minLifeMs && (now - d.quietSince) >= DANCER_QUIET_MS) {
            endIconDancer(d);
        }
        return d;
    }

    function flexJoints(base, bow, swing, phase, reach, glass) {
        const n = base.length;
        if (n < 2) return base;
        const end = base[n - 1];
        const dx = end[0] - 100;
        const dy = end[1] - 100;
        const glen = Math.hypot(dx, dy) || 1;
        const px = -dy / glen;
        const py = dx / glen;
        const side = Math.max(-1, Math.min(1, swing || 0));
        const amp = Math.max(0, Math.min(1, bow || 0));
        const pts = new Array(n);
        pts[0] = [100, 100];
        const last = n - 1;
        for (let i = 1; i < last; i += 1) {
            const t = i / last;
            const env = Math.sin(Math.PI * t);
            const lead = Math.sin((phase || 0) - i * DANCER_JOINT_LAG);
            const off = side * amp * env * (DANCER_WHIP + DANCER_JOINT * lead);
            pts[i] = [base[i][0] + px * off, base[i][1] + py * off];
        }
        const R = Math.max(ARC_STEP * 3, reach * glass);
        const tr = Math.hypot(end[0] - 100, end[1] - 100) || 1;
        pts[last] = [100 + (end[0] - 100) / tr * R, 100 + (end[1] - 100) / tr * R];
        return pts;
    }

    function dancerPoints(d, seed) {
        const skin = PLASMA_SKIN;
        const lightSeed = seed == null ? skin.arcSeed : seed;
        const birth = d.birth || 0;
        const rock = (d.swing || 0) * Math.max(0, Math.min(1, d.bow || 0)) * DANCER_ROCK;
        const a0 = hash2(lightSeed, 7) * Math.PI * 2 + birth * DANCER_GOLDEN + d.angle + rock;
        let k = 0;
        const rnd = () => hash2(birth, (lightSeed + birth * 41) * 100 + 20 + (k++));
        const glass = skin.arcGlass;
        const stopR = Math.max(ARC_STEP * 3, d.reach * glass);
        const base = dieWalk(a0, rnd, skin.arcTurn, 0, skin.arcBias, stopR);
        return flexJoints(base, d.bow, d.swing, d.phase, d.reach, glass);
    }

    function liveDancer(id, now, drive) {
        if (!(now > 0)) return null;
        let d = iconDancers[id];
        const lift = speechLift(drive);
        if (!d) {
            if (lift <= 0) return null;
            d = createIconDancer(now, takeSpeakBirth(id, arcSeedFor(id)));
            iconDancers[id] = d;
        }
        let dt = (now - d.then) / 1000;
        d.then = now;
        if (!(dt > 0) || dt > 0.1) dt = 0.016;
        stepIconDancer(d, drive, dt, now);
        if (d.done) {
            delete iconDancers[id];
            if (lift > 0) {
                d = createIconDancer(now, takeSpeakBirth(id, arcSeedFor(id)));
                iconDancers[id] = d;
            } else {
                return null;
            }
        }
        return d;
    }

    function hexRgb(hex) {
        const n = parseInt(String(hex || '').replace('#', ''), 16);
        if (!Number.isFinite(n)) return { r: 154, g: 154, b: 154 };
        return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
    }

    function mixRgb(a, b, t) {
        const k = Math.max(0, Math.min(1, t));
        return {
            r: Math.round(a.r + (b.r - a.r) * k),
            g: Math.round(a.g + (b.g - a.g) * k),
            b: Math.round(a.b + (b.b - a.b) * k),
        };
    }

    function rgba(c, a) {
        return 'rgba(' + c.r + ',' + c.g + ',' + c.b + ',' + a + ')';
    }

    function mapPt(p, cx, cy, s) {
        return { x: cx + (p[0] - 100) * s, y: cy + (p[1] - 100) * s };
    }

    function strokeFilament(ctx, pts, s, cx, cy, warm, hi, opacity, heat) {
        if (!pts || pts.length < 2 || opacity < 0.02) return;
        const hot = Math.max(0, Math.min(1, heat || 0));
        ctx.globalAlpha = opacity;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.beginPath();
        const first = mapPt(pts[0], cx, cy, s);
        ctx.moveTo(first.x, first.y);
        for (let i = 1; i < pts.length; i += 1) {
            const at = mapPt(pts[i], cx, cy, s);
            ctx.lineTo(at.x, at.y);
        }
        if (hot > 0.02) {
            ctx.strokeStyle = rgba(hi, 0.28 + 0.5 * hot);
            ctx.lineWidth = (6 + 10 * hot) * s;
            ctx.stroke();
        }
        ctx.strokeStyle = rgba(warm, hot > 0.02 ? 0.45 + 0.4 * hot : 0.35);
        ctx.lineWidth = (hot > 0.02 ? 4.5 + 2.5 * hot : 4) * s;
        ctx.stroke();
        ctx.strokeStyle = 'rgba(255,255,255,' + (hot > 0.02 ? 1 : 0.92) + ')';
        ctx.lineWidth = (hot > 0.02 ? 1.8 + 1.4 * hot : 1.4) * s;
        ctx.stroke();
        const tip = PLASMA_SKIN.arcTip;
        if (tip > 0) {
            const P = pts[pts.length - 1];
            const rx = P[0] - 100;
            const ry = P[1] - 100;
            const rTip = Math.hypot(rx, ry);
            const squash = Math.sqrt(Math.max(0, 1 - (rTip / PLASMA_SKIN.arcGlass) ** 2));
            const deg = (Math.atan2(ry, rx) * 180) / Math.PI + 90;
            const at = mapPt(P, cx, cy, s);
            const tipBrush = ctx.createRadialGradient(at.x, at.y, 0, at.x, at.y, tip * s);
            tipBrush.addColorStop(0, 'rgba(255,255,255,1)');
            tipBrush.addColorStop(0.45, 'rgba(255,241,228,0.72)');
            tipBrush.addColorStop(1, rgba(warm, 0));
            ctx.fillStyle = tipBrush;
            ctx.beginPath();
            ctx.ellipse(at.x, at.y, tip * s, Math.max(0.4, tip * squash * s), (deg * Math.PI) / 180, 0, Math.PI * 2);
            ctx.fill();
        }
        ctx.globalAlpha = 1;
    }

    function drawPlasma(canvas, agentId, timeMs, drive, presence, now) {
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const w = canvas.clientWidth || 88;
        const h = canvas.clientHeight || 88;
        const pw = Math.max(1, Math.round(w * dpr));
        const ph = Math.max(1, Math.round(h * dpr));
        if (canvas.width !== pw || canvas.height !== ph) {
            canvas.width = pw;
            canvas.height = ph;
        }
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, w, h);
        const lift = speechLift(drive);
        const scale = Math.max(0.84, Math.min(1.12, presence.scale));
        const cx = w / 2;
        const cy = h / 2;
        const s = (Math.min(w, h) / 200) * scale;
        const skin = PLASMA_SKIN;
        const tone = colorOf(agentId);
        const cool = Math.max(0, Math.min(1, presence.cool));
        const blue = { r: 158, g: 191, b: 218 };
        const bodyTone = ballTones(tone.h, tone.sat);
        const lobeA = ballTones(tone.ha, tone.sa);
        const lobeB = ballTones(tone.hb, tone.sb);
        const coolMix = (c) => mixRgb(c, blue, cool * 0.7);
        const hi = coolMix(bodyTone.hi);
        const mid = coolMix(bodyTone.mid);
        const base = coolMix(bodyTone.base);
        const warm = mixRgb(bodyTone.warm, { r: 255, g: 196, b: 160 }, presence.warmth * 0.45);
        const dim = Math.max(0, Math.min(1, presence.dim));
        const amp = Math.max(0, Math.min(1, drive || 0));
        const glass = skin.arcGlass * s;
        const hole = skin.haloHole * s;
        const feather = skin.hbFeather / 150;
        const stop2 = (90 - 78 * feather) / 100;
        const stop3 = (96 - 64 * feather) / 100;
        const haze = coronaHaze(PLASMA_REST_GAIN, lift, presence.corona, presence.warmth, presence.scale);
        const lobeAlpha = Math.max(0.2, Math.min(1, skin.hbOpacity * (0.72 + haze)));

        ctx.save();
        ctx.beginPath();
        ctx.rect(0, 0, w, h);
        ctx.arc(cx, cy, hole, 0, Math.PI * 2, true);
        ctx.clip('evenodd');
        ctx.globalAlpha = lobeAlpha;
        const lobes = plasmaLobes(timeMs);
        for (let i = 0; i < lobes.length; i += 1) {
            const lobe = lobes[i];
            const palette = lobe.even ? lobeA : lobeB;
            const at = mapPt([lobe.x, lobe.y], cx, cy, s);
            const rx = lobe.rx * s;
            const ry = lobe.ry * s;
            const rot = (lobe.rot * Math.PI) / 180;
            const brush = ctx.createRadialGradient(at.x, at.y, 0, at.x, at.y, Math.max(rx, ry));
            brush.addColorStop(0, rgba(palette.hi, 0.9));
            brush.addColorStop(stop2, rgba(palette.mid, 0.72));
            brush.addColorStop(stop3, rgba(palette.base, 0.4));
            brush.addColorStop(1, rgba(palette.base, 0));
            ctx.fillStyle = brush;
            ctx.beginPath();
            ctx.ellipse(at.x, at.y, rx, ry, rot, 0, Math.PI * 2);
            ctx.fill();
        }
        ctx.restore();

        const bodyBrush = ctx.createRadialGradient(cx, cy, 0, cx, cy, glass);
        bodyBrush.addColorStop(0, rgba(hi, 0.42));
        bodyBrush.addColorStop(0.4, rgba(mid, 0.62 * (1 - dim * 0.25)));
        bodyBrush.addColorStop(0.96, rgba(base, 0.94 * (1 - dim * 0.2)));
        bodyBrush.addColorStop(1, rgba(base, 0));
        ctx.fillStyle = bodyBrush;
        ctx.beginPath();
        ctx.arc(cx, cy, glass, 0, Math.PI * 2);
        ctx.fill();

        ctx.save();
        ctx.beginPath();
        ctx.arc(cx, cy, glass, 0, Math.PI * 2);
        ctx.clip();
        const seed = arcSeedFor(agentId);
        const arcs = plasmaArcs(timeMs, drive, seed);
        for (let i = 0; i < arcs.length; i += 1) {
            const arc = arcs[i];
            strokeFilament(ctx, arc.pts, s, cx, cy, warm, hi, arc.opacity * (1 - dim * 0.45), 0);
        }
        const dancer = liveDancer(agentId, now, drive);
        if (dancer) {
            strokeFilament(
                ctx,
                dancerPoints(dancer, seed),
                s,
                cx,
                cy,
                warm,
                hi,
                dancer.opacity * (1 - dim * 0.35),
                dancer.bow
            );
        }
        const breathe = timeMs <= 0 ? 0.99 : 0.99 - 0.09 * Math.cos((timeMs / (skin.breathe * 1000)) * Math.PI * 2);
        const nucleus = 16 * breathe * (1 + 0.75 * amp) * s * (1 + presence.bloom * 0.18);
        const nucleusBrush = ctx.createRadialGradient(cx, cy, 0, cx, cy, nucleus);
        const hotA = Math.min(1, 0.92 + 0.08 * (lift + presence.bloom));
        nucleusBrush.addColorStop(0, 'rgba(255,255,255,' + hotA + ')');
        nucleusBrush.addColorStop(0.3, 'rgba(255,241,228,0.95)');
        nucleusBrush.addColorStop(0.64, rgba(warm, 0.5 + 0.25 * presence.bloom));
        nucleusBrush.addColorStop(1, rgba(base, 0));
        ctx.fillStyle = nucleusBrush;
        ctx.beginPath();
        ctx.arc(cx, cy, nucleus, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
    }

    function nodOffset(id, gesture, now) {
        if (gesture !== 'nod') {
            delete nodStarted[id];
            return 0;
        }
        if (!nodStarted[id]) nodStarted[id] = now;
        const t = now - nodStarted[id];
        const cycle = 360;
        if (t >= cycle * 2) return 0;
        const u = (t % cycle) / cycle;
        const amp = u < 0.5 ? u * 2 : (1 - u) * 2;
        return amp * 3.5;
    }

    function motionReduced() {
        return window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    }

    function drawMountedBalls(now) {
        if (!stageEl) return;
        const spinning = !motionReduced();
        stageEl.querySelectorAll('canvas.plasma-ball').forEach((canvas) => {
            const wrap = canvas.closest('.light-face-wrap');
            const id = wrap ? wrap.dataset.face : '';
            if (!id) return;
            const timeMs = spinning ? now : 0;
            const talking = Boolean(speakingId && id === speakingId);
            fitWindowBall(canvas, wrap);
            const pose = poses[id] || '';
            const gesture = gestures[id] || '';
            const overlay = overlays[id] || '';
            const intent = intents[id] || '';
            const shown = stepBall(id, presenceOf(pose, overlay, intent), now);
            const drive = !talking ? 0 : lipSyncActive ? mouthOpen : 0.28;
            drawPlasma(canvas, id, timeMs, drive, shown, spinning ? now : 0);
            const lean = gesture === 'tilt' || intent === 'receiving' ? PLASMA_HOST_LEAN : 0;
            const tilt = gesture === 'tilt' ? 6 : 0;
            const nod = nodOffset(id, gesture, now);
            canvas.style.transform =
                'translate(' + lean * 100 + '%, ' + nod + 'px) rotate(' + tilt + 'deg)';
        });
    }

    function kickBalls() {
        if (ballFrame || !stageEl || stageEl.hidden) return;
        const loop = (now) => {
            if (!stageEl || stageEl.hidden || !stageEl.querySelector('canvas.plasma-ball')) {
                ballFrame = 0;
                return;
            }
            drawMountedBalls(now);
            ballFrame = window.requestAnimationFrame(loop);
        };
        ballFrame = window.requestAnimationFrame(loop);
    }

    function ensureBall(wrap) {
        let canvas = wrap.querySelector('canvas.plasma-ball');
        if (canvas) return canvas;
        canvas = document.createElement('canvas');
        canvas.className = 'plasma-ball light-face';
        canvas.setAttribute('aria-hidden', 'true');
        wrap.insertBefore(canvas, wrap.firstChild);
        return canvas;
    }

    function ensureWrap(agentId) {
        if (!stageEl) return null;
        const id = String(agentId || 'lumen').toLowerCase();
        let wrap = stageEl.querySelector('[data-face="' + id + '"]');
        if (wrap) return wrap;
        wrap = document.createElement('div');
        wrap.className = 'light-face-wrap';
        wrap.dataset.face = id;
        wrap.style.setProperty('--face-glow', glowHex(id));
        ensureBall(wrap);
        const name = document.createElement('div');
        name.className = 'light-face-name';
        name.textContent = faceLabel(id);
        wrap.appendChild(name);
        stageEl.appendChild(wrap);
        applyStageVisibility();
        kickBalls();
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
        cancelClosedEyeReopen(key);
        delete poses[key];
        delete gestures[key];
        delete overlays[key];
        delete intents[key];
        delete ballMotion[key];
        delete nodStarted[key];
        if (speakingId === key) {
            speakingId = '';
            speakPlan = null;
        }
    }

    /** Drop balls that are no longer in the stage cast (1:1 replace, Group refresh). */
    function pruneTo(keepIds) {
        const keep = new Set(
            (keepIds || []).map((id) => String(id || '').toLowerCase()).filter(Boolean)
        );
        if (speakingId && keep.has(speakingId) === false) {
            // Keep the currently speaking ball visible until audio ends,
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
        Object.keys(intents).forEach((id) => {
            if (!keep.has(id)) forgetAgent(id);
        });
        Object.keys(ballMotion).forEach((id) => {
            if (!keep.has(id)) forgetAgent(id);
        });
        Object.keys(nodStarted).forEach((id) => {
            if (!keep.has(id)) forgetAgent(id);
        });
        if (!keep.size && stageEl) applyStageVisibility();
    }

    function visibleKeep() {
        if (!soloStage) return [...presentIds];
        const focus = speakingId || lastShownId;
        return focus ? [focus] : [];
    }

    function rememberCast(ids) {
        presentIds = new Set(
            (ids || []).map((id) => String(id || '').toLowerCase()).filter(Boolean)
        );
        if (
            soloStage &&
            lastShownId &&
            lastShownId !== speakingId &&
            !presentIds.has(lastShownId)
        ) {
            lastShownId = '';
        }
        pruneTo(visibleKeep());
        paint();
    }

    function noteShown(agentId) {
        const id = String(agentId || '').toLowerCase();
        if (!id || !soloStage) return;
        lastShownId = id;
        pruneTo(visibleKeep());
        paint();
    }

    function setSolo(on) {
        soloStage = Boolean(on);
        if (stageEl) {
            pruneTo(visibleKeep());
            paint();
        }
    }

    function readWindowGeom() {
        try {
            const raw = JSON.parse(localStorage.getItem(WINDOW_KEY) || '');
            if (!raw || typeof raw !== 'object') return null;
            return raw;
        } catch {
            return null;
        }
    }

    function clampGeom(raw) {
        const vw = window.innerWidth || 800;
        const vh = window.innerHeight || 600;
        const narrow = vw < 640;
        const fallbackW = narrow ? Math.min(240, vw - 24) : 300;
        const fallbackH = narrow ? 280 : 380;
        const width = Math.max(180, Math.min(Number(raw && raw.width) || fallbackW, Math.max(180, vw - 16)));
        const height = Math.max(200, Math.min(Number(raw && raw.height) || fallbackH, Math.max(200, vh - 16)));
        let left = Number(raw && raw.left);
        let top = Number(raw && raw.top);
        if (!Number.isFinite(left)) left = vw - width - (narrow ? 12 : 16);
        if (!Number.isFinite(top)) top = narrow ? 60 : 72;
        left = Math.max(8, Math.min(left, Math.max(8, vw - width - 8)));
        top = Math.max(8, Math.min(top, Math.max(8, vh - height - 8)));
        return {
            left: Math.round(left),
            top: Math.round(top),
            width: Math.round(width),
            height: Math.round(height),
        };
    }

    function applyGeom(geom) {
        if (!stageEl || !geom) return;
        stageEl.style.left = geom.left + 'px';
        stageEl.style.top = geom.top + 'px';
        stageEl.style.right = 'auto';
        stageEl.style.width = geom.width + 'px';
        stageEl.style.height = geom.height + 'px';
    }

    function saveWindowGeom() {
        if (!stageEl) return;
        const rect = stageEl.getBoundingClientRect();
        if (rect.width < 40 || rect.height < 40) return;
        const geom = clampGeom({
            left: rect.left,
            top: rect.top,
            width: rect.width,
            height: rect.height,
        });
        try {
            localStorage.setItem(WINDOW_KEY, JSON.stringify(geom));
        } catch {
            /* ignore quota / private mode */
        }
    }

    function restoreWindowGeom() {
        const geom = clampGeom(readWindowGeom());
        applyGeom(geom);
        try {
            localStorage.setItem(WINDOW_KEY, JSON.stringify(geom));
        } catch {
            /* ignore quota / private mode */
        }
    }

    function bindWindowDrag(bar) {
        if (!bar || bar.dataset.bound === '1') return;
        bar.dataset.bound = '1';
        bar.addEventListener('pointerdown', (ev) => {
            if (ev.button != null && ev.button !== 0) return;
            if (!stageEl) return;
            ev.preventDefault();
            const rect = stageEl.getBoundingClientRect();
            const originX = ev.clientX;
            const originY = ev.clientY;
            const originLeft = rect.left;
            const originTop = rect.top;
            bar.classList.add('is-dragging');
            try {
                bar.setPointerCapture(ev.pointerId);
            } catch {
                /* ignore */
            }
            const move = (e) => {
                const next = clampGeom({
                    left: originLeft + (e.clientX - originX),
                    top: originTop + (e.clientY - originY),
                    width: rect.width,
                    height: rect.height,
                });
                applyGeom(next);
            };
            const up = () => {
                bar.classList.remove('is-dragging');
                bar.removeEventListener('pointermove', move);
                bar.removeEventListener('pointerup', up);
                bar.removeEventListener('pointercancel', up);
                saveWindowGeom();
            };
            bar.addEventListener('pointermove', move);
            bar.addEventListener('pointerup', up);
            bar.addEventListener('pointercancel', up);
        });
    }

    function watchWindowResize() {
        if (!stageEl || windowResizeObserver || typeof ResizeObserver !== 'function') return;
        windowResizeObserver = new ResizeObserver(() => {
            saveWindowGeom();
        });
        windowResizeObserver.observe(stageEl);
        window.addEventListener('resize', () => {
            if (!stageEl) return;
            const rect = stageEl.getBoundingClientRect();
            applyGeom(
                clampGeom({
                    left: rect.left,
                    top: rect.top,
                    width: rect.width,
                    height: rect.height,
                })
            );
        });
    }

    function ensureWindowChrome() {
        if (!stageEl) return;
        stageEl.classList.add('face-window');
        stageEl.setAttribute('role', 'region');
        stageEl.setAttribute('aria-label', 'Stage');
        let bar = stageEl.querySelector('.face-window-bar');
        if (!bar) {
            bar = document.createElement('div');
            bar.className = 'face-window-bar';
            const title = document.createElement('div');
            title.className = 'face-window-title';
            title.textContent = 'Stage';
            bar.appendChild(title);
            stageEl.insertBefore(bar, stageEl.firstChild);
        }
        bindWindowDrag(bar);
        restoreWindowGeom();
        watchWindowResize();
    }

    function syncWindowLabel() {
        if (!stageEl) return;
        const title = stageEl.querySelector('.face-window-title');
        if (!title) return;
        const wrap = stageEl.querySelector('.light-face-wrap');
        const id = wrap ? wrap.dataset.face || '' : '';
        title.textContent = id ? faceLabel(id) : 'Stage';
    }

    function fitWindowBall(canvas, wrap) {
        if (!stageEl || !stageEl.classList.contains('face-window') || !wrap) return;
        const name = wrap.querySelector('.light-face-name');
        const nameH = name ? name.offsetHeight + 8 : 22;
        const side = Math.floor(Math.min(wrap.clientWidth - 12, wrap.clientHeight - nameH));
        if (side < 48) return;
        canvas.style.width = side + 'px';
        canvas.style.height = side + 'px';
    }

    function applyStageVisibility() {
        if (!stageEl) return;
        const wraps = stageEl.querySelectorAll('.light-face-wrap');
        const hasFaces = wraps.length > 0;
        stageEl.hidden = !stageVisible || !hasFaces;
        syncStageToggleLabels();
        if (!stageEl.hidden) kickBalls();
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
        if (soloStage) {
            visibleKeep().forEach((id) => ensureWrap(id));
        } else {
            presentIds.forEach((id) => ensureWrap(id));
            if (speakingId && presentIds.has(speakingId)) ensureWrap(speakingId);
        }
        const wraps = stageEl.querySelectorAll('.light-face-wrap');
        wraps.forEach((wrap) => {
            const id = wrap.dataset.face;
            const talking = Boolean(speakingId && id === speakingId);
            const onStage = presentIds.has(id) || talking;
            const pose = onStage ? poses[id] || '' : '';
            const gesture = onStage ? gestures[id] || '' : '';
            const overlay = onStage ? overlays[id] || '' : '';
            const intent = onStage ? intents[id] || '' : '';
            wrap.style.setProperty('--face-glow', glowHex(id));
            wrap.classList.toggle('is-talking', talking);
            wrap.classList.toggle('is-lip-sync', talking && lipSyncActive);
            wrap.classList.toggle('is-present', presentIds.has(id) && !talking);
            wrap.classList.toggle('has-gesture', Boolean(gesture));
            if (!talking || !lipSyncActive) wrap.classList.remove('is-lip-sync');
            ensureBall(wrap);
            applyFaceClasses(wrap, pose, gesture, overlay, talking);
            wrap.classList.toggle('emo-receiving', intent === 'receiving');
            const face = wrap.querySelector('.plasma-ball');
            applyFaceClasses(face, pose, gesture, overlay, talking);
        });
        applyStageVisibility();
        syncWindowLabel();
        kickBalls();
    }

    function mount(selectorOrEl) {
        stageEl =
            typeof selectorOrEl === 'string'
                ? document.querySelector(selectorOrEl)
                : selectorOrEl;
        if (!stageEl) return null;
        stageEl.classList.add('face-stage');
        ensureWindowChrome();
        paint();
        return stageEl;
    }

    function setPresent(agentId) {
        const id = String(agentId || '').toLowerCase();
        rememberCast(id ? [id] : []);
    }

    function setPresentMany(ids) {
        rememberCast(ids);
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

    function faceLabel(id) {
        const named = displayNames[id];
        if (named && String(named).trim()) return String(named).trim();
        return id.charAt(0).toUpperCase() + id.slice(1);
    }

    function applyFaceLabel(wrap, id) {
        const name = wrap && wrap.querySelector('.light-face-name');
        if (name) name.textContent = faceLabel(id);
    }

    function setNames(map) {
        if (!map || typeof map !== 'object') return;
        Object.keys(map).forEach((key) => {
            const id = String(key || '').toLowerCase();
            const label = String(map[key] || '').trim();
            if (id && label) displayNames[id] = label;
        });
        if (!stageEl) return;
        stageEl.querySelectorAll('[data-face]').forEach((wrap) => {
            applyFaceLabel(wrap, wrap.dataset.face || '');
        });
    }

    function setColors(map) {
        Object.keys(colorOverrides).forEach((key) => {
            delete colorOverrides[key];
        });
        if (map && typeof map === 'object') {
            Object.keys(map).forEach((key) => {
                const id = String(key || '').toLowerCase();
                const hex = normalizeHex(map[key]);
                if (id && hex) colorOverrides[id] = hex;
            });
        }
        paint();
    }

    function iterCues(text) {
        const out = [];
        const raw = String(text || '');
        CUE_RE.lastIndex = 0;
        let match;
        while ((match = CUE_RE.exec(raw))) {
            const inner = (match[1] || match[2] || match[3] || '').trim();
            if (inner) out.push({ start: match.index, inner: inner, end: match.index + match[0].length });
        }
        return out;
    }

    function iterProseActions(text) {
        const out = [];
        const raw = String(text || '');
        const cueSpans = iterCues(raw);
        PROSE_ACTION_RE.lastIndex = 0;
        let match;
        while ((match = PROSE_ACTION_RE.exec(raw))) {
            const inner = (match[1] || '').trim();
            if (!inner) continue;
            const start = match.index + (match[0].length - (match[1] || '').length);
            if (
                cueSpans.some(
                    (span) => start >= span.start && start < span.end
                )
            ) {
                continue;
            }
            out.push({ start: start, inner: inner, end: start + inner.length });
        }
        return out;
    }

    function classifyCue(inner) {
        const text = String(inner || '').trim();
        if (!text || SKIP_THINK_RE.test(text)) return {};
        let pose = '';
        let gesture = '';
        if (PAUSE_RE.test(text)) {
            pose = PAUSE_SMILE_RE.test(text) ? 'pause_smile' : 'pause';
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
        if (BRIGHT_RE.test(text)) found.overlay = 'bright';
        const intent = intentOf(text);
        if (intent) found.intent = intent;
        return stampClosedEyeHold(found);
    }

    function iterEvents(text) {
        const raw = String(text || '');
        const events = [];
        iterCues(raw).forEach((cue) => {
            const classified = classifyCue(cue.inner);
            if (classified.pose || classified.gesture || classified.overlay || classified.intent || classified.hold_ms) {
                events.push({ start: cue.start, end: cue.end, classified: classified });
            }
        });
        iterProseActions(raw).forEach((cue) => {
            const classified = classifyCue(cue.inner);
            if (classified.pose || classified.gesture || classified.overlay || classified.intent || classified.hold_ms) {
                events.push({
                    start: cue.start,
                    end: cue.end,
                    classified: classified,
                });
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
            found = stampClosedEyeHold(Object.assign({}, found, classified));
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

    function stampClosedEyeHold(found) {
        const out = Object.assign({}, found);
        if (CLOSED_EYE_POSES.has(out.pose)) {
            out.hold_ms = String(CLOSED_EYE_HOLD_MS);
        } else {
            delete out.hold_ms;
        }
        return out;
    }

    function openEyesAfterClosedPose(pose) {
        if (pose === 'pause_smile' || pose === 'laugh') return 'smile';
        return 'soft';
    }

    function cancelClosedEyeReopen(id) {
        const key = String(id || '').toLowerCase();
        if (!key) return;
        if (closedEyeTimers[key]) {
            clearTimeout(closedEyeTimers[key]);
            delete closedEyeTimers[key];
        }
    }

    function scheduleClosedEyeReopen(agentId, pose) {
        const id = String(agentId || '').toLowerCase();
        cancelClosedEyeReopen(id);
        if (!id || !CLOSED_EYE_POSES.has(pose)) return;
        closedEyeTimers[id] = setTimeout(() => {
            delete closedEyeTimers[id];
            if (poses[id] !== pose) return;
            poses[id] = openEyesAfterClosedPose(pose);
            paint();
        }, CLOSED_EYE_HOLD_MS);
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
            Object.keys(intents).forEach((id) => {
                delete intents[id];
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
        if (parsed.intent) intents[id] = parsed.intent;
        else if (!holdIfNone) delete intents[id];
        if (parsed.hold_ms) {
            const ms = Number(parsed.hold_ms);
            emotionHoldMs = Number.isFinite(ms) && ms > 0 ? ms : EMO_HOLD_MS;
        } else if (CLOSED_EYE_POSES.has(parsed.pose)) {
            emotionHoldMs = CLOSED_EYE_HOLD_MS;
        } else if (parsed.pose || parsed.gesture || parsed.overlay || parsed.intent) {
            emotionHoldMs = EMO_HOLD_MS;
        }
        if (parsed.pose) {
            scheduleClosedEyeReopen(id, parsed.pose);
        }
        paint();
    }

    function emoteFromText(agentId, text, options) {
        const opts = options || {};
        noteShown(agentId);
        const parsed = emotionFromText(text, opts);
        if (!parsed.pose && !parsed.gesture && !parsed.overlay && !parsed.intent && !parsed.hold_ms) {
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
        kickBalls();
    }

    function setSpeaking(agentId, chunkText) {
        speakingId = String(agentId || '').toLowerCase();
        if (speakingId) lastShownId = speakingId;
        if (soloStage) pruneTo(visibleKeep());
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
            });
        }
        pruneTo(visibleKeep());
        paint();
        scheduleEmotionClear();
    }

    LH.faces = {
        mount,
        setPresent,
        setPresentMany,
        setSolo,
        noteShown,
        setVoices,
        setColors,
        setNames,
        setSpeaking,
        setMouthOpen,
        syncSpeakingProgress,
        clearSpeaking,
        emoteFromText,
        emotionFromText,
        emotionTimeline,
        classifyCue,
        intentOf,
        presenceOf,
        grainOf,
        plasmaArcs,
        plasmaArcCount,
        plasmaLobes,
        isStageVisible,
        setStageVisible,
        setupStageToggle,
    };
})(typeof window !== 'undefined' ? window : globalThis);
