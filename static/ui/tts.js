(function (global) {
    'use strict';

    const LH = (global.LightHouse = global.LightHouse || {});
    const VOICE_KEY = 'light_house_voice_enabled';
    const MAX_CHUNK_CHARS = 280;
    // Keep clips long enough that playback usually outlasts Kokoro synth of N+1
    // on this CPU box (~0.3–1.3s per sentence). Tiny clips caused 0.5–4s stalls.
    const MIN_CHUNK_CHARS = 72;
    // Start the next sentence this many seconds before the current one ends,
    // but only if N+1 is already decoded — never cut into a synth wait.
    const CHUNK_HANDOFF_SEC = 0.09;

    let serverReady = false;
    let currentAudio = null;
    let speakSeq = 0;
    let activeSpeakBtn = null;
    /** @type {Set<HTMLAudioElement>} */
    const liveAudios = new Set();
    /** True while a speak() job (all its chunks) is in progress. */
    let speakBusy = false;
    /**
     * Auto-speak queue (group chat, etc.). Manual replay uses force and clears this.
     * @type {Array<{text: string, agentId: string, options: object}>}
     */
    const speakQueue = [];
    const MAX_SPEAK_QUEUE = 12;

    /** @type {AudioContext|null} */
    let audioCtx = null;
    /** @type {AnalyserNode|null} */
    let analyser = null;
    /** @type {WeakMap<HTMLAudioElement, MediaElementAudioSourceNode>} */
    const mediaSources = new WeakMap();
    let lipRaf = 0;
    let lipSmooth = 0;

    const SPEAKER_SVG =
        '<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true" focusable="false">' +
        '<path fill="currentColor" d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z"/>' +
        '</svg>';

    function isVoiceOn() {
        return localStorage.getItem(VOICE_KEY) === '1';
    }

    function setVoiceOn(on) {
        localStorage.setItem(VOICE_KEY, on ? '1' : '0');
        syncToggleLabels();
        if (!on) stop();
    }

    function facesApi() {
        return LH.faces || null;
    }

    function prefersReducedMotion() {
        try {
            return Boolean(
                global.matchMedia &&
                    global.matchMedia('(prefers-reduced-motion: reduce)').matches
            );
        } catch {
            return false;
        }
    }

    function ensureAnalyser() {
        const AC = global.AudioContext || global.webkitAudioContext;
        if (!AC) return null;
        if (!audioCtx) {
            audioCtx = new AC();
            analyser = audioCtx.createAnalyser();
            analyser.fftSize = 256;
            analyser.smoothingTimeConstant = 0.4;
            // Route through analyser so MediaElementSource still reaches speakers.
            analyser.connect(audioCtx.destination);
        }
        if (audioCtx.state === 'suspended') {
            audioCtx.resume().catch(() => {});
        }
        return analyser;
    }

    function connectLipSyncSource(audio) {
        const node = ensureAnalyser();
        if (!node || !audioCtx || !audio) return false;
        try {
            let src = mediaSources.get(audio);
            if (!src) {
                src = audioCtx.createMediaElementSource(audio);
                mediaSources.set(audio, src);
                src.connect(node);
            }
            return true;
        } catch (err) {
            console.warn('TTS lip-sync connect failed:', err);
            return false;
        }
    }

    function stopLipSync() {
        if (lipRaf) {
            cancelAnimationFrame(lipRaf);
            lipRaf = 0;
        }
        lipSmooth = 0;
        const faces = facesApi();
        if (faces && typeof faces.setMouthOpen === 'function') {
            faces.setMouthOpen(0);
        }
    }

    function startLipSync(audio, seq) {
        stopLipSync();
        if (prefersReducedMotion()) return;
        if (!connectLipSyncSource(audio) || !analyser) return;
        const data = new Uint8Array(analyser.fftSize);
        const tick = () => {
            if (seq !== speakSeq || currentAudio !== audio) {
                stopLipSync();
                return;
            }
            analyser.getByteTimeDomainData(data);
            let sum = 0;
            for (let i = 0; i < data.length; i += 1) {
                const v = (data[i] - 128) / 128;
                sum += v * v;
            }
            const rms = Math.sqrt(sum / data.length);
            // Quiet floor → open mouth on speech peaks (Kokoro WAV is fairly hot).
            const raw = Math.min(1, Math.max(0, (rms - 0.018) / 0.22));
            lipSmooth = lipSmooth * 0.5 + raw * 0.5;
            const faces = facesApi();
            if (faces && typeof faces.setMouthOpen === 'function') {
                faces.setMouthOpen(lipSmooth);
            }
            lipRaf = global.requestAnimationFrame(tick);
        };
        lipRaf = global.requestAnimationFrame(tick);
    }

    function clearSpeakingUi() {
        if (activeSpeakBtn) {
            activeSpeakBtn.classList.remove('is-speaking');
            activeSpeakBtn.setAttribute('aria-pressed', 'false');
            activeSpeakBtn = null;
        }
        document.querySelectorAll('.message-speak.is-speaking').forEach((el) => {
            el.classList.remove('is-speaking');
            el.setAttribute('aria-pressed', 'false');
        });
        stopLipSync();
        const faces = facesApi();
        if (faces && typeof faces.clearSpeaking === 'function') {
            faces.clearSpeaking();
        }
    }

    function syncToggleLabels() {
        document.querySelectorAll('[data-voice-toggle]').forEach((el) => {
            const on = isVoiceOn();
            if (el.hasAttribute('data-compact')) {
                el.textContent = 'Voice';
                el.setAttribute('aria-pressed', on ? 'true' : 'false');
                el.classList.toggle('is-on', on);
            } else {
                el.textContent = on ? 'Voice off' : 'Voice on';
                el.removeAttribute('aria-pressed');
                el.classList.remove('is-on');
            }
            el.hidden = false;
            el.disabled = false;
            el.removeAttribute('hidden');
        });
        document.querySelectorAll('.message-speak').forEach((el) => {
            el.hidden = !serverReady;
            el.disabled = !serverReady;
        });
    }

    function stop() {
        speakSeq += 1;
        speakQueue.length = 0;
        speakBusy = false;
        for (const audio of liveAudios) {
            try {
                audio.pause();
            } catch {
                /* ignore */
            }
        }
        liveAudios.clear();
        currentAudio = null;
        clearSpeakingUi();
    }

    /**
     * Split assistant text into speakable chunks (sentence-ish).
     * Exported for tests / future token streaming.
     */
    function splitIntoSpeechChunks(text) {
        const raw = String(text || '').replace(/\r\n/g, '\n').trim();
        if (!raw) return [];

        // Protect common abbreviations so "Dr. Jones" stays one piece.
        const protectedText = raw
            .replace(/\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|e\.g|i\.e|U\.S|U\.K)\./gi, (m) =>
                m.replace(/\./g, '\u2024')
            );

        const rough = [];
        const paragraphParts = protectedText.split(/\n{2,}/);
        for (const para of paragraphParts) {
            const line = para.replace(/\s*\n\s*/g, ' ').trim();
            if (!line) continue;
            const pieces = line.match(/[^.!?]+(?:[.!?]+["'\u201d\u2019]*)?|[^.!?]+$/g);
            if (pieces) {
                for (const p of pieces) {
                    const piece = p.replace(/\u2024/g, '.').trim();
                    if (piece) rough.push(piece);
                }
            } else {
                rough.push(line.replace(/\u2024/g, '.').trim());
            }
        }

        const merged = [];
        for (const piece of rough) {
            if (
                merged.length &&
                (merged[merged.length - 1].length < MIN_CHUNK_CHARS ||
                    piece.length < MIN_CHUNK_CHARS)
            ) {
                merged[merged.length - 1] = (merged[merged.length - 1] + ' ' + piece).trim();
            } else {
                merged.push(piece);
            }
        }

        const chunks = [];
        for (const piece of merged) {
            if (piece.length <= MAX_CHUNK_CHARS) {
                chunks.push(piece);
                continue;
            }
            let rest = piece;
            while (rest.length > MAX_CHUNK_CHARS) {
                let cut = rest.lastIndexOf(' ', MAX_CHUNK_CHARS);
                if (cut < MAX_CHUNK_CHARS * 0.5) cut = MAX_CHUNK_CHARS;
                chunks.push(rest.slice(0, cut).trim());
                rest = rest.slice(cut).trim();
            }
            if (rest) chunks.push(rest);
        }
        return chunks.filter(Boolean);
    }

    async function refreshServerStatus() {
        try {
            const res = await fetch('/v1/tts/status', { cache: 'no-store' });
            if (!res.ok) {
                serverReady = false;
                syncToggleLabels();
                return false;
            }
            const data = await res.json();
            serverReady = Boolean(data.enabled && data.ready);
            const faces = facesApi();
            if (faces && typeof faces.setVoices === 'function' && data.voices) {
                faces.setVoices(data.voices);
            }
            syncToggleLabels();
            return serverReady;
        } catch {
            serverReady = false;
            syncToggleLabels();
            return false;
        }
    }

    async function fetchSpeechBlob(text, agentId, voice) {
        const payload = {
            text,
            agent_id: agentId || 'lumen',
        };
        if (voice) payload.voice = voice;
        const res = await fetch('/v1/tts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        if (!res.ok) {
            const detail = (await res.json().catch(() => ({}))).detail;
            throw new Error(detail || `TTS HTTP ${res.status}`);
        }
        return res.blob();
    }

    function prepareAudio(blob) {
        const url = URL.createObjectURL(blob);
        const audio = new Audio();
        audio.preload = 'auto';
        audio.src = url;
        return new Promise((resolve, reject) => {
            let settled = false;
            const ok = () => {
                if (settled) return;
                settled = true;
                resolve({ audio, url });
            };
            const fail = () => {
                if (settled) return;
                settled = true;
                URL.revokeObjectURL(url);
                reject(new Error('Audio failed to load'));
            };
            audio.addEventListener('canplaythrough', ok, { once: true });
            audio.addEventListener('error', fail, { once: true });
            // Some browsers never fire canplaythrough for short clips.
            audio.addEventListener('loadeddata', ok, { once: true });
            try {
                audio.load();
            } catch {
                ok();
            }
        });
    }

    function notifySpeaking(agentId, chunkText) {
        const faces = facesApi();
        if (faces && typeof faces.setSpeaking === 'function' && agentId) {
            faces.setSpeaking(agentId, chunkText || '');
        }
    }

    /**
     * Play one clip. May resolve slightly before it ends so the next sentence
     * can start without a full HTMLAudioElement gap — but only when the next
     * clip is already ready (see canHandoff). Otherwise play to the end so we
     * don't replace audio with silence while Kokoro is still synthesizing.
     */
    function playPrepared(prepared, seq, { canHandoff = null, agentId = null, chunkText = '' } = {}) {
        return new Promise((resolve) => {
            if (!prepared || seq !== speakSeq) {
                resolve();
                return;
            }
            const { audio, url } = prepared;
            let done = false;
            const finish = (revokeNow) => {
                if (done) return;
                done = true;
                audio.removeEventListener('timeupdate', onTime);
                audio.removeEventListener('ended', onEnded);
                audio.removeEventListener('error', onEnded);
                audio.removeEventListener('playing', onPlaying);
                if (revokeNow) {
                    liveAudios.delete(audio);
                    try {
                        URL.revokeObjectURL(url);
                    } catch {
                        /* ignore */
                    }
                } else {
                    // Let the tail finish; revoke when it truly ends.
                    const cleanup = () => {
                        liveAudios.delete(audio);
                        try {
                            URL.revokeObjectURL(url);
                        } catch {
                            /* ignore */
                        }
                        audio.removeEventListener('ended', cleanup);
                        audio.removeEventListener('error', cleanup);
                    };
                    audio.addEventListener('ended', cleanup, { once: true });
                    audio.addEventListener('error', cleanup, { once: true });
                }
                if (currentAudio === audio) currentAudio = null;
                resolve();
            };
            const onEnded = () => finish(true);
            const onPlaying = () => {
                if (seq !== speakSeq) return;
                notifySpeaking(agentId || 'lumen', chunkText);
                startLipSync(audio, seq);
            };
            const onTime = () => {
                if (seq !== speakSeq) return;
                const dur = audio.duration;
                if (dur && !Number.isNaN(dur) && dur > 0) {
                    const faces = facesApi();
                    if (faces && typeof faces.syncSpeakingProgress === 'function') {
                        faces.syncSpeakingProgress(audio.currentTime / dur);
                    }
                }
                if (typeof canHandoff !== 'function' || !canHandoff()) return;
                if (!dur || Number.isNaN(dur)) return;
                if (audio.currentTime >= Math.max(0, dur - CHUNK_HANDOFF_SEC)) {
                    finish(false);
                }
            };

            currentAudio = audio;
            liveAudios.add(audio);
            audio.addEventListener('playing', onPlaying);
            audio.addEventListener('timeupdate', onTime);
            audio.addEventListener('ended', onEnded);
            audio.addEventListener('error', onEnded);
            const playResult = audio.play();
            if (playResult && typeof playResult.catch === 'function') {
                playResult.catch((err) => {
                    console.warn('TTS play failed:', err);
                    liveAudios.delete(audio);
                    finish(true);
                });
            }
        });
    }

    async function speak(text, agentId, options) {
        const opts = options || {};
        const force = Boolean(opts.force);
        const body = (text || '').trim();
        if (!body) return;
        if (!serverReady) {
            // Still show face emotes even if TTS server is down.
            const faces = facesApi();
            if (faces && typeof faces.emoteFromText === 'function') {
                faces.emoteFromText(agentId || 'lumen', body, {
                    first: true,
                    persist: false,
                });
            }
            return;
        }
        if (!force && !isVoiceOn()) {
            const faces = facesApi();
            if (faces && typeof faces.emoteFromText === 'function') {
                faces.emoteFromText(agentId || 'lumen', body, {
                    first: true,
                    persist: false,
                });
            }
            return;
        }

        const job = {
            text: body,
            agentId: agentId || 'lumen',
            options: opts,
        };

        // Manual replay / explicit interrupt: drop the queue and cut in.
        if (force) {
            speakQueue.length = 0;
            return runSpeakJob(job, { interrupt: true });
        }

        // Auto-speak (group): finish the current light before starting the next.
        if (speakBusy) {
            if (speakQueue.length >= MAX_SPEAK_QUEUE) {
                speakQueue.shift();
            }
            speakQueue.push(job);
            return;
        }
        return runSpeakJob(job, { interrupt: false });
    }

    /**
     * Run one utterance (all sentence chunks). When it finishes cleanly, start
     * the next queued auto-speak job if any.
     */
    async function runSpeakJob(job, { interrupt }) {
        const body = job.text;
        const agentId = job.agentId;
        const opts = job.options || {};
        const faces = facesApi();
        if (faces && typeof faces.emoteFromText === 'function') {
            faces.emoteFromText(agentId, body, {
                first: true,
                persist: true,
            });
        }

        const chunks = splitIntoSpeechChunks(body);
        if (!chunks.length) {
            pumpSpeakQueue();
            return;
        }

        if (interrupt) {
            // Same as former speak() start: cancel anything in flight.
            speakSeq += 1;
            for (const audio of liveAudios) {
                try {
                    audio.pause();
                } catch {
                    /* ignore */
                }
            }
            liveAudios.clear();
            currentAudio = null;
            stopLipSync();
        }

        speakBusy = true;
        const seq = speakSeq;

        const speakBtn = opts.button || null;
        clearSpeakingUi();
        if (speakBtn) {
            activeSpeakBtn = speakBtn;
            speakBtn.classList.add('is-speaking');
            speakBtn.setAttribute('aria-pressed', 'true');
        }

        const voice = opts.voice || null;

        let nextPrepared = null;
        let nextPreparedPromise = fetchSpeechBlob(chunks[0], agentId, voice)
            .then(prepareAudio)
            .then((prepared) => {
                nextPrepared = prepared;
                return prepared;
            });
        try {
            for (let i = 0; i < chunks.length; i += 1) {
                if (seq !== speakSeq) return;
                let prepared;
                try {
                    prepared = await nextPreparedPromise;
                } catch (err) {
                    console.warn('TTS failed:', err);
                    if (seq === speakSeq) clearSpeakingUi();
                    return;
                }
                if (seq !== speakSeq) return;

                const hasNext = i + 1 < chunks.length;
                nextPrepared = null;
                if (hasNext) {
                    nextPreparedPromise = fetchSpeechBlob(
                        chunks[i + 1],
                        agentId,
                        voice
                    )
                        .then(prepareAudio)
                        .then((p) => {
                            nextPrepared = p;
                            return p;
                        });
                }

                await playPrepared(prepared, seq, {
                    canHandoff: hasNext ? () => nextPrepared != null : null,
                    agentId: agentId,
                    chunkText: chunks[i] || '',
                });
            }
            if (seq === speakSeq) clearSpeakingUi();
        } finally {
            // Only the job that still owns speakSeq may clear busy / pump.
            // An interrupted job must not clobber a newer force/stop owner.
            if (seq === speakSeq) {
                speakBusy = false;
                pumpSpeakQueue();
            }
        }
    }

    function pumpSpeakQueue() {
        if (speakBusy) return;
        if (!isVoiceOn() || !serverReady) {
            speakQueue.length = 0;
            return;
        }
        const next = speakQueue.shift();
        if (!next) return;
        runSpeakJob(next, { interrupt: false });
    }

    /**
     * Replay a bubble's text aloud (ignores Voice on/off toggle).
     * Pressing again while this bubble is playing stops playback.
     */
    function replay(text, agentId, button, voice) {
        if (!serverReady) return Promise.resolve();
        if (button && button.classList.contains('is-speaking')) {
            stop();
            return Promise.resolve();
        }
        const resolvedVoice =
            voice || (button && button.dataset.voice) || null;
        return speak(text, agentId, {
            force: true,
            button: button || null,
            voice: resolvedVoice,
        });
    }

    function attachSpeakControl(messageEl, text, agentId, voice) {
        try {
            if (!messageEl) return null;
            if (messageEl.querySelector('.message-speak')) return null;
            const body = (text || '').trim();
            if (!body) return null;

            const tray =
                (LH.bubbleActions &&
                    typeof LH.bubbleActions.ensureActionsTray === 'function' &&
                    LH.bubbleActions.ensureActionsTray(messageEl)) ||
                null;

            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'message-action message-speak';
            btn.setAttribute('aria-label', 'Play aloud');
            btn.setAttribute('title', 'Play aloud');
            btn.setAttribute('aria-pressed', 'false');
            btn.innerHTML = SPEAKER_SVG;
            btn.hidden = !serverReady;
            btn.disabled = !serverReady;
            btn.dataset.agentId = agentId || 'lumen';
            if (voice) btn.dataset.voice = voice;
            btn.addEventListener('click', (ev) => {
                ev.preventDefault();
                ev.stopPropagation();
                replay(
                    body,
                    btn.dataset.agentId || agentId || 'lumen',
                    btn,
                    btn.dataset.voice || voice || null
                );
            });
            if (tray) tray.appendChild(btn);
            else {
                messageEl.classList.add('has-actions');
                messageEl.appendChild(btn);
            }
            return btn;
        } catch (err) {
            console.warn('attachSpeakControl failed:', err);
            return null;
        }
    }

    function setupVoiceToggle(buttonOrSelector) {
        const toggle =
            typeof buttonOrSelector === 'string'
                ? document.querySelector(buttonOrSelector)
                : buttonOrSelector;
        if (!toggle) return;
        toggle.setAttribute('data-voice-toggle', '');
        toggle.hidden = false;
        toggle.removeAttribute('hidden');
        syncToggleLabels();
        toggle.addEventListener('click', () => {
            setVoiceOn(!isVoiceOn());
        });
        refreshServerStatus();
    }

    function speakAgent(agentId, text, options) {
        return speak(text, agentId, options);
    }

    LH.tts = {
        VOICE_KEY,
        isVoiceOn,
        setVoiceOn,
        stop,
        speak,
        speakAgent,
        replay,
        attachSpeakControl,
        splitIntoSpeechChunks,
        refreshServerStatus,
        setupVoiceToggle,
        isServerReady: () => serverReady,
    };
})(typeof window !== 'undefined' ? window : globalThis);
