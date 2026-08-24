(function (global) {
    'use strict';

    const LH = (global.LightHouse = global.LightHouse || {});

    const NOTE_EXT = '(?:md|txt|markdown)';
    // Path body after a known root (shared/, agent/, or relative private folder).
    const PATH_BODY = '[A-Za-z0-9_.\\-/]+';
    const PRIVATE_ROOTS =
        'writing|journal|memory|mailbox|ideas|reports|plans|briefs|gallery|persona';

    function escapeHtml(text) {
        return String(text || '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    function notesUrl(file, agentId) {
        let href = '/notes.html?file=' + encodeURIComponent(file);
        if (agentId) {
            href += '&agent=' + encodeURIComponent(agentId);
        }
        return href;
    }

    function normalizeNoteFile(raw, agentId, agentIds) {
        let p = String(raw || '')
            .trim()
            .replace(/\\/g, '/')
            .replace(/^\.\//, '');
        if (!p) return '';
        if (p.startsWith('notes/')) p = p.slice('notes/'.length);
        if (p.startsWith('shared/')) return p;
        const first = (p.split('/')[0] || '').toLowerCase();
        const ids = new Set(
            (agentIds || []).filter(Boolean).map((id) => String(id).toLowerCase())
        );
        // Already agent-prefixed (ara/writing/…); keep as listed in the notes API.
        if (ids.has(first) || (agentId && first === String(agentId).toLowerCase())) {
            return p;
        }
        // Relative private path (writing/…, journal/…) → current light's folder.
        if (agentId) return agentId + '/' + p.replace(/^\//, '');
        return p;
    }

    function isNotePathCandidate(raw) {
        const p = String(raw || '').trim();
        if (!p || /\s/.test(p)) return false;
        if (/^https?:\/\//i.test(p)) return false;
        if (!new RegExp('\\.' + NOTE_EXT + '$', 'i').test(p)) return false;
        if (p.length > 220) return false;
        return true;
    }

    function looksLikeNotePath(raw, options) {
        if (!isNotePathCandidate(raw)) return false;
        let p = String(raw).trim().replace(/\\/g, '/');
        if (p.startsWith('notes/')) p = p.slice('notes/'.length);
        if (p.startsWith('shared/')) return true;
        const agentId = options && options.agentId;
        const agentIds = (options && options.agentIds) || [];
        const ids = new Set(
            [agentId].concat(agentIds).filter(Boolean).map((id) => String(id).toLowerCase())
        );
        const first = p.split('/')[0].toLowerCase();
        if (p.includes('/')) {
            if (ids.has(first)) return true;
            // Before lights load, allow light-id shaped prefixes (ara/…, lumen/…).
            if (
                ids.size === 0 &&
                /^[a-z][a-z0-9_-]{1,31}$/i.test(first) &&
                !['http', 'https', 'www', 'ftp', 'mailto', 'file'].includes(first)
            ) {
                return true;
            }
        }
        if (new RegExp('^(?:' + PRIVATE_ROOTS + ')/', 'i').test(p)) return true;
        // Backtick-only: allow a single private file at agent root (policy.md).
        if (options && options.allowBareFile && agentId && !p.includes('/')) return true;
        return false;
    }

    function mdLink(label, file, agentId) {
        // Angle-bracket destination avoids ')' issues in rare paths.
        return '[' + label + '](<' + notesUrl(file, agentId) + '>)';
    }

    function htmlLink(label, file, agentId) {
        return (
            '<a href="' +
            escapeHtml(notesUrl(file, agentId)) +
            '" title="Open note">' +
            escapeHtml(label) +
            '</a>'
        );
    }

    function splitFenced(text) {
        const parts = [];
        const re = /(```[\s\S]*?```|~~~[\s\S]*?~~~)/g;
        let last = 0;
        let m;
        while ((m = re.exec(text)) !== null) {
            if (m.index > last) {
                parts.push({ code: false, text: text.slice(last, m.index) });
            }
            parts.push({ code: true, text: m[0] });
            last = m.index + m[0].length;
        }
        if (last < text.length) parts.push({ code: false, text: text.slice(last) });
        return parts.length ? parts : [{ code: false, text: text }];
    }

    function replaceInOutsideLinks(segment, replacer) {
        // Skip existing markdown links: [label](dest) or [label](<dest>)
        const parts = segment.split(/(\[[^\]]*\]\([^)]*\))/g);
        return parts
            .map((part, i) => {
                if (i % 2 === 1) return part;
                return replacer(part);
            })
            .join('');
    }

    function linkifySegmentAsMarkdown(segment, options) {
        const agentId = options && options.agentId;

        let out = replaceInOutsideLinks(segment, (plain) => {
            // Inline `note/path.md` → markdown link (keep visible path as label).
            return plain.replace(/`([^`\n]+)`/g, (full, inner) => {
                if (!looksLikeNotePath(inner, { ...options, allowBareFile: true })) {
                    return full;
                }
                const file = normalizeNoteFile(inner, agentId, options && options.agentIds);
                if (!file) return full;
                return mdLink(inner.trim(), file, agentId);
            });
        });

        out = replaceInOutsideLinks(out, (plain) => {
            const bareRe = new RegExp(
                '(^|[^\\w./`-])((?:notes/)?(?:shared|[A-Za-z][A-Za-z0-9_-]*)/(?:' +
                    PATH_BODY +
                    ')\\.' +
                    NOTE_EXT +
                    ')',
                'gi'
            );
            return plain.replace(bareRe, (full, prefix, path) => {
                if (!looksLikeNotePath(path, options)) return full;
                const file = normalizeNoteFile(path, agentId, options && options.agentIds);
                if (!file) return full;
                return prefix + mdLink(path, file, agentId);
            });
        });

        // Relative private roots without backticks: writing/foo.md
        out = replaceInOutsideLinks(out, (plain) => {
            const relRe = new RegExp(
                '(^|[^\\w./`-])((?:' +
                    PRIVATE_ROOTS +
                    ')/(?:' +
                    PATH_BODY +
                    ')\\.' +
                    NOTE_EXT +
                    ')',
                'gi'
            );
            return plain.replace(relRe, (full, prefix, path) => {
                if (!agentId) return full;
                if (!looksLikeNotePath(path, options)) return full;
                const file = normalizeNoteFile(path, agentId, options && options.agentIds);
                if (!file) return full;
                return prefix + mdLink(path, file, agentId);
            });
        });

        return out;
    }

    function linkifyNotePathsAsMarkdown(text, options) {
        const opts = options || {};
        return splitFenced(String(text || ''))
            .map((part) => (part.code ? part.text : linkifySegmentAsMarkdown(part.text, opts)))
            .join('');
    }

    function linkifyEscapedSegmentAsHtml(escapedSegment, options) {
        const agentId = options && options.agentId;
        // Work on escaped text: paths only use safe chars, so they survive escapeHtml.
        let out = escapedSegment;

        out = out.replace(/`([^`\n]+)`/g, (full, inner) => {
            if (!looksLikeNotePath(inner, { ...options, allowBareFile: true })) {
                return full;
            }
            const file = normalizeNoteFile(inner, agentId, options && options.agentIds);
            if (!file) return full;
            return htmlLink(inner.trim(), file, agentId);
        });

        const bareRe = new RegExp(
            '(^|[^\\w./`-])((?:notes/)?(?:shared|[A-Za-z][A-Za-z0-9_-]*)/(?:' +
                PATH_BODY +
                ')\\.' +
                NOTE_EXT +
                ')',
            'gi'
        );
        out = out.replace(bareRe, (full, prefix, path) => {
            if (!looksLikeNotePath(path, options)) return full;
            const file = normalizeNoteFile(path, agentId, options && options.agentIds);
            if (!file) return full;
            return prefix + htmlLink(path, file, agentId);
        });

        if (agentId) {
            const relRe = new RegExp(
                '(^|[^\\w./`-])((?:' +
                    PRIVATE_ROOTS +
                    ')/(?:' +
                    PATH_BODY +
                    ')\\.' +
                    NOTE_EXT +
                    ')',
                'gi'
            );
            out = out.replace(relRe, (full, prefix, path) => {
                if (!looksLikeNotePath(path, options)) return full;
                const file = normalizeNoteFile(path, agentId, options && options.agentIds);
                if (!file) return full;
                return prefix + htmlLink(path, file, agentId);
            });
        }

        return out;
    }

    function renderPlain(text, options) {
        const escaped = escapeHtml(text || '').replace(/\n/g, '<br>');
        return linkifyEscapedSegmentAsHtml(escaped, options || {});
    }

    function render(text, options) {
        const opts = options || {};
        const raw = linkifyNotePathsAsMarkdown(text || '', opts);
        if (typeof marked !== 'undefined' && typeof DOMPurify !== 'undefined') {
            return DOMPurify.sanitize(marked.parse(raw));
        }
        return renderPlain(raw, opts);
    }

    function setElementContent(element, text, { asMarkdown = true, agentId = null, agentIds = null } = {}) {
        const options = { agentId, agentIds: agentIds || undefined };
        if (asMarkdown) {
            element.classList.add('message-markdown');
            element.innerHTML = render(text, options);
        } else {
            element.classList.remove('message-markdown');
            element.textContent = text;
        }
    }

    LH.markdown = {
        render,
        renderPlain,
        setElementContent,
        notesUrl,
        normalizeNoteFile,
        linkifyNotePathsAsMarkdown,
        looksLikeNotePath,
    };
})(typeof window !== 'undefined' ? window : globalThis);
