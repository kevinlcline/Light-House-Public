/* Attach one picture to a chat or group prompt. */
(function () {
    const LightHouse = (window.LightHouse = window.LightHouse || {});
    const MAX_EDGE = 1600;
    const TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/gif'];

    function readBlob(blob) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(String(reader.result || ''));
            reader.onerror = () => reject(new Error('Could not read that image.'));
            reader.readAsDataURL(blob);
        });
    }

    function encodeCanvas(canvas, type, quality) {
        return new Promise((resolve, reject) => {
            canvas.toBlob(
                (blob) => {
                    if (!blob) reject(new Error('Could not read that image.'));
                    else resolve(blob);
                },
                type,
                quality
            );
        });
    }

    async function fileToImage(file) {
        if (!file || TYPES.indexOf(file.type) < 0) {
            throw new Error('Use a JPEG, PNG, WebP, or GIF.');
        }
        const objectUrl = URL.createObjectURL(file);
        try {
            const img = await new Promise((resolve, reject) => {
                const el = new Image();
                el.onload = () => resolve(el);
                el.onerror = () => reject(new Error('Could not read that image.'));
                el.src = objectUrl;
            });
            const scale = Math.min(1, MAX_EDGE / Math.max(img.width, img.height, 1));
            const canvas = document.createElement('canvas');
            canvas.width = Math.max(1, Math.round(img.width * scale));
            canvas.height = Math.max(1, Math.round(img.height * scale));
            const ctx = canvas.getContext('2d');
            if (!ctx) throw new Error('Could not read that image.');
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            let mediaType = file.type === 'image/png' ? 'image/png' : 'image/jpeg';
            let blob = await encodeCanvas(canvas, mediaType, mediaType === 'image/jpeg' ? 0.85 : undefined);
            if (blob.size > 3_500_000) {
                mediaType = 'image/jpeg';
                blob = await encodeCanvas(canvas, mediaType, 0.7);
            }
            const dataUrl = await readBlob(blob);
            const comma = dataUrl.indexOf(',');
            return {
                media_type: mediaType,
                data_base64: comma >= 0 ? dataUrl.slice(comma + 1) : '',
                previewUrl: dataUrl,
            };
        } finally {
            URL.revokeObjectURL(objectUrl);
        }
    }

    function bind(opts) {
        const button = opts.button;
        const fileInput = opts.fileInput;
        const preview = opts.preview;
        const textarea = opts.textarea;
        let current = null;

        function paint() {
            preview.replaceChildren();
            if (!current) {
                preview.hidden = true;
                return;
            }
            preview.hidden = false;
            const img = document.createElement('img');
            img.src = current.previewUrl;
            img.alt = 'Image to send';
            const remove = document.createElement('button');
            remove.type = 'button';
            remove.className = 'chat-chrome-toggle';
            remove.textContent = 'Remove';
            remove.addEventListener('click', () => {
                current = null;
                paint();
            });
            preview.append(img, remove);
        }

        async function takeFile(file) {
            try {
                current = await fileToImage(file);
                paint();
            } catch (err) {
                if (typeof opts.onError === 'function') {
                    opts.onError(err instanceof Error ? err.message : 'Could not read that image.');
                }
            }
        }

        button.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', () => {
            const file = fileInput.files && fileInput.files[0];
            fileInput.value = '';
            if (file) takeFile(file);
        });
        if (textarea) {
            textarea.addEventListener('paste', (event) => {
                const items = event.clipboardData && event.clipboardData.items;
                if (!items) return;
                for (const item of items) {
                    if (item.kind === 'file' && TYPES.indexOf(item.type) >= 0) {
                        const file = item.getAsFile();
                        if (!file) continue;
                        event.preventDefault();
                        takeFile(file);
                        return;
                    }
                }
            });
        }

        return {
            payload() {
                if (!current) return null;
                return { media_type: current.media_type, data_base64: current.data_base64 };
            },
            has() {
                return Boolean(current);
            },
            clear() {
                current = null;
                paint();
            },
        };
    }

    function appendToBubble(bubble, url) {
        if (!url) return;
        const img = document.createElement('img');
        img.className = 'prompt-image';
        img.alt = 'Attached image';
        img.src = url;
        bubble.appendChild(img);
    }

    LightHouse.promptImage = { bind, appendToBubble };
})();
