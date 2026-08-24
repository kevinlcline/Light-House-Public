(function (global) {
    'use strict';

    const LH = (global.LightHouse = global.LightHouse || {});

    async function restartServer() {
        if (!window.confirm('Restart Light-House now? The site will be unavailable briefly.')) {
            return false;
        }
        try {
            const res = await fetch('/v1/admin/restart', { method: 'POST' });
            if (res.status === 404) {
                window.alert('Restart is disabled (ENV_EDITOR_ENABLED=false).');
                return false;
            }
            if (!res.ok) {
                window.alert('Restart request failed (' + res.status + ').');
                return false;
            }
            const data = await res.json();
            window.alert(data.message || 'Restart requested.');
            return true;
        } catch (err) {
            window.alert('Restart request failed: ' + (err.message || 'unknown error'));
            return false;
        }
    }

    function bindRestartButton(selector) {
        const button = document.querySelector(selector);
        if (!button) return;
        button.addEventListener('click', () => {
            restartServer();
        });
    }

    LH.admin = {
        restartServer,
        bindRestartButton,
    };
})(typeof window !== 'undefined' ? window : globalThis);
