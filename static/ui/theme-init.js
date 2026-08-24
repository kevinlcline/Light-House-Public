/** Apply saved theme before first paint to avoid flash. */
(function () {
    'use strict';
    window.LightHouse = window.LightHouse || {};
    var t = localStorage.getItem('light_house_theme');
    if (t === 'light' || t === 'dark') {
        document.documentElement.setAttribute('data-theme', t);
    }
})();
