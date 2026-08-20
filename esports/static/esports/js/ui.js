'use strict';
/* Shared UI motion layer: scroll-reveal, route transitions, navbar shadow.
   Everything is gated behind a .js flag and prefers-reduced-motion so the
   site stays fully usable with no JS and for users who ask for less motion. */
(function () {
  var root = document.documentElement;
  root.classList.add('js');
  var reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function onReady(fn) {
    if (document.readyState !== 'loading') fn();
    else document.addEventListener('DOMContentLoaded', fn);
  }

  onReady(function () {
    document.body.classList.add('page-loaded');

    // ── Scroll reveal ────────────────────────────────────────────
    var els = Array.prototype.slice.call(document.querySelectorAll('.reveal'));
    if (reduce || !('IntersectionObserver' in window)) {
      els.forEach(function (el) { el.classList.add('in'); });
    } else {
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); }
        });
      }, { threshold: 0.12, rootMargin: '0px 0px -8% 0px' });
      els.forEach(function (el) { io.observe(el); });
    }

    // ── Navbar shadow on scroll ──────────────────────────────────
    var nav = document.querySelector('.navbar');
    if (nav) {
      var ticking = false;
      var onScroll = function () {
        if (ticking) return;
        ticking = true;
        window.requestAnimationFrame(function () {
          nav.classList.toggle('scrolled', window.scrollY > 8);
          ticking = false;
        });
      };
      window.addEventListener('scroll', onScroll, { passive: true });
      onScroll();
    }

    // ── Lightweight route transition on internal link clicks ─────
    if (!reduce) {
      document.addEventListener('click', function (ev) {
        var a = ev.target.closest && ev.target.closest('a');
        if (!a) return;
        var href = a.getAttribute('href');
        if (!href || href.charAt(0) === '#') return;
        if (a.target && a.target !== '_self') return;
        if (a.hasAttribute('download')) return;
        if (ev.defaultPrevented || ev.metaKey || ev.ctrlKey || ev.shiftKey || ev.altKey || ev.button !== 0) return;
        var url;
        try { url = new URL(a.href, location.href); } catch (e) { return; }
        if (url.origin !== location.origin) return;                         // external link
        if (url.pathname === location.pathname && url.search === location.search) return; // same page
        ev.preventDefault();
        document.body.classList.add('page-leaving');
        setTimeout(function () { window.location.href = a.href; }, 170);
      });
      // Restore if the page is shown again from the back/forward cache.
      window.addEventListener('pageshow', function () {
        document.body.classList.remove('page-leaving');
      });
    }
  });
})();
