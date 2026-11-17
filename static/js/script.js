/* =========================================================
   CROPWISE - SHARED INTERACTIONS
   ========================================================= */

document.addEventListener("DOMContentLoaded", () => {
    "use strict";

    const menuToggle = document.getElementById("menuToggle");
    const sidebar = document.getElementById("sidebar");
    const overlay = document.getElementById("mobileOverlay");
    const yearNode = document.getElementById("currentYear");

    if (yearNode) {
        yearNode.textContent = String(new Date().getFullYear());
    }

    const closeMenu = () => {
        if (!sidebar || !overlay || !menuToggle) return;
        sidebar.classList.remove("is-open");
        overlay.classList.remove("is-visible");
        menuToggle.setAttribute("aria-expanded", "false");
        document.body.classList.remove("navigation-open");
    };

    if (menuToggle && sidebar && overlay) {
        menuToggle.addEventListener("click", () => {
            const willOpen = !sidebar.classList.contains("is-open");
            sidebar.classList.toggle("is-open", willOpen);
            overlay.classList.toggle("is-visible", willOpen);
            menuToggle.setAttribute("aria-expanded", String(willOpen));
            document.body.classList.toggle("navigation-open", willOpen);
        });

        overlay.addEventListener("click", closeMenu);

        sidebar.querySelectorAll("a").forEach((link) => {
            link.addEventListener("click", closeMenu);
        });

        document.addEventListener("keydown", (event) => {
            if (event.key === "Escape") closeMenu();
        });

        window.addEventListener("resize", () => {
            if (window.innerWidth > 820) closeMenu();
        });
    }

    // Reveal content gently as it enters the viewport.
    const revealElements = document.querySelectorAll(".reveal");
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    if (reduceMotion || !("IntersectionObserver" in window)) {
        revealElements.forEach((element) => element.classList.add("is-visible"));
    } else {
        const revealObserver = new IntersectionObserver((entries, observer) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    entry.target.classList.add("is-visible");
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.08 });

        revealElements.forEach((element) => revealObserver.observe(element));
    }

    // Animate numeric overview figures once when visible.
    const counters = document.querySelectorAll("[data-count]");

    const animateCounter = (element) => {
        const target = Number(element.dataset.count);
        if (!Number.isFinite(target)) return;

        const decimalPlaces = Number(element.dataset.decimals || 0);
        const duration = 950;
        const start = performance.now();

        const update = (now) => {
            const progress = Math.min((now - start) / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);
            const current = target * eased;

            element.textContent = current.toLocaleString("en-IN", {
                minimumFractionDigits: decimalPlaces,
                maximumFractionDigits: decimalPlaces
            });

            if (progress < 1) requestAnimationFrame(update);
        };

        requestAnimationFrame(update);
    };

    if (reduceMotion || !("IntersectionObserver" in window)) {
        counters.forEach(animateCounter);
    } else {
        const counterObserver = new IntersectionObserver((entries, observer) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    animateCounter(entry.target);
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.45 });

        counters.forEach((counter) => counterObserver.observe(counter));
    }
});
