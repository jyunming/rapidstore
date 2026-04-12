document.addEventListener('DOMContentLoaded', () => {

    // ── Copy: pip install cmd ─────────────────────────────────────────────────
    window.copyInstallCmd = function() {
        const cmd = 'pip install tqdb';
        navigator.clipboard.writeText(cmd).then(() => {
            const btn = document.querySelector('.copy-btn[onclick="copyInstallCmd()"]');
            const orig = btn.innerHTML;
            btn.innerHTML = '<span style="font-size:14px;font-weight:bold;color:var(--accent-primary)">Copied!</span>';
            setTimeout(() => { btn.innerHTML = orig; }, 2000);
        });
    };

    // ── Copy: code block ──────────────────────────────────────────────────────
    const copyCodeBtn = document.querySelector('.copy-code');
    if (copyCodeBtn) {
        copyCodeBtn.addEventListener('click', () => {
            const target = document.getElementById(copyCodeBtn.dataset.target);
            navigator.clipboard.writeText(target.innerText).then(() => {
                const orig = copyCodeBtn.innerText;
                copyCodeBtn.innerText = 'Copied!';
                setTimeout(() => { copyCodeBtn.innerText = orig; }, 2000);
            });
        });
    }

    // ── Scroll-triggered entrance animations ─────────────────────────────────
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.15, rootMargin: '0px 0px -60px 0px' });

    document.querySelectorAll('.fade-in-up, .fade-in-left, .fade-in-right')
        .forEach(el => observer.observe(el));

    // ── Version badge — GitHub releases API ──────────────────────────────────
    const badge = document.getElementById('version-badge');
    if (badge) {
        fetch('https://api.github.com/repos/jyunming/TurboQuantDB/releases/latest', {
            headers: { Accept: 'application/vnd.github+json' }
        })
        .then(r => r.ok ? r.json() : Promise.reject())
        .then(data => { badge.textContent = data.tag_name + ' Released'; })
        .catch(() => { badge.textContent = 'Latest Release'; });
    }

});
