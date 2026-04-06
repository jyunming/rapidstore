document.addEventListener('DOMContentLoaded', () => {
    // --- Global Helpers ---
    window.copyInstallCmd = function() {
        const cmd = 'pip install tqdb';
        navigator.clipboard.writeText(cmd).then(() => {
            const btn = document.querySelector('.copy-btn[onclick="copyInstallCmd()"]');
            const originalHTML = btn.innerHTML;
            btn.innerHTML = '<span style="font-size: 14px; font-weight: bold; color: var(--accent-primary);">Copied!</span>';
            setTimeout(() => { btn.innerHTML = originalHTML; }, 2000);
        });
    };

    const copyCodeBtn = document.querySelector('.copy-code');
    if (copyCodeBtn) {
        copyCodeBtn.addEventListener('click', () => {
            const codeTarget = document.getElementById(copyCodeBtn.getAttribute('data-target'));
            navigator.clipboard.writeText(codeTarget.innerText).then(() => {
                const originalText = copyCodeBtn.innerText;
                copyCodeBtn.innerText = 'Copied!';
                setTimeout(() => { copyCodeBtn.innerText = originalText; }, 2000);
            });
        });
    }

    // Intersection Observer for scroll animations
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1 });

    document.querySelectorAll('.fade-in-up').forEach(el => observer.observe(el));

    // --- Explainer Engine ---
    let currentStep = 0;
    const steps = [
        {
            title: "The \"Turbo\" Trick: Random Rotation",
            desc: "Real-world data is often highly skewed. TurboQuant applies a random rotation matrix to transform the distribution into a predictable Gaussian (Normal) shape, making coordinates nearly independent.",
            setup: setupRotationStep
        },
        {
            title: "Optimal 1D Quantization",
            desc: "Once rotated, each coordinate follows a Normal distribution. We can then place quantization 'buckets' (centroids) optimally using the Lloyd-Max algorithm to minimize error.",
            setup: setupQuantizationStep
        },
        {
            title: "The \"Prod\" Trick: Unbiased Inner Products",
            desc: "Quantization introduced a bias. TurboQuant corrects this by using 1 bit of our budget to sketch the quantization residual using a QJL transform, preserving the expected inner product.",
            setup: setupResidualStep
        }
    ];

    const stepTitle = document.getElementById('step-title');
    const stepDesc = document.getElementById('step-desc');
    const stepVisual = document.getElementById('step-visual-container');
    const stepControls = document.getElementById('step-controls');
    const nextBtn = document.getElementById('next-step');
    const prevBtn = document.getElementById('prev-step');
    const navItems = document.querySelectorAll('.step-item');

    function updateExplainer() {
        // Update Nav
        navItems.forEach((item, i) => {
            item.classList.toggle('active', i === currentStep);
            item.classList.toggle('past', i < currentStep);
        });

        // Update Content
        const step = steps[currentStep];
        stepTitle.innerText = step.title;
        stepDesc.innerText = step.desc;
        
        // Clear and Setup Visuals
        stepVisual.innerHTML = '';
        stepControls.innerHTML = '';
        step.setup(stepVisual, stepControls);

        // Update Buttons
        prevBtn.disabled = currentStep === 0;
        nextBtn.innerText = currentStep === steps.length - 1 ? "Finish" : "Next Step →";
    }

    nextBtn.addEventListener('click', () => {
        if (currentStep < steps.length - 1) {
            currentStep++;
            updateExplainer();
        } else {
            // Scroll to ecosystem or top
            document.getElementById('ecosystem').scrollIntoView({ behavior: 'smooth' });
        }
    });

    prevBtn.addEventListener('click', () => {
        if (currentStep > 0) {
            currentStep--;
            updateExplainer();
        }
    });

    navItems.forEach((item, i) => {
        item.addEventListener('click', () => {
            currentStep = i;
            updateExplainer();
        });
    });

    // --- Step 1: Rotation Visualization ---
    function setupRotationStep(container, controls) {
        const scatter = document.createElement('div');
        scatter.className = 'scatter-container';
        container.appendChild(scatter);

        const applyBtn = document.createElement('button');
        applyBtn.className = 'btn btn-primary';
        applyBtn.innerText = 'Apply Random Rotation';
        controls.appendChild(applyBtn);

        const points = [];
        const count = 100;
        for (let i = 0; i < count; i++) {
            const t = (Math.random() - 0.5) * 1.5;
            const noiseX = (Math.random() - 0.5) * 0.2;
            const noiseY = (Math.random() - 0.5) * 0.2;
            
            const p = document.createElement('div');
            p.className = 'scatter-point';
            p.style.backgroundColor = '#60a5fa'; // Blue 400
            
            const data = {
                origX: (t + noiseX + 1) * 50,
                origY: (t + noiseY + 1) * 50,
                targetX: ((Math.random() + Math.random() + Math.random() - 1.5) * 0.6 + 1) * 50,
                targetY: ((Math.random() + Math.random() + Math.random() - 1.5) * 0.6 + 1) * 50,
                el: p
            };
            
            p.style.left = `${data.origX}%`;
            p.style.bottom = `${data.origY}%`;
            scatter.appendChild(p);
            points.push(data);
        }

        let rotated = false;
        applyBtn.addEventListener('click', () => {
            rotated = !rotated;
            applyBtn.innerText = rotated ? 'Reset' : 'Apply Random Rotation';
            points.forEach(p => {
                p.el.style.left = `${rotated ? p.targetX : p.origX}%`;
                p.el.style.bottom = `${rotated ? p.targetY : p.origY}%`;
                p.el.style.backgroundColor = rotated ? '#34d399' : '#60a5fa';
            });
        });
    }

    // --- Step 2: Quantization Visualization ---
    function setupQuantizationStep(container, controls) {
        const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        svg.setAttribute("viewBox", "0 0 100 100");
        svg.className = "gaussian-curve-container";
        svg.innerHTML = `
            <path d="M 0 100 Q 25 100 40 50 T 50 10 T 60 50 T 100 100" fill="rgba(59, 130, 246, 0.1)" stroke="rgba(59, 130, 246, 0.5)" stroke-width="1.5" />
            <line x1="0" y1="95" x2="100" y2="95" stroke="#334155" stroke-width="0.5" />
        `;
        container.appendChild(svg);

        const centroids = {
            1: [-0.798, 0.798],
            2: [-1.51, -0.453, 0.453, 1.51],
            4: [-2.5, -1.9, -1.5, -1.1, -0.8, -0.5, -0.2, -0.05, 0.05, 0.2, 0.5, 0.8, 1.1, 1.5, 1.9, 2.5]
        };

        const btnGroup = document.createElement('div');
        btnGroup.style.display = 'flex';
        btnGroup.style.gap = '10px';
        [1, 2, 4].forEach(b => {
            const bBtn = document.createElement('button');
            bBtn.className = 'btn btn-outline';
            bBtn.innerText = `${b}-bit`;
            bBtn.addEventListener('click', () => renderQuant(b));
            btnGroup.appendChild(bBtn);
        });
        controls.appendChild(btnGroup);

        function renderQuant(bits) {
            // Remove old centroids
            container.querySelectorAll('.centroid-marker').forEach(m => m.remove());
            container.querySelectorAll('.data-point').forEach(p => p.remove());

            const activeCentroids = centroids[bits];
            activeCentroids.forEach(c => {
                const marker = document.createElement('div');
                marker.className = 'centroid-marker';
                marker.innerHTML = '<div class="centroid-line"></div><div class="centroid-dot"></div>';
                marker.style.left = `${(c + 3) / 6 * 100}%`;
                container.appendChild(marker);
            });

            // Add animated data points
            [-1.2, 0.3, 1.8, -2.1, 0.9].forEach((val, i) => {
                const nearest = activeCentroids.reduce((prev, curr) => 
                    Math.abs(curr - val) < Math.abs(prev - val) ? curr : prev
                );

                const p = document.createElement('div');
                p.className = 'data-point';
                p.style.bottom = '80px';
                p.style.left = `${(val + 3) / 6 * 100}%`;
                container.appendChild(p);

                setTimeout(() => {
                    p.style.bottom = '46px';
                    p.style.left = `${(nearest + 3) / 6 * 100}%`;
                }, 500 + i * 100);
            });
        }

        renderQuant(1);
    }

    // --- Step 3: Residual Step ---
    function setupResidualStep(container, controls) {
        const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        svg.setAttribute("viewBox", "-120 -120 240 240");
        svg.className = "residual-vector-container";
        svg.style.width = "100%";
        svg.style.height = "100%";
        
        svg.innerHTML = `
            <g stroke="rgba(255,255,255,0.05)" stroke-width="1">
                <line x1="-120" y1="0" x2="120" y2="0" />
                <line x1="0" y1="-120" x2="0" y2="120" />
            </g>
            <!-- Original x -->
            <line x1="0" y1="0" x2="80" y2="-90" stroke="rgba(255,255,255,0.3)" stroke-width="2" />
            <circle cx="80" cy="-90" r="3" fill="rgba(255,255,255,0.5)" />
            <text x="80" y="-100" fill="white" font-size="8" opacity="0.6" text-anchor="middle">original x</text>

            <!-- MSE Quantized -->
            <line x1="0" y1="0" x2="60" y2="-50" stroke="#3b82f6" stroke-width="3" />
            <circle cx="60" cy="-50" r="4" fill="#3b82f6" />
            <text x="65" y="-40" fill="#3b82f6" font-size="9" font-weight="bold">x_mse</text>

            <!-- Residual r -->
            <line id="residual-line" x1="60" y1="-50" x2="80" y2="-90" stroke="#ef4444" stroke-width="2" stroke-dasharray="4" />
        `;
        container.appendChild(svg);

        const applyBtn = document.createElement('button');
        applyBtn.className = 'btn btn-primary';
        applyBtn.innerText = 'Apply QJL Correction';
        controls.appendChild(applyBtn);

        let active = false;
        applyBtn.addEventListener('click', () => {
            active = !active;
            applyBtn.innerText = active ? 'Reset' : 'Apply QJL Correction';
            
            if (active) {
                // Show QJL(r) addition
                const qjlLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
                qjlLine.setAttribute("x1", "60");
                qjlLine.setAttribute("y1", "-50");
                qjlLine.setAttribute("x2", "85");
                qjlLine.setAttribute("y2", "-85");
                qjlLine.setAttribute("stroke", "#10b981");
                qjlLine.setAttribute("stroke-width", "3");
                qjlLine.setAttribute("id", "qjl-line");
                svg.appendChild(qjlLine);

                const approxLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
                approxLine.setAttribute("x1", "0");
                approxLine.setAttribute("y1", "0");
                approxLine.setAttribute("x2", "85");
                approxLine.setAttribute("y2", "-85");
                approxLine.setAttribute("stroke", "#10b981");
                approxLine.setAttribute("stroke-width", "2");
                approxLine.setAttribute("stroke-dasharray", "4");
                approxLine.setAttribute("id", "approx-line");
                svg.appendChild(approxLine);
            } else {
                svg.getElementById('qjl-line')?.remove();
                svg.getElementById('approx-line')?.remove();
            }
        });
    }

    // Initialize
    updateExplainer();
});
