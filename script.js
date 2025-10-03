// Knowledge Metabolism Tracker - Interactive Functionality

document.addEventListener('DOMContentLoaded', function() {
    initializeNavigation();
    initializeCharts();
    initializeInteractiveElements();
    initializeCaptureFeatures();
    initializeQuickAddForm();
});

// Navigation System
function initializeNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    const pages = document.querySelectorAll('.page');

    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();

            // Remove active class from all nav items and pages
            document.querySelectorAll('.nav-item').forEach(item => item.classList.remove('active'));
            pages.forEach(page => page.classList.remove('active'));

            // Add active class to clicked nav item
            this.parentElement.classList.add('active');

            // Show corresponding page
            const targetPage = this.getAttribute('data-page');
            const targetPageElement = document.getElementById(targetPage + '-page');
            if (targetPageElement) {
                targetPageElement.classList.add('active');
            }
        });
    });
}

// Chart Initialization (Placeholder for actual chart libraries)
function initializeCharts() {
    // Knowledge Consumption Chart
    const consumptionChart = document.getElementById('consumptionChart');
    if (consumptionChart) {
        // Placeholder for Chart.js or similar library
        consumptionChart.innerHTML = '<div style="display: flex; align-items: center; justify-content: center; height: 100%; color: #6b7280;">Knowledge Consumption Trends</div>';
    }

    // Gap Forecast Chart
    const gapChart = document.getElementById('gapChart');
    if (gapChart) {
        gapChart.innerHTML = '<div style="display: flex; align-items: center; justify-content: center; height: 100%; color: #6b7280;">Knowledge Gap Forecast</div>';
    }
}

// Interactive Elements
function initializeInteractiveElements() {
    // Animate progress bars
    animateProgressBars();

    // Add hover effects to cards
    addCardHoverEffects();

    // Initialize tooltips
    initializeTooltips();

    // Add click handlers for insight items
    addInsightClickHandlers();
}

function animateProgressBars() {
    const progressBars = document.querySelectorAll('.progress-fill');

    progressBars.forEach(bar => {
        const targetWidth = bar.style.width;
        bar.style.width = '0%';

        setTimeout(() => {
            bar.style.width = targetWidth;
        }, 500);
    });
}

function addCardHoverEffects() {
    const cards = document.querySelectorAll('.card');

    cards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-4px)';
        });

        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(-2px)';
        });
    });
}

function initializeTooltips() {
    const tooltipElements = document.querySelectorAll('[data-tooltip]');

    tooltipElements.forEach(element => {
        element.addEventListener('mouseenter', function() {
            showTooltip(this, this.getAttribute('data-tooltip'));
        });

        element.addEventListener('mouseleave', function() {
            hideTooltip();
        });
    });
}

function showTooltip(element, text) {
    const tooltip = document.createElement('div');
    tooltip.className = 'tooltip';
    tooltip.textContent = text;
    tooltip.style.cssText = `
        position: absolute;
        background: #1f1f1f;
        color: #e5e5e5;
        padding: 0.5rem 0.75rem;
        border-radius: 6px;
        font-size: 0.75rem;
        z-index: 1000;
        border: 1px solid #2a2a2a;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    `;

    document.body.appendChild(tooltip);

    const rect = element.getBoundingClientRect();
    tooltip.style.left = rect.left + (rect.width / 2) - (tooltip.offsetWidth / 2) + 'px';
    tooltip.style.top = rect.top - tooltip.offsetHeight - 8 + 'px';
}

function hideTooltip() {
    const tooltip = document.querySelector('.tooltip');
    if (tooltip) {
        tooltip.remove();
    }
}

function addInsightClickHandlers() {
    // Dormant knowledge items
    const dormantItems = document.querySelectorAll('.dormant-item');
    dormantItems.forEach(item => {
        item.addEventListener('click', function() {
            showInsightDetail('dormant', this);
        });
    });

    // Prediction items
    const predictionItems = document.querySelectorAll('.prediction-item');
    predictionItems.forEach(item => {
        item.addEventListener('click', function() {
            showInsightDetail('prediction', this);
        });
    });
}

function showInsightDetail(type, element) {
    // Create modal or expand detail view
    const modal = document.createElement('div');
    modal.className = 'insight-modal';
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.8);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 1000;
    `;

    const modalContent = document.createElement('div');
    modalContent.className = 'modal-content';
    modalContent.style.cssText = `
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 12px;
        padding: 2rem;
        max-width: 600px;
        width: 90%;
        max-height: 80vh;
        overflow-y: auto;
    `;

    modalContent.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
            <h3 style="color: #e5e5e5; margin: 0;">Insight Details</h3>
            <button class="close-modal" style="background: none; border: none; color: #9ca3af; font-size: 1.5rem; cursor: pointer;">&times;</button>
        </div>
        <div style="color: #d1d5db;">
            ${element.innerHTML}
        </div>
        <div style="margin-top: 1.5rem; display: flex; gap: 1rem;">
            <button class="btn btn-primary">Apply Insight</button>
            <button class="btn btn-secondary">Save for Later</button>
        </div>
    `;

    modal.appendChild(modalContent);
    document.body.appendChild(modal);

    // Close modal handlers
    modal.querySelector('.close-modal').addEventListener('click', () => modal.remove());
    modal.addEventListener('click', (e) => {
        if (e.target === modal) modal.remove();
    });
}

// Capture Features
function initializeCaptureFeatures() {
    initializeToolbar();
    initializeTagging();
    initializeFileUpload();
    initializeAutoSave();
}

function initializeToolbar() {
    const toolbarButtons = document.querySelectorAll('.toolbar-btn');

    toolbarButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Remove active class from all buttons
            toolbarButtons.forEach(btn => btn.classList.remove('active'));

            // Add active class to clicked button
            this.classList.add('active');

            // Handle different content types
            const contentType = this.getAttribute('data-type');
            handleContentType(contentType);
        });
    });
}

function handleContentType(type) {
    const editor = document.querySelector('.content-editor');

    switch(type) {
        case 'text':
            editor.placeholder = 'Start capturing your knowledge...';
            break;
        case 'image':
            editor.placeholder = 'Describe the image or add context...';
            break;
        case 'link':
            editor.placeholder = 'Add notes about this link...';
            break;
        case 'document':
            editor.placeholder = 'Summarize the document or add key insights...';
            break;
    }
}

function initializeTagging() {
    const tagInput = document.querySelector('.tag-input input');
    const tagSuggestions = document.querySelector('.tag-suggestions');

    if (tagInput) {
        tagInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                addTag(this.value.trim());
                this.value = '';
            }
        });
    }

    // Make suggestion tags clickable
    if (tagSuggestions) {
        tagSuggestions.addEventListener('click', function(e) {
            if (e.target.classList.contains('tag')) {
                addTag(e.target.textContent);
            }
        });
    }
}

function addTag(tagText) {
    if (!tagText) return;

    const tagContainer = document.querySelector('.tag-suggestions');
    const newTag = document.createElement('span');
    newTag.className = 'tag';
    newTag.textContent = tagText;
    newTag.style.cursor = 'pointer';

    // Add remove functionality
    newTag.addEventListener('click', function() {
        this.remove();
    });

    tagContainer.appendChild(newTag);
}

function initializeFileUpload() {
    const uploadArea = document.querySelector('.upload-area');

    if (uploadArea) {
        uploadArea.addEventListener('dragover', function(e) {
            e.preventDefault();
            this.style.borderColor = '#6366f1';
            this.style.backgroundColor = 'rgba(99, 102, 241, 0.1)';
        });

        uploadArea.addEventListener('dragleave', function(e) {
            e.preventDefault();
            this.style.borderColor = '#2a2a2a';
            this.style.backgroundColor = 'transparent';
        });

        uploadArea.addEventListener('drop', function(e) {
            e.preventDefault();
            this.style.borderColor = '#2a2a2a';
            this.style.backgroundColor = 'transparent';

            const files = e.dataTransfer.files;
            handleFileUpload(files);
        });

        uploadArea.addEventListener('click', function() {
            const fileInput = document.createElement('input');
            fileInput.type = 'file';
            fileInput.multiple = true;
            fileInput.accept = '.pdf,.doc,.docx,.txt,.md';

            fileInput.addEventListener('change', function() {
                handleFileUpload(this.files);
            });

            fileInput.click();
        });
    }
}

function handleFileUpload(files) {
    Array.from(files).forEach(file => {
        console.log('Uploading file:', file.name);
        // Implement actual file upload logic here
        showNotification(`File "${file.name}" uploaded successfully`, 'success');
    });
}

function initializeAutoSave() {
    const titleInput = document.querySelector('.title-input');
    const contentEditor = document.querySelector('.content-editor');

    let autoSaveTimeout;

    function autoSave() {
        clearTimeout(autoSaveTimeout);
        autoSaveTimeout = setTimeout(() => {
            const title = titleInput?.value || '';
            const content = contentEditor?.value || '';

            if (title || content) {
                // Implement actual save logic here
                showNotification('Draft saved automatically', 'info');
            }
        }, 2000);
    }

    if (titleInput) {
        titleInput.addEventListener('input', autoSave);
    }

    if (contentEditor) {
        contentEditor.addEventListener('input', autoSave);
    }
}

// Utility Functions
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.style.cssText = `
        position: fixed;
        top: 2rem;
        right: 2rem;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        color: white;
        font-weight: 500;
        z-index: 1000;
        animation: slideIn 0.3s ease;
    `;

    switch(type) {
        case 'success':
            notification.style.background = 'linear-gradient(135deg, #10b981, #059669)';
            break;
        case 'error':
            notification.style.background = 'linear-gradient(135deg, #ef4444, #dc2626)';
            break;
        case 'info':
        default:
            notification.style.background = 'linear-gradient(135deg, #6366f1, #8b5cf6)';
            break;
    }

    notification.textContent = message;
    document.body.appendChild(notification);

    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }

    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Simulate real-time data updates
function simulateDataUpdates() {
    setInterval(() => {
        // Update metrics with slight variations
        updateMetrics();

        // Occasionally show new insights
        if (Math.random() < 0.1) {
            showNotification('New insight available!', 'info');
        }
    }, 30000); // Every 30 seconds
}

function updateMetrics() {
    const metricValues = document.querySelectorAll('.metric-value');

    metricValues.forEach(metric => {
        const currentValue = parseInt(metric.textContent);
        if (!isNaN(currentValue)) {
            const variation = Math.floor(Math.random() * 3) - 1; // -1, 0, or 1
            const newValue = Math.max(0, currentValue + variation);
            metric.textContent = newValue;
        }
    });
}

// Initialize data simulation
setTimeout(simulateDataUpdates, 5000);


// Quick Add Item form -> posts to /api/knowledge with basic validation
function initializeQuickAddForm() {
    const form = document.getElementById('quick-add-form');
    if (!form) return;

    const titleEl = document.getElementById('qa-title');
    const contentEl = document.getElementById('qa-content');
    const typeEl = document.getElementById('qa-content-type');
    const tagsEl = document.getElementById('qa-tags');
    const chipsEl = document.getElementById('qa-tag-chips');
    const suggestWrap = document.getElementById('qa-tag-suggest');
    const suggestList = document.getElementById('qa-tag-suggest-list');
    const linkWrap = document.getElementById('qa-link-url-wrap');
    const linkUrlEl = document.getElementById('qa-link-url');
    const mdPreview = document.getElementById('qa-md-preview');
    const msgEl = document.getElementById('quick-form-msg');

    let tags = [];

    const ALLOWED = new Set(['text','link','markdown','pdf','video','audio','image','code']);

    function renderChips() {
        chipsEl.innerHTML = '';
        tags.forEach(t => chipsEl.appendChild(createChip(t, (text, chip) => {
            tags = tags.filter(x => x.toLowerCase() !== text.toLowerCase());
            chip.remove();
        })));
    }

    async function showSuggestions(q) {
        const list = await fetchTagSuggestions(q);
        if (!list.length) { suggestWrap.style.display = 'none'; return; }
        suggestList.innerHTML = '';
        list.forEach(item => {
            const name = item.name || item;
            const el = document.createElement('div');
            el.className = 'tag';
            el.style.cursor = 'pointer';
            el.textContent = name;
            el.addEventListener('click', () => {
                if (tags.length >= 20) return;
                if (!tags.some(t => t.toLowerCase() === name.toLowerCase())) {
                    tags.push(name);
                    renderChips();
                }
                suggestWrap.style.display = 'none';
                tagsEl.value = '';
            });
            suggestList.appendChild(el);
        });
        suggestWrap.style.display = 'block';
    }

    tagsEl.addEventListener('input', (e) => {
        const q = e.target.value.trim();
        showSuggestions(q);
    });
    tagsEl.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            const val = (tagsEl.value || '').trim();
            if (val && tags.length < 20 && !tags.some(t => t.toLowerCase() === val.toLowerCase())) {
                tags.push(val);
                renderChips();
            }
            tagsEl.value = '';
            suggestWrap.style.display = 'none';
        }
    });
    document.addEventListener('click', (e) => {
        if (!suggestWrap.contains(e.target) && e.target !== tagsEl) {
            suggestWrap.style.display = 'none';
        }
    });

    function onTypeChanged() {
        const ct = typeEl.value;
        linkWrap.style.display = ct === 'link' ? 'block' : 'none';
        mdPreview.style.display = ct === 'markdown' ? 'block' : 'none';
        if (ct === 'markdown') {
            const md = contentEl.value || '';
            if (window.marked) mdPreview.innerHTML = window.marked.parse(md);
        }
    }
    typeEl.addEventListener('change', onTypeChanged);
    contentEl.addEventListener('input', () => {
        if (typeEl.value === 'markdown' && window.marked) {
            mdPreview.innerHTML = window.marked.parse(contentEl.value || '');
        }
    });
    onTypeChanged();

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        msgEl.textContent = '';

        const title = (titleEl.value || '').trim();
        const content = contentEl.value || '';
        const content_type = typeEl.value;
        const source_url = (linkUrlEl.value || '').trim();

        // Client-side validations to match backend rules (backend still validates)
        if (!title) { msgEl.textContent = 'Title is required'; return; }
        if (!ALLOWED.has(content_type)) { msgEl.textContent = 'Invalid content type'; return; }
        if (tags.length > 20) { msgEl.textContent = 'Maximum 20 tags allowed'; return; }
        if (content_type === 'link') {
            if (!source_url || !isValidUrl(source_url)) { msgEl.textContent = 'Please provide a valid Link URL'; return; }
        }

        const payload = { title, content, content_type, tags };
        if (content_type === 'link') payload.source_url = source_url;

        try {
            const res = await fetch('/api/knowledge', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const body = await res.json().catch(() => ({}));
            if (!res.ok) throw new Error(body?.error || 'Request failed');
            showNotification('Item created', 'success');
            msgEl.textContent = `Created: ${body?.data?.id || 'OK'}`;
            form.reset();
            tags = [];
            renderChips();
            onTypeChanged();
        } catch (err) {
            showNotification(err.message || 'Failed to create', 'error');
            msgEl.textContent = 'Error: ' + err.message;
        }
    });

// Recent items with pagination and simple filters
function initializeRecentItems() {
    const listEl = document.getElementById('recent-items');
    const prevBtn = document.getElementById('recent-prev');
    const nextBtn = document.getElementById('recent-next');
    const pageEl = document.getElementById('recent-page');
    const typeFilter = document.getElementById('recent-type-filter');
    const tagFilter = document.getElementById('recent-tag-filter');
    const applyBtn = document.getElementById('recent-apply');
    if (!listEl) return;

    let page = 1;
    const per_page = 10;
    let total_pages = 1;

    async function load() {
        listEl.textContent = 'Loading...';
        const params = new URLSearchParams({ page: String(page), per_page: String(per_page) });
        const typeVal = (typeFilter.value || '').trim();
        const tagVal = (tagFilter.value || '').trim();
        if (typeVal) params.set('type', typeVal);
        if (tagVal) params.set('tag', tagVal);

        const res = await fetch('/api/knowledge?' + params.toString());
        const body = await res.json();
        const data = Array.isArray(body?.data) ? body.data : [];
        const pagination = body?.pagination || { page, per_page, total_pages: 1 };
        total_pages = pagination.total_pages || 1;

        if (!data.length) {
            listEl.textContent = 'No items.';
        } else {
            listEl.innerHTML = data.map(it => (
                `<div class=\"item\">
                    <div><strong>${(it.title||'').replace(/[&<>]/g, s => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[s]))}</strong></div>
                    <div>Type: ${it.content_type}</div>
                    <div>Created: ${it.created_at}</div>
                    <div>${(it.content||'').slice(0, 160)}</div>
                </div>`
            )).join('');
        }

        pageEl.textContent = `Page ${pagination.page || page} / ${total_pages}`;
        prevBtn.disabled = page <= 1;
        nextBtn.disabled = page >= total_pages;
    }

    prevBtn?.addEventListener('click', () => { if (page > 1) { page -= 1; load(); } });
    nextBtn?.addEventListener('click', () => { if (page < total_pages) { page += 1; load(); } });
    applyBtn?.addEventListener('click', () => { page = 1; load(); });

    load();
}

// Initialize Recent Items after DOM ready
document.addEventListener('DOMContentLoaded', initializeRecentItems);

}


/**
 * Knowledge Metabolism Tracker – Frontend Interaction Layer
 * - Navigation enhancements (active state sync, mobile toggle, header hide-on-scroll)
 * - Staggered reveal animations powered by IntersectionObserver
 * - Microinteractions for buttons, cards, and metric counters
 * - Parallax accents with reduced-motion fallbacks
 * - Lightweight utilities (throttle, clamp) for performance
 */

document.addEventListener('DOMContentLoaded', () => {
    const state = {
        prefersReducedMotion: window.matchMedia('(prefers-reduced-motion: reduce)').matches,
        headerHidden: false,
        lastScrollY: window.scrollY
    };

    initializeNavigation(state);
    initializeRevealAnimations(state);
    initializeParallaxLayers(state);
    initializeMicroInteractions(state);
    initializeMetricCounters(state);
});

/* -------------------------------------------------------------------------- */
/* Navigation & Header                                                         */
/* -------------------------------------------------------------------------- */
function initializeNavigation(state) {
    const header = document.querySelector('.site__header');
    const nav = document.querySelector('.nav');
    const navToggle = document.querySelector('.nav__toggle');
    const navLinks = document.querySelectorAll('.nav__link');
    const sections = document.querySelectorAll('[data-section-id]');

    if (!header) return;

    /* Mobile toggle */
    if (navToggle && nav) {
        navToggle.addEventListener('click', () => {
            const isExpanded = navToggle.getAttribute('aria-expanded') === 'true';
            navToggle.setAttribute('aria-expanded', String(!isExpanded));
            nav.dataset.open = String(!isExpanded);
            document.body.classList.toggle('nav-open', !isExpanded);
        });
    }

    /* Close nav after clicking a link */
    navLinks.forEach((link) => {
        link.addEventListener('click', () => {
            if (navToggle && nav.dataset.open === 'true') {
                navToggle.click();
            }
        });
    });

    /* Active section syncing */
    const observerOptions = {
        root: null,
        rootMargin: '-40% 0px -40% 0px',
        threshold: 0
    };

    const sectionObserver = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
            if (!entry.isIntersecting) return;
            const sectionId = entry.target.dataset.sectionId;
            if (!sectionId) return;

            navLinks.forEach((link) => {
                const target = link.getAttribute('href')?.replace('#', '');
                link.setAttribute('aria-current', target === sectionId ? 'page' : 'false');
            });
        });
    }, observerOptions);

    sections.forEach((section) => sectionObserver.observe(section));

    /* Header hide-on-scroll */
    const handleScroll = throttle(() => {
        const currentScroll = window.scrollY;
        const isScrollingDown = currentScroll > state.lastScrollY;
        const hideThreshold = 140;

        if (currentScroll > hideThreshold && isScrollingDown && !state.headerHidden) {
            header.setAttribute('aria-hidden', 'true');
            state.headerHidden = true;
        }

        if ((currentScroll < state.lastScrollY || currentScroll <= hideThreshold) && state.headerHidden) {
            header.removeAttribute('aria-hidden');
            state.headerHidden = false;
        }

        state.lastScrollY = currentScroll <= 0 ? 0 : currentScroll;
    }, 160);

    window.addEventListener('scroll', handleScroll, { passive: true });
}

/* -------------------------------------------------------------------------- */
/* Reveal Animations                                                           */
/* -------------------------------------------------------------------------- */
function initializeRevealAnimations(state) {
    const revealables = document.querySelectorAll('[data-animate]');

    if (!revealables.length) return;

    if (state.prefersReducedMotion) {
        revealables.forEach((el) => el.classList.add('is-visible'));
        return;
    }

    const observer = new IntersectionObserver((entries, obs) => {
        entries.forEach((entry) => {
            if (!entry.isIntersecting) return;

            const target = entry.target;
            const stagger = Number(target.dataset.animateDelay || 0);
            setTimeout(() => target.classList.add('is-visible'), stagger);
            obs.unobserve(target);
        });
    }, { threshold: 0.35, rootMargin: '0px 0px -10% 0px' });

    revealables.forEach((el) => observer.observe(el));
}

/* -------------------------------------------------------------------------- */
/* Parallax Accents                                                            */
/* -------------------------------------------------------------------------- */
function initializeParallaxLayers(state) {
    if (state.prefersReducedMotion) return;

    const parallaxItems = document.querySelectorAll('[data-parallax-depth]');
    if (!parallaxItems.length) return;

    let ticking = false;
    const parallaxState = { x: 0, y: 0 };

    const updateParallax = () => {
        parallaxItems.forEach((item) => {
            const depth = Number(item.dataset.parallaxDepth) || 0;
            const translateX = -parallaxState.x * depth;
            const translateY = -parallaxState.y * depth;
            item.style.transform = `translate3d(${translateX}px, ${translateY}px, 0)`;
        });
        ticking = false;
    };

    const handlePointer = (event) => {
        if (ticking) return;
        ticking = true;

        const { innerWidth, innerHeight } = window;
        parallaxState.x = (event.clientX / innerWidth - 0.5) * 16;
        parallaxState.y = (event.clientY / innerHeight - 0.5) * 14;

        window.requestAnimationFrame(updateParallax);
    };

    window.addEventListener('pointermove', handlePointer);
}

/* -------------------------------------------------------------------------- */
/* Microinteractions                                                           */
/* -------------------------------------------------------------------------- */
function initializeMicroInteractions(state) {
    const cards = document.querySelectorAll('.card');
    const buttons = document.querySelectorAll('.button');

    /* Card tilt */
    if (!state.prefersReducedMotion) {
        cards.forEach((card) => {
            card.addEventListener('pointermove', (event) => {
                const { left, top, width, height } = card.getBoundingClientRect();
                const x = (event.clientX - left) / width;
                const y = (event.clientY - top) / height;
                const tiltX = clamp((0.5 - y) * 10, -8, 8);
                const tiltY = clamp((x - 0.5) * 10, -8, 8);

                card.style.transform = `perspective(900px) rotateX(${tiltX}deg) rotateY(${tiltY}deg) translateY(-6px)`;
            });

            card.addEventListener('pointerleave', () => {
                card.style.transform = '';
            });
        });
    }

    /* Button hover pulse */
    buttons.forEach((button) => {
        button.addEventListener('pointerenter', () => {
            button.classList.add('is-hovered');
        });
        button.addEventListener('pointerleave', () => {
            button.classList.remove('is-hovered');
        });
    });
}

/* -------------------------------------------------------------------------- */
/* Metric Counters                                                             */
/* -------------------------------------------------------------------------- */
function initializeMetricCounters(state) {
    const counters = document.querySelectorAll('[data-count-to]');
    if (!counters.length || state.prefersReducedMotion) return;

    const observer = new IntersectionObserver((entries, obs) => {
        entries.forEach((entry) => {
            if (!entry.isIntersecting) return;

            const el = entry.target;
            const target = Number(el.dataset.countTo);
            const duration = Number(el.dataset.countDuration || 1800);

            animateCounter(el, target, duration);
            obs.unobserve(el);
        });
    }, { threshold: 0.6 });

    counters.forEach((counter) => observer.observe(counter));
}

function animateCounter(element, endValue, duration) {
    const startValue = Number(element.textContent.replace(/[^0-9.]/g, '')) || 0;
    const startTime = performance.now();
    const formatter = new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 });

    const update = (currentTime) => {
        const elapsed = currentTime - startTime;
        const progress = clamp(elapsed / duration, 0, 1);
        const easedProgress = easeOutCubic(progress);
        const currentValue = startValue + (endValue - startValue) * easedProgress;

        element.textContent = formatter.format(Math.round(currentValue));

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    };

    requestAnimationFrame(update);
}

/* -------------------------------------------------------------------------- */
/* Helpers                                                                     */
/* -------------------------------------------------------------------------- */
function throttle(callback, delay) {
    let lastCall = 0;
    let timeoutId;

    return (...args) => {
        const now = Date.now();
        const remaining = delay - (now - lastCall);

        if (remaining <= 0) {
            clearTimeout(timeoutId);
            lastCall = now;
            callback.apply(this, args);
        } else if (!timeoutId) {
            timeoutId = setTimeout(() => {
                lastCall = Date.now();
                timeoutId = null;
                callback.apply(this, args);
            }, remaining);
        }
    };
}

function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
}

function easeOutCubic(t) {
    return 1 - Math.pow(1 - t, 3);
}

// Utilities for tag chips and autocomplete
function createChip(text, onRemove) {
    const chip = document.createElement('span');
    chip.className = 'tag';
    chip.textContent = text;
    chip.style.cursor = 'pointer';
    chip.addEventListener('click', () => onRemove(text, chip));
    return chip;
}

async function fetchTagSuggestions(q) {
    try {
        const res = await fetch('/api/tags');
        const body = await res.json();
        const all = Array.isArray(body?.data) ? body.data : [];
        if (!q) return all.slice(0, 10);
        const lower = q.toLowerCase();
        return all.filter(t => (t.name || t).toLowerCase().includes(lower)).slice(0, 10);
    } catch { return []; }
}

function isValidUrl(u) {
    try { new URL(u); return true; } catch { return false; }
}

