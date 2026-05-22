const API_BASE = `${window.location.protocol}//${window.location.host}`;
const WS_PROTOCOL = window.location.protocol === 'https:' ? 'wss:' : 'ws:';

let sessionId = null;
let ws = null;
let metrics = { turns: 0, latencies: [] };

// DOM Cache
const dom = {
    messages: document.getElementById('messages-list'),
    input: document.getElementById('chat-input'),
    sendBtn: document.getElementById('send-btn'),
    ragToggle: document.getElementById('rag-toggle'),
    typingIndicator: document.getElementById('typing-indicator'),
    hero: document.getElementById('hero'),
    status: document.getElementById('ws-status'),
    statusIndicator: document.querySelector('.status-indicator'),
    sidebar: document.getElementById('sidebar'),
    openSidebar: document.getElementById('open-sidebar'),
    closeSidebar: document.getElementById('close-sidebar'),
    clearBtn: document.getElementById('clear-memory-btn'),
    stats: {
        latency: document.getElementById('stat-latency'),
        turns: document.getElementById('stat-turns'),
        intent: document.getElementById('stat-intent'),
        confidenceBar: document.getElementById('intent-confidence-bar')
    },
    entities: document.getElementById('entity-list')
};

// Sidebar Logic
dom.openSidebar.addEventListener('click', () => dom.sidebar.classList.add('open'));
dom.closeSidebar.addEventListener('click', () => dom.sidebar.classList.remove('open'));

// Auto-resize input
dom.input.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
    dom.sendBtn.disabled = !this.value.trim();
});

// Enter key to send
dom.input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Clear Memory
dom.clearBtn.addEventListener('click', () => {
    dom.messages.innerHTML = '';
    dom.hero.style.display = 'block';
    metrics = { turns: 0, latencies: [] };
    updateTelemetry({ intent: 'Initializing', sentiment_score: 0 });
    dom.stats.turns.textContent = '0';
    dom.stats.latency.textContent = '0';
    dom.entities.innerHTML = '<div class="empty-state">No entities parsed</div>';
    
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'clear_history', payload: {} }));
    }
    sessionId = null; // Next message gets a new session
});

// Init
document.addEventListener('DOMContentLoaded', initWebSocket);

function initWebSocket() {
    ws = new WebSocket(`${WS_PROTOCOL}//${window.location.host}/ws/chat`);

    ws.onopen = () => {
        dom.status.textContent = 'System Online';
        dom.statusIndicator.classList.remove('error');
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleMessage(data);
    };

    ws.onclose = () => {
        dom.status.textContent = 'Connection Lost';
        dom.statusIndicator.classList.add('error');
        setTimeout(initWebSocket, 3000); // Auto-reconnect
    };

    ws.onerror = () => {
        dom.statusIndicator.classList.add('error');
    };
}

function handleMessage(data) {
    switch (data.type) {
        case 'connected':
            sessionId = data.payload.session_id;
            break;
        case 'typing':
            if (data.payload.is_typing) {
                dom.typingIndicator.classList.add('active');
                scrollToBottom();
            } else {
                dom.typingIndicator.classList.remove('active');
            }
            break;
        case 'response':
            dom.typingIndicator.classList.remove('active');
            appendMessage('ai', data.payload);
            updateTelemetry(data.payload);
            break;
        case 'error':
            dom.typingIndicator.classList.remove('active');
            appendMessage('ai', { text: `System Error: ${data.payload.message}`, intent: 'error' });
            break;
    }
}

async function sendMessage() {
    const text = dom.input.value.trim();
    if (!text) return;

    // Reset UI
    dom.input.value = '';
    dom.input.style.height = 'auto';
    dom.sendBtn.disabled = true;
    if (dom.hero) dom.hero.style.display = 'none';

    // Append user message
    appendMessage('user', { text });

    const payload = { 
        text, 
        use_rag: dom.ragToggle.checked,
        session_id: sessionId
    };

    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'message', payload }));
    } else {
        await fallbackREST(payload);
    }
}

async function fallbackREST(payload) {
    dom.typingIndicator.classList.add('active');
    scrollToBottom();
    try {
        if (payload.use_rag) payload.top_k = 3;
        
        const res = await fetch(`${API_BASE}/api/v1/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        const data = await res.json();
        
        if (data.answer && !data.text) data.text = data.answer;
        if (data.latency && !data.processing_time_ms) data.processing_time_ms = data.latency.total_ms;
        
        sessionId = data.session_id;
        dom.typingIndicator.classList.remove('active');
        appendMessage('ai', data);
        updateTelemetry(data);
    } catch (err) {
        dom.typingIndicator.classList.remove('active');
        appendMessage('ai', { text: 'Network Error: Failed to reach Nexus core.', intent: 'error' });
    }
}

// UI Rendering
function appendMessage(role, data) {
    const row = document.createElement('div');
    row.className = `msg-row ${role}`;
    
    const icon = role === 'user' ? 'U' : '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>';
    
    // Parse text (fallback for API v2/v3 diffs)
    let text = data.text || data.answer || '';
    let htmlContent = role === 'ai' ? formatMarkdown(text) : escapeHtml(text);

    let html = `
        <div class="msg-avatar">${icon}</div>
        <div class="msg-bubble glass-panel">
            <div class="msg-content ${role === 'ai' ? 'streaming-text' : ''}">${htmlContent}</div>
    `;

    // Render citations and reasoning
    if (role === 'ai' && data.citations && data.citations.length > 0) {
        html += `<div class="reasoning-panel mt-2">
            <div class="reasoning-header">
                <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>
                Grounding Sources
            </div>`;
        data.citations.forEach(c => {
            html += `
                <div class="citation-box">
                    <div class="citation-src">${escapeHtml(c.title || c.source)} — ${(c.score*100).toFixed(1)}%</div>
                    <div class="citation-txt">"${escapeHtml(c.snippet)}"</div>
                </div>`;
        });
        html += `</div>`;
    }

    html += `</div>`;
    row.innerHTML = html;
    dom.messages.appendChild(row);
    scrollToBottom();

    // Stream effect removal
    if (role === 'ai') {
        setTimeout(() => {
            const contentDiv = row.querySelector('.msg-content');
            if (contentDiv) contentDiv.classList.remove('streaming-text');
        }, text.length * 10 + 500); // Rough estimate of streaming duration
    }
}

function updateTelemetry(data) {
    metrics.turns++;
    const latency = data.processing_time_ms || data.latency?.total_ms || 0;
    if (latency > 0) metrics.latencies.push(latency);

    dom.stats.turns.textContent = metrics.turns;
    
    if (metrics.latencies.length > 0) {
        const avg = metrics.latencies.reduce((a, b) => a + b, 0) / metrics.latencies.length;
        dom.stats.latency.textContent = Math.round(avg);
    }

    if (data.intent) {
        dom.stats.intent.textContent = data.intent;
        // Map confidence/sentiment to progress bar
        let conf = data.sentiment_score !== undefined ? (data.sentiment_score * 100) : 100;
        if (conf < 0) conf = Math.abs(conf) * 100; // Normalizing just for visual
        dom.stats.confidenceBar.style.width = `${Math.min(100, Math.max(10, conf))}%`;
        dom.stats.confidenceBar.style.backgroundColor = data.sentiment === 'negative' ? 'var(--error)' : 'var(--primary)';
    }

    if (data.entities && data.entities.length > 0) {
        dom.entities.innerHTML = '';
        data.entities.forEach(e => {
            dom.entities.innerHTML += `
                <div class="entity-tag">
                    <span>${escapeHtml(e.value)}</span>
                    <span class="entity-type">${escapeHtml(e.type)}</span>
                </div>
            `;
        });
    }
}

function scrollToBottom() {
    const viewport = document.getElementById('chat-viewport');
    viewport.scrollTop = viewport.scrollHeight;
}

// Basic Markdown Parser
function formatMarkdown(text) {
    if (!text) return '';
    
    let html = escapeHtml(text);
    
    // Code blocks
    html = html.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    // Bold
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    // Italic
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    // Line breaks
    html = html.replace(/\n/g, '<br>');
    
    return html;
}

function escapeHtml(unsafe) {
    if (!unsafe) return '';
    return unsafe
         .toString()
         .replace(/&/g, "&amp;")
         .replace(/</g, "&lt;")
         .replace(/>/g, "&gt;")
         .replace(/"/g, "&quot;")
         .replace(/'/g, "&#039;");
}
