/* ==========================================================================
   VerifAI Modern Frontend Application Logic (ES6+)
   Seamless Async Verification, Live Fact Checking & Visual Analytics
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const titleInput = document.getElementById('titleInput');
  const contextInput = document.getElementById('contextInput');
  const btnVerify = document.getElementById('btnVerify');
  const btnClear = document.getElementById('btnClear');
  const btnSpinner = document.getElementById('btnSpinner');
  const btnText = document.getElementById('btnText');
  const accordionToggle = document.getElementById('accordionToggle');
  const accordionBody = document.getElementById('accordionBody');
  const charCounter = document.getElementById('charCounter');
  const factCheckSwitch = document.getElementById('factCheckSwitch');
  const modePills = document.querySelectorAll('.mode-pill');
  const exampleChips = document.querySelectorAll('.chip-btn');
  
  // Results Elements
  const resultsSection = document.getElementById('resultsSection');
  const verdictCard = document.getElementById('verdictCard');
  const verdictBadge = document.getElementById('verdictBadge');
  const verdictConfText = document.getElementById('verdictConfText');
  const verdictDesc = document.getElementById('verdictDesc');
  const realProgressFill = document.getElementById('realProgressFill');
  const fakeProgressFill = document.getElementById('fakeProgressFill');
  const realMetricValue = document.getElementById('realMetricValue');
  const fakeMetricValue = document.getElementById('fakeMetricValue');
  
  // Tab elements
  const tabButtons = document.querySelectorAll('.tab-btn');
  const tabPanes = document.querySelectorAll('.tab-pane');
  
  // Evidence & Detail Containers
  const factEvidenceContainer = document.getElementById('factEvidenceContainer');
  const recognizedTokensList = document.getElementById('recognizedTokensList');
  const oovTokensList = document.getElementById('oovTokensList');
  const rawStatsContainer = document.getElementById('rawStatsContainer');
  const limeContainer = document.getElementById('limeContainer');
  const btnFetchLime = document.getElementById('btnFetchLime');
  const limeSpinner = document.getElementById('limeSpinner');

  let currentMode = 'headline';
  let lastVerifiedText = '';

  // Initialize Character Counter
  titleInput.addEventListener('input', () => {
    const len = titleInput.value.trim().length;
    charCounter.textContent = `${len} chars`;
  });

  // Toggle Context Accordion
  accordionToggle.addEventListener('click', () => {
    const isOpen = accordionBody.classList.toggle('open');
    accordionToggle.innerHTML = isOpen 
      ? '▲ Hide article context' 
      : '▼ Add article context / paragraph (optional)';
  });

  // Mode Selection Pills
  modePills.forEach(pill => {
    pill.addEventListener('click', () => {
      modePills.forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
      currentMode = pill.getAttribute('data-mode');
    });
  });

  // Quick Example Chips
  exampleChips.forEach(chip => {
    chip.addEventListener('click', () => {
      const text = chip.getAttribute('data-text');
      const context = chip.getAttribute('data-context') || '';
      
      titleInput.value = text;
      contextInput.value = context;
      charCounter.textContent = `${text.length} chars`;
      
      if (context && !accordionBody.classList.contains('open')) {
        accordionBody.classList.add('open');
        accordionToggle.innerHTML = '▲ Hide article context';
      }
      
      // Auto run verification
      triggerVerification();
    });
  });

  // Clear Form
  btnClear.addEventListener('click', () => {
    titleInput.value = '';
    contextInput.value = '';
    charCounter.textContent = '0 chars';
    resultsSection.style.display = 'none';
    lastVerifiedText = '';
    titleInput.focus();
  });

  // Keyboard shortcut: Ctrl + Enter
  window.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      triggerVerification();
    }
  });

  btnVerify.addEventListener('click', () => {
    triggerVerification();
  });

  // Verification Main Function
  async function triggerVerification() {
    const title = titleInput.value.trim();
    const context = contextInput.value.trim();

    if (!title) {
      titleInput.focus();
      titleInput.style.borderColor = 'var(--color-fake)';
      setTimeout(() => {
        titleInput.style.borderColor = '';
      }, 1500);
      return;
    }

    // Set Loading State
    setLoading(true);

    try {
      const response = await fetch('/api/verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: title,
          context: context,
          mode: currentMode,
          enable_fact_check: factCheckSwitch.checked
        })
      });

      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}`);
      }

      const data = await response.json();
      lastVerifiedText = (title + ' ' + context).trim();
      renderResults(data);
    } catch (err) {
      alert(`Verification Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }

  function setLoading(loading) {
    if (loading) {
      btnSpinner.style.display = 'inline-block';
      btnText.textContent = 'Analyzing Claim...';
      btnVerify.disabled = true;
    } else {
      btnSpinner.style.display = 'none';
      btnText.textContent = 'Verify Claim';
      btnVerify.disabled = false;
    }
  }

  // Render Verification Results
  function renderResults(data) {
    resultsSection.style.display = 'block';

    // 1. Verdict Card Styling
    verdictCard.className = `verdict-hero-card ${data.verdict_type}`;
    
    let icon = '⚪';
    if (data.verdict_type === 'real_verified' || data.verdict_type === 'real') {
      icon = '🟢';
    } else if (data.verdict_type === 'fake') {
      icon = '🔴';
    }

    verdictBadge.innerHTML = `${icon} ${data.verdict}`;
    verdictConfText.textContent = `${data.confidence}% Confidence`;
    verdictDesc.textContent = data.description;

    // 2. Dual Credibility Meters
    realMetricValue.textContent = `${data.real_probability}%`;
    fakeMetricValue.textContent = `${data.fake_probability}%`;
    
    // Smooth meter animation
    setTimeout(() => {
      realProgressFill.style.width = `${data.real_probability}%`;
      fakeProgressFill.style.width = `${data.fake_probability}%`;
    }, 50);

    // 3. Fact Checking Evidence Tab
    renderFactEvidence(data.fact_check);

    // 4. ML Vocabulary Analysis Tab
    renderMLAnalysis(data.ml_analysis, data);

    // Reset LIME tab for new query
    limeContainer.innerHTML = '<p style="color: var(--text-muted); font-size: 0.9rem;">Click the button below to generate deep linguistic feature importance using LIME.</p>';
    btnFetchLime.style.display = 'inline-flex';

    // Scroll smoothly to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function renderFactEvidence(fact) {
    if (!fact) {
      factEvidenceContainer.innerHTML = `
        <div class="fact-box" style="border-color: var(--border-subtle); background: rgba(255,255,255,0.02);">
          <h4 style="color: var(--text-muted); margin-bottom: 0.4rem;">No Live Knowledge Match</h4>
          <p style="color: var(--text-secondary); font-size: 0.9rem;">
            The factual database did not find direct reference documentation for this exact query. The ML stylometric engine evaluated linguistic credibility.
          </p>
        </div>
      `;
      return;
    }

    const badgeStatus = fact.is_verified 
      ? '<span style="color: #4ade80; font-weight: 700;">✅ Factually Verified Match</span>'
      : '<span style="color: #f59e0b; font-weight: 600;">ℹ️ Closest Reference Article</span>';

    const keywords = fact.matched_keywords && fact.matched_keywords.length > 0
      ? fact.matched_keywords.map(k => `<span class="token-pill">${k}</span>`).join(' ')
      : '<span style="color: var(--text-muted);">None</span>';

    factEvidenceContainer.innerHTML = `
      <div class="fact-box">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
          <a href="${fact.page_url}" target="_blank" class="fact-title-link">
            📖 ${fact.title} (Wikipedia) ↗
          </a>
          <div>${badgeStatus}</div>
        </div>
        <p class="fact-excerpt">${fact.extract}</p>
        <div style="font-size: 0.85rem; color: var(--text-muted);">
          <strong>Matched Claim Concepts:</strong> ${keywords}
          <span style="margin-left: 0.75rem; color: var(--accent-cyan); font-weight: 600;">${Math.round(fact.match_ratio * 100)}% Keyword Alignment</span>
        </div>
      </div>
    `;
  }

  function renderMLAnalysis(ml, data) {
    // Recognized Tokens
    if (ml.matched_tokens && ml.matched_tokens.length > 0) {
      recognizedTokensList.innerHTML = ml.matched_tokens
        .map(t => `<span class="token-pill">${t}</span>`)
        .join('');
    } else {
      recognizedTokensList.innerHTML = '<span style="color: var(--text-muted); font-size: 0.85rem;">None</span>';
    }

    // Out of Vocabulary Tokens
    if (ml.oov_tokens && ml.oov_tokens.length > 0) {
      oovTokensList.innerHTML = ml.oov_tokens
        .map(t => `<span class="token-pill oov">${t}</span>`)
        .join('');
    } else {
      oovTokensList.innerHTML = '<span style="color: var(--text-muted); font-size: 0.85rem;">None (all words matched)</span>';
    }

    // Raw ML Stats
    rawStatsContainer.innerHTML = `
      <div style="font-size: 0.85rem; color: var(--text-secondary); line-height: 1.8;">
        <div>• <strong>Raw Model Probability:</strong> Real: ${data.raw_real_probability}% | Fake: ${data.raw_fake_probability}%</div>
        <div>• <strong>Calibrated Probability:</strong> Real: ${data.calibrated_real_probability}%</div>
        <div>• <strong>Active Mode:</strong> ${currentMode === 'headline' ? 'Headline Mode (Calibrated)' : 'Full Article Mode (Raw 50%)'}</div>
        <div>• <strong>Total Tokens Analyzed:</strong> ${ml.total_tokens_count} (${ml.matched_tokens_count} recognized in 7,000 TF-IDF features)</div>
      </div>
    `;
  }

  // LIME Explanation Fetcher
  btnFetchLime.addEventListener('click', async () => {
    if (!lastVerifiedText) return;

    btnFetchLime.disabled = true;
    limeSpinner.style.display = 'inline-block';

    try {
      const response = await fetch('/api/explain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: lastVerifiedText })
      });

      if (!response.ok) {
        throw new Error('Could not generate LIME explanation');
      }

      const data = await response.json();
      renderLimeFeatures(data.features);
    } catch (err) {
      limeContainer.innerHTML = `<p style="color: var(--color-fake); font-size: 0.9rem;">Failed to generate LIME analysis: ${err.message}</p>`;
    } finally {
      btnFetchLime.disabled = false;
      limeSpinner.style.display = 'none';
    }
  });

  function renderLimeFeatures(features) {
    if (!features || features.length === 0) {
      limeContainer.innerHTML = '<p style="color: var(--text-muted);">No significant feature contributions found.</p>';
      return;
    }

    limeContainer.innerHTML = features.map(f => {
      const isReal = f.type === 'real';
      const sign = isReal ? '+' : '';
      return `
        <div class="lime-word-card ${f.type}" title="${isReal ? 'Supports Real' : 'Supports Fake'}">
          <span>${f.word}</span>
          <span class="lime-score">${sign}${f.weight}</span>
        </div>
      `;
    }).join('');
  }

  // Tabs Switching Logic
  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      tabButtons.forEach(b => b.classList.remove('active'));
      tabPanes.forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const targetId = btn.getAttribute('data-tab');
      document.getElementById(targetId).classList.add('active');
    });
  });

  // Load Model Metrics into Benchmark Tab
  async function loadMetrics() {
    try {
      const res = await fetch('/api/metrics');
      if (!res.ok) return;
      const data = await res.json();
      
      const metricsSummary = document.getElementById('metricsSummary');
      if (metricsSummary) {
        metricsSummary.innerHTML = `
          <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem;">
            <div style="background: rgba(255,255,255,0.03); padding: 0.8rem; border-radius: 8px; text-align: center; border: 1px solid var(--border-subtle);">
              <div style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase;">Accuracy</div>
              <div style="font-size: 1.4rem; font-weight: 700; color: #4ade80;">${data.accuracy}%</div>
            </div>
            <div style="background: rgba(255,255,255,0.03); padding: 0.8rem; border-radius: 8px; text-align: center; border: 1px solid var(--border-subtle);">
              <div style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase;">Precision</div>
              <div style="font-size: 1.4rem; font-weight: 700; color: #38bdf8;">${data.precision}%</div>
            </div>
            <div style="background: rgba(255,255,255,0.03); padding: 0.8rem; border-radius: 8px; text-align: center; border: 1px solid var(--border-subtle);">
              <div style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase;">Recall</div>
              <div style="font-size: 1.4rem; font-weight: 700; color: #a78bfa;">${data.recall}%</div>
            </div>
            <div style="background: rgba(255,255,255,0.03); padding: 0.8rem; border-radius: 8px; text-align: center; border: 1px solid var(--border-subtle);">
              <div style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase;">F1 Score</div>
              <div style="font-size: 1.4rem; font-weight: 700; color: #f43f5e;">${data.f1_score}%</div>
            </div>
          </div>
        `;
      }
    } catch (_) {}
  }

  loadMetrics();
});
