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

    // 2.5 Prominent AI Explanation & Diagnostic Evidence
    renderExplanation(data.explanation, data);

    // 3. Fact Checking Evidence Tab
    renderFactEvidence(data.fact_check);

    // 4. ML Vocabulary Analysis Tab
    renderMLAnalysis(data.ml_analysis, data);

    // 5. Dynamic Claim Confusion Matrix & Dynamic ROC Curve
    renderClaimEvaluation(data.claim_evaluation, data.stylometrics, data);

    // Reset LIME tab for new query
    limeContainer.innerHTML = '<p style="color: var(--text-muted); font-size: 0.9rem;">Click the button below to generate deep linguistic feature importance using LIME.</p>';
    btnFetchLime.style.display = 'inline-flex';

    // Scroll smoothly to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // Global store for threshold simulation
  let currentClaimRealProb = 0.50;

  function renderClaimEvaluation(claimEval, stylometrics, data) {
    if (!claimEval) return;

    currentClaimRealProb = data.real_probability / 100.0;

    // 1. Update Claim Locator Banner
    const quadrantText = document.getElementById('claimQuadrantText');
    const quadrantDetail = document.getElementById('claimQuadrantDetail');
    const domainText = document.getElementById('claimDomainText');
    const marginText = document.getElementById('claimMarginText');
    const likelihoodText = document.getElementById('claimLikelihoodText');

    if (quadrantText) quadrantText.textContent = `${claimEval.quadrant_name} (${claimEval.quadrant})`;
    if (quadrantDetail) quadrantDetail.textContent = claimEval.detail;
    if (domainText && stylometrics) domainText.textContent = stylometrics.domain;
    if (marginText) {
      const marginSign = claimEval.decision_margin >= 0 ? '+' : '';
      marginText.textContent = `${marginSign}${claimEval.decision_margin}%`;
      marginText.style.color = claimEval.decision_margin >= 0 ? '#4ade80' : '#f43f5e';
    }
    if (likelihoodText) likelihoodText.textContent = claimEval.likelihood_ratio;

    // 2. Update Forensic Stylometrics
    if (stylometrics) {
      const sensVal = document.getElementById('forensicSensationalism');
      const sensDesc = document.getElementById('forensicSensationalismDesc');
      const ttrVal = document.getElementById('forensicTTR');
      const lenVal = document.getElementById('forensicLength');
      const charsVal = document.getElementById('forensicChars');
      const punctVal = document.getElementById('forensicPunctuation');

      if (sensVal) {
        sensVal.textContent = `${stylometrics.sensationalism_index} / 100`;
        sensVal.style.color = stylometrics.sensationalism_index > 50 ? '#f43f5e' : (stylometrics.sensationalism_index > 20 ? '#fbbf24' : '#4ade80');
      }
      if (sensDesc) {
        sensDesc.textContent = stylometrics.sensationalism_index > 50 
          ? 'High sensationalism & viral clickbait triggers' 
          : (stylometrics.sensationalism_index > 20 ? 'Moderate rhetorical tone' : 'Clean journalistic & objective language');
      }
      if (ttrVal) ttrVal.textContent = `${stylometrics.lexical_diversity}%`;
      if (lenVal) lenVal.textContent = `${stylometrics.word_count} words`;
      if (charsVal) charsVal.textContent = `${stylometrics.char_count} chars • ${stylometrics.sentence_count} sentence(s)`;
      if (punctVal) {
        const pTotal = stylometrics.exclamations + stylometrics.all_caps_count;
        punctVal.textContent = pTotal === 0 ? 'Clean' : `${pTotal} flags`;
        punctVal.style.color = pTotal === 0 ? '#4ade80' : '#f43f5e';
      }
    }

    // 3. Highlight Matching Quadrant in the 2x2 Confusion Matrix
    ['cmCellTN', 'cmCellFP', 'cmCellFN', 'cmCellTP'].forEach(id => {
      const el = document.getElementById(id);
      if (el) {
        el.classList.remove('active-claim-quadrant');
        const badge = el.querySelector('.claim-pinpoint-badge');
        if (badge) badge.remove();
      }
    });

    const activeCellId = 'cmCell' + claimEval.quadrant;
    const targetCell = document.getElementById(activeCellId);
    if (targetCell) {
      targetCell.classList.add('active-claim-quadrant');
      const badge = document.createElement('div');
      badge.className = 'claim-pinpoint-badge';
      badge.innerHTML = '📍 THIS ARTICLE';
      targetCell.appendChild(badge);
    }

    // 4. Update Dynamic ROC Operating Point on SVG Curve
    const rocDot = document.getElementById('claimRocDot');
    const rocFpr = document.getElementById('rocClaimFpr');
    const rocTpr = document.getElementById('rocClaimTpr');
    const rocStatusBadge = document.getElementById('claimRocStatusBadge');

    if (rocDot && claimEval.roc_point) {
      const fprClamped = Math.min(100, Math.max(0, claimEval.roc_point.fpr));
      const tprClamped = Math.min(100, Math.max(0, claimEval.roc_point.tpr));

      const cx = 35 + (fprClamped / 100) * 270;
      const cy = 205 - (tprClamped / 100) * 190;

      rocDot.setAttribute('cx', cx.toFixed(1));
      rocDot.setAttribute('cy', cy.toFixed(1));
      rocDot.setAttribute('fill', data.real_probability >= 50 ? '#22c55e' : '#f43f5e');

      if (rocFpr) rocFpr.textContent = `${claimEval.roc_point.fpr}%`;
      if (rocTpr) rocTpr.textContent = `${claimEval.roc_point.tpr}%`;
      if (rocStatusBadge) rocStatusBadge.textContent = claimEval.roc_point.status;
    }

    // 5. Reset Decision Threshold Slider to 0.50 default and sync
    const slider = document.getElementById('thresholdSlider');
    if (slider) slider.value = 0.50;
    updateThresholdSimulation(0.50);
  }

  // Interactive Decision Cutoff Simulator
  const thresholdSlider = document.getElementById('thresholdSlider');
  const sliderValBox = document.getElementById('sliderValBox');
  const simCutoffLabel = document.getElementById('simCutoffLabel');
  const simVerdictBadge = document.getElementById('simVerdictBadge');
  const simMarginVal = document.getElementById('simMarginVal');

  if (thresholdSlider) {
    thresholdSlider.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      updateThresholdSimulation(val);
    });
  }

  function updateThresholdSimulation(tau) {
    if (sliderValBox) sliderValBox.textContent = `τ = ${tau.toFixed(2)}`;
    if (simCutoffLabel) simCutoffLabel.textContent = tau.toFixed(2);

    const isReal = currentClaimRealProb >= tau;
    const margin = Math.round((currentClaimRealProb - tau) * 1000) / 10;
    const sign = margin >= 0 ? '+' : '';

    if (simVerdictBadge) {
      simVerdictBadge.textContent = isReal ? 'Real News (Accepted)' : 'Fake / Unverified (Flagged)';
      simVerdictBadge.style.color = isReal ? '#4ade80' : '#f43f5e';
    }

    if (simMarginVal) {
      simMarginVal.textContent = `${sign}${margin}%`;
      simMarginVal.style.color = margin >= 0 ? '#38bdf8' : '#fb7185';
    }
  }

  // Print Verification Dossier
  const btnPrintReport = document.getElementById('btnPrintReport');
  if (btnPrintReport) {
    btnPrintReport.addEventListener('click', () => {
      window.print();
    });
  }

  function renderExplanation(exp, data) {
    const expCard = document.getElementById('explanationCard');
    if (!expCard || !exp) return;

    const isFake = data.verdict_type === 'fake';
    const isVerified = data.verdict_type === 'real_verified';
    const isReal = data.verdict_type === 'real' || isVerified;

    // Accent coloring
    const accentColor = isFake ? '#f43f5e' : (isReal ? '#22c55e' : '#f59e0b');
    expCard.style.borderLeftColor = accentColor;

    const keywordEl = document.getElementById('expVerdictKeyword');
    if (keywordEl) {
      keywordEl.textContent = isFake ? 'Fake' : (isReal ? 'Real' : 'Inconclusive');
      keywordEl.style.color = accentColor;
    }

    const badgePill = document.getElementById('expVerdictBadgePill');
    if (badgePill) {
      badgePill.textContent = `${data.confidence}% Confidence`;
      badgePill.style.background = isFake ? 'rgba(244, 63, 94, 0.18)' : 'rgba(34, 197, 94, 0.18)';
      badgePill.style.color = isFake ? '#fb7185' : '#4ade80';
      badgePill.style.borderColor = isFake ? 'rgba(244, 63, 94, 0.35)' : 'rgba(34, 197, 94, 0.35)';
    }

    const summaryText = document.getElementById('expSummaryText');
    if (summaryText) {
      summaryText.textContent = exp.summary;
    }

    // 3 Diagnostic Pillars
    const factIcon = document.getElementById('expFactIcon');
    const factDesc = document.getElementById('expFactDesc');
    if (factIcon) factIcon.textContent = isVerified ? '✅' : (isFake ? '❌' : 'ℹ️');
    if (factDesc) factDesc.textContent = exp.fact_reason;

    const mlIcon = document.getElementById('expMlIcon');
    const mlDesc = document.getElementById('expMlDesc');
    if (mlIcon) mlIcon.textContent = isFake ? '⚠️' : '🎯';
    if (mlDesc) mlDesc.textContent = exp.ml_reason;

    const toneIcon = document.getElementById('expToneIcon');
    const toneDesc = document.getElementById('expToneDesc');
    if (toneIcon) toneIcon.textContent = '🔬';
    if (toneDesc) toneDesc.textContent = exp.tone_reason;

    // Influential Word Chips
    const chipsWrap = document.getElementById('expChipsWrap');
    if (chipsWrap) {
      if (exp.token_impacts && exp.token_impacts.length > 0) {
        chipsWrap.innerHTML = exp.token_impacts.map(t => {
          const isWordReal = t.type === 'real';
          const bg = isWordReal ? 'rgba(34, 197, 94, 0.15)' : 'rgba(244, 63, 94, 0.15)';
          const textCol = isWordReal ? '#4ade80' : '#fb7185';
          const borderCol = isWordReal ? 'rgba(34, 197, 94, 0.35)' : 'rgba(244, 63, 94, 0.35)';
          const icon = isWordReal ? '🟢' : '🔴';
          return `
            <div class="exp-word-chip" style="display: inline-flex; align-items: center; gap: 0.35rem; padding: 0.32rem 0.7rem; border-radius: 999px; font-size: 0.8rem; background: ${bg}; color: ${textCol}; border: 1px solid ${borderCol};">
              <span>${icon} <strong>${t.word}</strong></span>
              <span style="opacity: 0.85; font-size: 0.72rem; padding: 1px 5px; background: rgba(0,0,0,0.35); border-radius: 4px;">${t.label}</span>
            </div>
          `;
        }).join('');
      } else {
        chipsWrap.innerHTML = '<span style="color: var(--text-muted); font-size: 0.85rem;">No strong vocabulary weights found for this short input.</span>';
      }
    }
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

  // Load Model Metrics into Evolution & Benchmarks
  async function loadMetrics() {
    try {
      const res = await fetch('/api/metrics');
      if (!res.ok) return;
      const data = await res.json();
      
      // Update Evolution Table with server data
      const tbody = document.getElementById('evolutionTableBody');
      if (tbody && data.model_evolution && data.model_evolution.length > 0) {
        tbody.innerHTML = data.model_evolution.map(m => {
          const isChampion = m.status.includes('Champion');
          const stageBadge = isChampion 
            ? `<span class="evolution-stage-badge" style="background: rgba(34, 197, 94, 0.25); color: #4ade80;">${m.stage}</span>`
            : `<span class="evolution-stage-badge">${m.stage}</span>`;
          const champTag = isChampion ? `<span class="champion-tag">Champion</span>` : '';
          const rowClass = isChampion ? 'champion-row' : '';
          const accColor = isChampion ? '#4ade80' : (m.accuracy > 99.2 ? '#a78bfa' : (m.accuracy > 98 ? '#38bdf8' : '#94a3b8'));
          
          return `
            <tr class="${rowClass}">
              <td>${stageBadge}</td>
              <td><strong>${m.model}</strong> ${champTag}</td>
              <td><span style="color: var(--text-muted); font-size: 0.8rem;">${m.category}</span></td>
              <td><strong style="color: ${accColor}; font-size: 0.95rem;">${m.accuracy}%</strong></td>
              <td>${m.precision}%</td>
              <td>${m.recall}%</td>
              <td>${m.f1_score}%</td>
              <td style="color: ${isChampion ? '#86efac' : 'var(--text-muted)'}; font-size: 0.8rem;">${m.notes}</td>
            </tr>
          `;
        }).join('');
      }
    } catch (err) {
      console.warn('Metrics load error:', err);
    }
  }

  loadMetrics();
});
