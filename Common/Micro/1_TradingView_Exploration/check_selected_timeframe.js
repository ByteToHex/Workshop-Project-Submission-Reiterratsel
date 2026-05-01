(() => {
  // Paste this entire block into browser console on the target page.
  const CANDIDATE_SELECTORS = [
    // "#financials-page-tabs",
    // "#financials-earnings-tabs",
    '[data-name="square-tabs-buttons"] [role="tablist"]',
    // '[data-name="square-tabs-buttons"]',
    // ".scrollWrap-mf1FlhVw [role='tablist']",
    // ".scrollWrap-mf1FlhVw",
    // "[role='tablist']",
  ];

  function findBestContainer() {
    for (const selector of CANDIDATE_SELECTORS) {
      const el = document.querySelector(selector);
      if (!el) continue;

      // If this is a wrapper div, use its inner tablist if available.
      if (el.getAttribute("role") !== "tablist") {
        const nestedTablist = el.querySelector('[role="tablist"]');
        if (nestedTablist) return { container: nestedTablist, matchedSelector: selector };
      }

      return { container: el, matchedSelector: selector };
    }

    return { container: null, matchedSelector: null };
  }

  function detectSelectedTimeframe() {
    const { container, matchedSelector } = findBestContainer();
    if (!container) {
      return {
        ok: false,
        reason: "Could not find timeframe tab container.",
        attemptedSelectors: CANDIDATE_SELECTORS,
      };
    }

    // Primary: semantic state marker.
    let selectedBtn = container.querySelector('button[role="tab"][aria-selected="true"]');

    // Fallback: TradingView selected class pattern.
    if (!selectedBtn) {
      const tabs = container.querySelectorAll('button[role="tab"]');
      selectedBtn = Array.from(tabs).find((btn) =>
        Array.from(btn.classList).some((cls) => cls.startsWith("selected-"))
      );
    }

    if (!selectedBtn) {
      return {
        ok: false,
        reason: "No selected timeframe button detected.",
      };
    }

    const timeframe =
      selectedBtn.getAttribute("data-overflow-tooltip-text")?.trim() ||
      selectedBtn.textContent?.trim() ||
      selectedBtn.id ||
      null;

    return {
      ok: true,
      timeframe,
      buttonId: selectedBtn.id || null,
      ariaSelected: selectedBtn.getAttribute("aria-selected"),
      matchedSelector,
      element: selectedBtn,
    };
  }

  const result = detectSelectedTimeframe();
  if (result.ok) {
    console.log(`Selected timeframe: ${result.timeframe}`);
  } else {
    console.warn(result.reason);
  }
  return result;
})();
