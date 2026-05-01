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

      if (el.getAttribute("role") !== "tablist") {
        const nestedTablist = el.querySelector('[role="tablist"]');
        if (nestedTablist) return { container: nestedTablist, matchedSelector: selector };
      }

      return { container: el, matchedSelector: selector };
    }

    return { container: null, matchedSelector: null };
  }

  function getSelectedButton(container) {
    let selectedBtn = container.querySelector('button[role="tab"][aria-selected="true"]');
    if (selectedBtn) return selectedBtn;

    const tabs = container.querySelectorAll('button[role="tab"]');
    return Array.from(tabs).find((btn) =>
      Array.from(btn.classList).some((cls) => cls.startsWith("selected-"))
    );
  }

  function getTimeframe(btn) {
    if (!btn) return null;
    return (
      btn.getAttribute("data-overflow-tooltip-text")?.trim() ||
      btn.textContent?.trim() ||
      btn.id ||
      null
    );
  }

  function findAnnualButton(container) {
    // Primary: stable id from your examples.
    let annualBtn = container.querySelector('button[role="tab"]#FY');
    if (annualBtn) return annualBtn;

    // Fallback: tooltip/value text.
    const tabs = container.querySelectorAll('button[role="tab"]');
    annualBtn = Array.from(tabs).find((btn) => {
      const tooltip = btn.getAttribute("data-overflow-tooltip-text")?.trim().toLowerCase();
      const text = btn.textContent?.trim().toLowerCase();
      return tooltip === "annual" || text === "annual";
    });

    return annualBtn || null;
  }

  function waitForAnnualSelected(container, timeoutMs = 3000) {
    return new Promise((resolve) => {
      const start = Date.now();
      const timer = setInterval(() => {
        const selected = getSelectedButton(container);
        const timeframe = getTimeframe(selected)?.toLowerCase();
        if (timeframe === "annual" || Date.now() - start >= timeoutMs) {
          clearInterval(timer);
          resolve(timeframe === "annual");
        }
      }, 100);
    });
  }

  async function run() {
    const { container, matchedSelector } = findBestContainer();
    if (!container) {
      const result = {
        ok: false,
        changed: false,
        reason: "Could not find timeframe tab container.",
        attemptedSelectors: CANDIDATE_SELECTORS,
      };
      console.warn(result.reason);
      return result;
    }

    const selectedBefore = getSelectedButton(container);
    const timeframeBefore = getTimeframe(selectedBefore);
    const annualBtn = findAnnualButton(container);

    if (!annualBtn) {
      const result = {
        ok: false,
        changed: false,
        reason: "Could not find an Annual timeframe button.",
        matchedSelector,
        timeframeBefore,
      };
      console.warn(result.reason);
      return result;
    }

    const alreadyAnnual =
      annualBtn.getAttribute("aria-selected") === "true" ||
      getTimeframe(annualBtn)?.toLowerCase() === (timeframeBefore || "").toLowerCase();

    if (!alreadyAnnual) {
      annualBtn.click();
      await waitForAnnualSelected(container);
    }

    const selectedAfter = getSelectedButton(container);
    const timeframeAfter = getTimeframe(selectedAfter);
    const isAnnualNow = (timeframeAfter || "").toLowerCase() === "annual";

    const result = {
      ok: isAnnualNow,
      changed: !alreadyAnnual,
      matchedSelector,
      timeframeBefore: timeframeBefore || null,
      timeframeAfter: timeframeAfter || null,
      selectedButtonId: selectedAfter?.id || null,
    };

    if (result.ok) {
      console.log(`Timeframe is now: ${result.timeframeAfter}`);
    } else {
      console.warn("Attempted to switch to Annual, but current selection is:", result.timeframeAfter);
    }

    return result;
  }

  return run();
})();
