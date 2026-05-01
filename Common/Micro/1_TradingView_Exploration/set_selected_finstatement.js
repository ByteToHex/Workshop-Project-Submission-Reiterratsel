(() => {
  // Paste into browser console, then call:
  // setSelectedFinStatement(0) // Overview
  // setSelectedFinStatement(1) // Statements
  // ...
  // setSelectedFinStatement(5) // Revenue

  const INDEX_TO_TAB_ID = {
    0: "overview",
    1: "statements",
    2: "statistics",
    3: "dividends",
    4: "earnings",
    5: "revenue",
  };

  // Prefer #financials-tabs so we never grab #financials-page-statements-tabs (same data-name wrapper).
  const FALLBACK_SELECTORS = [
    '[data-name="round-tabs-anchors"] [role="tablist"]',
    '[data-name="round-tabs-anchors"]',
    ".scrollWrap-vgCB17hK [role='tablist']",
    "[role='tablist']",
  ];

  function findBestContainer() {
    const byId = document.getElementById("financials-tabs");
    if (byId?.getAttribute("role") === "tablist") {
      return { container: byId, matchedSelector: "#financials-tabs" };
    }

    for (const el of document.querySelectorAll(
      '[data-name="round-tabs-anchors"] [role="tablist"]'
    )) {
      if (el.id === "financials-tabs") {
        return {
          container: el,
          matchedSelector:
            '[data-name="round-tabs-anchors"] [role="tablist"]#financials-tabs',
        };
      }
    }

    for (const selector of FALLBACK_SELECTORS) {
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

  function getLabel(tab) {
    return (
      tab?.getAttribute("data-overflow-tooltip-text")?.trim() ||
      tab?.textContent?.trim() ||
      tab?.id ||
      null
    );
  }

  function isSelected(tab) {
    if (!tab) return false;
    if (tab.getAttribute("aria-selected") === "true") return true;
    return Array.from(tab.classList).some((cls) => cls.startsWith("selected-"));
  }

  function getSelectedTab(container) {
    const tabs = Array.from(container.querySelectorAll('a[role="tab"], button[role="tab"]'));
    return tabs.find((tab) => isSelected(tab)) || null;
  }

  function waitForSelection(container, expectedId, timeoutMs = 4000) {
    return new Promise((resolve) => {
      const start = Date.now();
      const timer = setInterval(() => {
        const selected = getSelectedTab(container);
        const done = selected?.id === expectedId || Date.now() - start >= timeoutMs;
        if (done) {
          clearInterval(timer);
          resolve(selected?.id === expectedId);
        }
      }, 100);
    });
  }

  async function setSelectedFinStatement(targetIndex) {
    const parsed = Number(targetIndex);
    if (!Number.isInteger(parsed) || !(parsed in INDEX_TO_TAB_ID)) {
      const result = {
        ok: false,
        changed: false,
        reason: "Invalid argument. Use integer 0..5.",
        validMap: INDEX_TO_TAB_ID,
      };
      console.warn(result.reason, result.validMap);
      return result;
    }

    const targetId = INDEX_TO_TAB_ID[parsed];
    const { container, matchedSelector } = findBestContainer();
    if (!container) {
      const result = {
        ok: false,
        changed: false,
        reason: "Could not find financial statement tab container.",
        attemptedSelectors: ["#financials-tabs", ...FALLBACK_SELECTORS],
      };
      console.warn(result.reason);
      return result;
    }

    const selectedBefore = getSelectedTab(container);
    const before = {
      id: selectedBefore?.id || null,
      label: getLabel(selectedBefore),
    };

    const targetBtn = container.querySelector(`#${targetId}[role="tab"]`) ||
      container.querySelector(`[role="tab"]#${targetId}`);
    if (!targetBtn) {
      const result = {
        ok: false,
        changed: false,
        reason: `Target tab not found: ${targetId}`,
        matchedSelector,
        before,
      };
      console.warn(result.reason);
      return result;
    }

    let changed = false;
    if (!isSelected(targetBtn)) {
      targetBtn.click();
      changed = true;
      await waitForSelection(container, targetId);
    }

    const selectedAfter = getSelectedTab(container);
    const after = {
      id: selectedAfter?.id || null,
      label: getLabel(selectedAfter),
    };

    const result = {
      ok: after.id === targetId,
      changed,
      matchedSelector,
      requestedIndex: parsed,
      requestedTabId: targetId,
      before,
      after,
    };

    if (result.ok) {
      console.log(`Financial statement selected: ${after.label} (${after.id})`);
    } else {
      console.warn("Selection may not have updated as expected.", result);
    }

    return result;
  }

  // Expose function for repeated calls without re-pasting the script.
  window.setSelectedFinStatement = setSelectedFinStatement;
  console.log("Ready. Call setSelectedFinStatement(index) with index 0..5.");
})();
