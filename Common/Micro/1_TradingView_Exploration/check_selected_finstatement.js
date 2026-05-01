(() => {
  // Paste this entire block into browser console on the target page.
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
      tab.getAttribute("data-overflow-tooltip-text")?.trim() ||
      tab.textContent?.trim() ||
      tab.id ||
      null
    );
  }

  function isSelected(tab) {
    if (tab.getAttribute("aria-selected") === "true") return true;
    return Array.from(tab.classList).some((cls) => cls.startsWith("selected-"));
  }

  function detectSelectedFinStatement() {
    const { container, matchedSelector } = findBestContainer();
    if (!container) {
      return {
        ok: false,
        reason: "Could not find financial statement tab container.",
        attemptedSelectors: ["#financials-tabs", ...FALLBACK_SELECTORS],
      };
    }

    const tabs = Array.from(container.querySelectorAll('a[role="tab"], button[role="tab"]'));
    if (!tabs.length) {
      return {
        ok: false,
        reason: "No tab elements found in container.",
        matchedSelector,
      };
    }

    const tabStates = tabs.map((tab) => ({
      id: tab.id || null,
      label: getLabel(tab),
      selected: isSelected(tab),
      ariaSelected: tab.getAttribute("aria-selected"),
    }));

    const selectedTab = tabStates.find((t) => t.selected) || null;

    return {
      ok: true,
      matchedSelector,
      selected: selectedTab,
      tabs: tabStates,
    };
  }

  const result = detectSelectedFinStatement();

  if (!result.ok) {
    console.warn(result.reason);
    return result;
  }

  if (result.selected) {
    console.log(`Selected financial statement tab: ${result.selected.label} (${result.selected.id})`);
  } else {
    console.warn("No selected tab detected.");
  }

  console.table(result.tabs);
  return result;
})();
