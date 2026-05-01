/**
 * Financials tree expanders use paired CSS-module classes: arrow-{hash} / opened-{hash}.
 * The hash changes on each TV build; match by prefix, not fixed class names.
 */
function queryClosedExpandArrowElements() {
  const closed = [];
  for (const el of document.querySelectorAll("[class*='arrow-']")) {
    let hash = null;
    for (const c of el.classList) {
      if (c.startsWith("arrow-") && c.length > "arrow-".length) {
        hash = c.slice("arrow-".length);
        break;
      }
    }
    if (!hash) continue;
    if (el.classList.contains(`opened-${hash}`)) continue;
    closed.push(el);
  }
  return closed;
}

(function expandAll() {
  const closedArrows = queryClosedExpandArrowElements();

  if (closedArrows.length > 0) {
    closedArrows.forEach((el) => el.click());
    setTimeout(expandAll, 300);
  } else {
    console.log("All levels expanded.");
  }
})();
