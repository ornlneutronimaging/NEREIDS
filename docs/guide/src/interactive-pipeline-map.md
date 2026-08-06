# Pipeline Map (interactive)

The **[interactive pipeline map](pipeline-map.html)** is the physics-first design
contract for NEREIDS and the recommended starting point for understanding the
analysis: the physics spine (resonances → Doppler → Beer-Lambert → time-of-flight
→ instrument kernel → counts), the four fitting modes run live on-screen, the
data taxonomy, and the API contract the code is reconciled against.

It is a standalone HTML page (open it in a browser; it needs no server). The
document is the agreed refactor gate — its closing rule, that no public
function is added without updating the map first, was signed off with the
Phase 1–4 contract and gains automated CI enforcement in the final acceptance
wave of the refactor. Its review-record tab preserves the adjudicated history
of every correction.
