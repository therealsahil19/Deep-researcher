# Palette's Journal

## 2025-02-18 - [Destructive Action Confirmation]
**Learning:** Users can accidentally wipe their entire session history with a single click. Destructive actions in Streamlit require explicit confirmation patterns since there are no native "confirm" dialogs.
**Action:** Always wrap destructive actions (like "Clear History") in a confirmation state or popover to prevent data loss.
