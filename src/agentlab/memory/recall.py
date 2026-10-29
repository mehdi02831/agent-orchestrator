def build_note(hits, threshold: float) -> str:
    """One-line memory note from the best hit at or above the threshold, or '' if none qualifies."""
    good = [h for h in hits if h["similarity"] >= threshold]
    if not good:
        return ""
    h = max(good, key=lambda x: x["similarity"])
    return ("Memory (may be wrong or outdated; the rules above still apply): an earlier similar question was "
            f'"{h["prompt"]}" and the answer given was "{h["answer"]}".')
