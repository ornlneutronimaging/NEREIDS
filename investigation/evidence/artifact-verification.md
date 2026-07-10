# Investigation artifact verification

Command:

```text
pixi run python investigation/verify_artifacts.py
```

Exit status 0. Output:

```text
required_artifacts=5
python_files_compiled=13
contract_checked_requirements=12
archive_inventory_rows=71
relative_links_checked=51
verification=PASS
```

This is a mechanical check of required/nonempty artifacts, in-memory Python
syntax compilation, contract checkbox count, exact ZIP inventory-row count,
relative Markdown links, and absence of the prior temporary-script/placeholder
command patterns. It does not validate scientific conclusions; those rely on
the separate probes and adversarial review.
